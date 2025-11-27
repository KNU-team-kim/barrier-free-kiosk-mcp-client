# client.py

import json
import uvicorn
import os
import re
from dotenv import load_dotenv
from typing import Any, List, Dict
from loguru import logger

from fastapi import FastAPI, WebSocket
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.shared.context import RequestContext
from mcp.client.streamable_http import streamablehttp_client

# .env 파일에서 환경 변수 로드
load_dotenv()

app = FastAPI()

# OpenAI 클라이언트 초기화
llm_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# 활성 웹소켓 연결을 관리하는 딕셔너리
WEBSOCKET_MANAGER: dict[str, WebSocket] = {}

# MCP 서버 주소 (환경 변수에서 로드)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")

# --- MCP Tool -> OpenAI Schema 변환 함수 ---
def to_openai_schema(tool) -> Dict[str, Any]:
    """MCP 도구 명세를 OpenAI API가 이해할 수 있는 JSON 스키마로 변환합니다."""
    raw_schema = (
        getattr(tool, "inputSchema", None)
        or getattr(tool, "input_schema", None)
        or getattr(tool, "parameters", None)
    )

    if raw_schema is None:
        schema: Dict[str, Any] = {"type": "object", "properties": {}}
    elif isinstance(raw_schema, dict):
        schema = raw_schema
    elif hasattr(raw_schema, "model_json_schema"):
        schema = raw_schema.model_json_schema()
    else:
        schema = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": to_camel_case(tool.name),
            "description": getattr(tool, "description", ""),
            "parameters": schema,
        }
    }


# --- Elicitation 콜백 ---
async def smart_elicitation_callback(
    context: RequestContext["ClientSession", Any],
    params: types.ElicitRequestParams,
):
    """MCP 서버로부터 Elicit 요청을 받았을 때 호출되는 콜백 함수."""
    data = json.loads(params.message)
    session_id = data["session_id"]
    
    user_message = None
    if data.get("retrieve_output", True):
        await WEBSOCKET_MANAGER[session_id].send_json(
            {"message": data["ai_message"], "step_name": data["step_name"]}
        )
        user_response = await WEBSOCKET_MANAGER[session_id].receive_json()
        user_message = user_response.get("message")
    else:
        await WEBSOCKET_MANAGER[session_id].send_json(
            {"message": data["ai_message"], "step_name": data["step_name"], "data": data.get("data")}
        )
    
    return types.ElicitResult(
        action="accept",
        content={"user_message": user_message},
    )

# --- 이름 변환 유틸리티 함수 ---
def to_kebab_case(name: str) -> str:
    """CamelCase 문자열을 kebab-case로 변환합니다."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1).lower()

def to_camel_case(name: str) -> str:
    """kebab-case 또는 snake_case 문자열을 CamelCase로 변환합니다."""
    name = name.replace('-', '_')
    return ''.join(word.capitalize() for word in name.split('_'))

def looks_like_tool_call(data: dict) -> bool:
    return (
        isinstance(data, dict)
        and "name" in data
        and "arguments" in data
    )

# --- 웹소켓 엔드포인트 ---
@app.websocket("/conversation")
async def conversation(websocket: WebSocket):
    """무인민원발급기 대화를 위한 웹소켓 엔드포인트"""
    await websocket.accept()
    session_id = None

    try:
        async with streamablehttp_client(url=MCP_SERVER_URL) as (read_stream, write_stream, get_session_id):
            async with ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
                elicitation_callback=smart_elicitation_callback,
            ) as session:
                await session.initialize()
                
                session_id = get_session_id()
                WEBSOCKET_MANAGER[session_id] = websocket

                tool_list_response = await session.list_tools()
                actual_tools = tool_list_response.tools
                
                if not actual_tools:
                    logger.error("MCP 서버에서 사용 가능한 Tool을 찾을 수 없습니다.")
                    await websocket.send_json({"message": "오류: 서버에 설정된 서비스가 없습니다.", "step_name": "error"})
                    return

                tool_schemas = [to_openai_schema(tool) for tool in actual_tools]

                # --- 1. 대화 기록(messages)을 루프 외부에 선언 ---
                # 시스템 프롬프트를 처음에 설정하여 대화의 전체 맥락을 관리
                messages: List[ChatCompletionMessageParam] = [
                    {"role": "system", "content": (
                        "당신은 AI 무인민원발급기 안내원입니다. 사용자의 요청을 분석하여, 필요하다면 제공된 Tool을 호출하고 그 결과를 바탕으로 사용자에게 친절하게 최종 답변을 생성해주세요."
                        "아래와 같은 규칙을 따라주세요: "
                        "- 지원하지 않는 서비스에 대한 문의는 명확하게 불가능하다고 답변해주세요. "
                        "- 메시지를 음성으로 변환해야하기 때문에, 모든 대화는 이모티콘이나 특수문자(개행문자 포함), 마크다운 문법 없이 반드시 대화하는 형식으로 이루어져야 합니다."
                    )},
                ]

                # --- 대화 시작 ---
                initial_message = "AI 무인민원발급기입니다. 어떤 서비스를 원하시나요?"
                await websocket.send_json({"message": initial_message, "step_name": "home"})
                
                # --- 2. 연속적인 대화를 위한 while 루프 유지 ---
                while True:
                    user_response = await websocket.receive_json()
                    user_message = user_response["message"]
                    
                    # 사용자의 새 메시지를 대화 기록에 추가
                    messages.append({"role": "user", "content": user_message})
                    
                    # --- 3. LLM에 1차 요청 (누적된 전체 대화 기록 전달) ---
                    response = await llm_client.chat.completions.create(
                        model="gpt-4.1",
                        messages=messages,
                        tools=tool_schemas,
                        tool_choice="auto"
                    )
                    
                    response_message = response.choices[0].message
                    tool_calls = response_message.tool_calls

                    # --- 4. LLM 응답 분석 및 분기 처리 ---
                    if not tool_calls:
                        # 4-1. Tool 호출이 없을 때: LLM 답변을 바로 반환하고 대화 기록에 추가
                        logger.info("No tool call. Returning direct response.")
                        direct_answer = response_message.content or "죄송합니다. 요청을 처리할 수 없습니다."

                        # 4-2. Tool Call 과정에서 content에 tool 정보가 들어오는 오류가 발생했을 때
                        try:
                            json.loads(direct_answer)
                            direct_answer = "죄송합니다. 다시 한 번 말씀해주세요."
                        except json.JSONDecodeError:
                            pass
                        
                        messages.append({"role": "assistant", "content": direct_answer}) # LLM 답변을 기록에 추가
                        await websocket.send_json({"message": direct_answer, "step_name": "home"})
                        continue # 다음 사용자 입력을 위해 루프 계속

                    # 4-2. Tool 호출이 있을 때: Tool 실행 및 2차 호출 진행
                    logger.info(f"LLM decided to call tools: {[tc.function.name for tc in tool_calls]}")
                    
                    for tool_call in tool_calls:
                        tool_name_to_call = to_kebab_case(tool_call.function.name)
                        
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse arguments: {tool_call.function.arguments}")
                        else:
                            arguments["session_id"] = session_id
                            result = await session.call_tool(name=tool_name_to_call, arguments=arguments)

                        messages.append({"role": "assistant", "content": "완료되었습니다."})

                        await websocket.send_json({"message": result.structuredContent["message"], "step_name": result.structuredContent["step_name"]})

                    # 루프는 계속되어 사용자의 다음 질문을 기다린다.

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        if websocket and not websocket.client_state.name == 'DISCONNECTED':
            await websocket.send_json({"message": f"오류가 발생했습니다: {e}", "step_name": "error"})
    finally:
        if session_id and session_id in WEBSOCKET_MANAGER:
            await WEBSOCKET_MANAGER[session_id].close()
            del WEBSOCKET_MANAGER[session_id]
            logger.info(f"Session {session_id} and websocket connection closed.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")