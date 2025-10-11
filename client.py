# client.py

import json
import uvicorn
from dotenv import load_dotenv
from typing import Any, Optional, Literal
from loguru import logger
import mcp.types as types
from fastapi import FastAPI, WebSocket
from mcp.client.session import ClientSession
from mcp.shared.context import RequestContext
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

# .env 파일에서 환경 변수 로드
load_dotenv()

app = FastAPI()

# 활성 웹소켓 연결을 관리하는 딕셔너리
WEBSOCKET_MANAGER: dict[str, WebSocket] = {}
# MCP 서버 주소
MCP_SERVER_URL = "http://localhost:8001/mcp"


# --- Pydantic 모델 및 LLM 체인 (클라이언트 측 라우팅용) ---
class ServiceRouterOutput(BaseModel):
    tool_name: Literal["issue_resident_registration", "process_move_in_declaration", "unsupported"] = Field(
        description="사용자의 요청에 가장 적합한 Tool의 이름."
    )
    ai_message: Optional[str] = Field(default=None, description="사용자에게 안내할 메시지 (Tool 호출 전)")


async def create_router_chain():
    """ 사용자의 의도를 분석하여 호출할 Tool을 결정하는 LLM 체인 생성 """
    system_prompt = """
    당신은 AI 무인민원발급기 안내원입니다. 사용자의 요청을 듣고 어떤 Tool을 실행해야 할지 결정해야 합니다.
    - '주민등록등본', '초본' 등 관련 단어가 있으면 'issue_resident_registration' Tool을 선택합니다.
    - '전입', '이사' 등 관련 단어가 있으면 'process_move_in_declaration' Tool을 선택합니다.
    - 지원하지 않는 서비스나 모호한 요청은 'unsupported'로 판단하고, 가능한 서비스를 안내하는 메시지를 생성합니다.

    JSON 형식으로만 응답해야 합니다.
    {{"tool_name": "issue_resident_registration"}}
    또는
    {{"tool_name": "unsupported", "ai_message": "현재 '주민등록초본 발급'과 '전입신고'만 가능합니다. 어떤 서비스를 원하시나요?"}}
    """
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ServiceRouterOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{user_message}")
    ])
    return prompt | llm | parser


# --- Elicitation 콜백 ---
async def smart_elicitation_callback(
    context: RequestContext["ClientSession", Any],
    params: types.ElicitRequestParams,
):
    """
    MCP 서버로부터 Elicit 요청을 받았을 때 호출되는 콜백 함수.
    서버의 메시지를 웹소켓 클라이언트에게 전송하고, 사용자의 응답을 받아 서버로 반환합니다.
    """
    print(f"Server -> Client: {params.message}")
    data = json.loads(params.message)
    await WEBSOCKET_MANAGER[data["session_id"]].send_json(
        {"message": data["ai_message"], "step_name": data["step_name"]}
    )

    user_message = None
    if data["retrieve_output"] is True:
        user_response = await WEBSOCKET_MANAGER[data["session_id"]].receive_json()
        user_message = user_response["message"]
        print(f"User -> Server: {user_message}")

    return types.ElicitResult(
        action="accept",
        content={"user_message": user_message},
    )


@app.websocket("/conversation")
async def civil_complaint_conversation(websocket: WebSocket):
    """
    무인민원발급기 대화를 위한 웹소켓 엔드포인트
    """
    async with streamablehttp_client(url=MCP_SERVER_URL) as (read_stream, write_stream, get_session_id):
        async with ClientSession(
            read_stream=read_stream,
            write_stream=write_stream,
            elicitation_callback=smart_elicitation_callback,
        ) as session:
            await session.initialize()
            
            session_id = get_session_id()
            try:
                await websocket.accept()
                WEBSOCKET_MANAGER[session_id] = websocket
                
                # 라우터 체인 생성
                router_chain = await create_router_chain()
                
                # 1. 초기 안내 및 사용자 첫 발화 수신
                initial_message = "AI 무인민원발급기입니다. 어떤 서비스를 원하시나요?"
                await websocket.send_json({"message": initial_message, "step_name": "service_selection"})
                user_response = await websocket.receive_json()
                user_message = user_response["message"]
                
                # 2. LLM을 이용해 호출할 Tool 결정
                selected_tool = await router_chain.ainvoke({"user_message": user_message})
                
                # 3. 지원하지 않거나 모호한 요청 처리
                while selected_tool.tool_name == "unsupported":
                    await websocket.send_json({"message": selected_tool.ai_message, "step_name": "service_selection_retry"})
                    user_response = await websocket.receive_json()
                    user_message = user_response["message"]
                    selected_tool = await router_chain.ainvoke({"user_message": user_message})

                # 4. 결정된 MCP Tool 호출
                logger.info(f"[{session_id}] User wants '{user_message}'. Calling tool: {selected_tool.tool_name}")
                result = await session.call_tool(
                    name=selected_tool.tool_name,
                    arguments={"session_id": session_id},
                )
                
                # 최종 결과를 클라이언트에게 전송
                if result.structuredContent:
                    await websocket.send_json(result.structuredContent)

            except Exception as e:
                logger.exception(e)
            finally:
                if session_id in WEBSOCKET_MANAGER:
                    await websocket.close()
                    del WEBSOCKET_MANAGER[session_id]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)