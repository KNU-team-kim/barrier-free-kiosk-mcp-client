## 종합설계프로젝트2 MCP Client

#### 실행 방법
1. `.env` 파일을 생성 후, 아래와 같이 입력한다.
```bash
OPENAI_API_KEY="ajtlrlajtlrl"
OPENAI_API_URL="https://ajtlrlajtlrl.com"
MCP_SERVER_URL="https://ajtlrlajtlrl.com"
```

2. 아래 명령어를 입력한다.
```bash
uv sync --frozen
uv run python client.py
```