# 커스텀 미들웨어 추가 방법

에이전트를 LangGraph Platform에 배포할 때, 서버에 커스텀 미들웨어를 추가하여 요청 메트릭 로깅, 헤더 삽입 또는 확인, 핵심 서버 로직을 수정하지 않고 보안 정책 시행과 같은 관심사를 처리할 수 있습니다. 이는 [커스텀 라우트 추가](./custom_routes.md)와 동일한 방식으로 작동합니다. 자신만의 [`Starlette`](https://www.starlette.io/applications/) 앱([`FastAPI`](https://fastapi.tiangolo.com/), [`FastHTML`](https://fastht.ml/) 및 기타 호환 가능한 앱 포함)을 제공하기만 하면 됩니다.

미들웨어를 추가하면 커스텀 엔드포인트를 호출하든 내장 LangGraph Platform API를 호출하든 관계없이 배포 전체에서 요청과 응답을 전역적으로 가로채고 수정할 수 있습니다.

다음은 FastAPI를 사용한 예제입니다.

???+ note "Python만 지원"

    현재 `langgraph-api>=0.0.26`을 사용하는 Python 배포에서만 커스텀 미들웨어를 지원합니다.

## 앱 생성

**기존** LangGraph Platform 애플리케이션에서 시작하여 webapp 파일에 다음 미들웨어 코드를 추가합니다. 처음부터 시작하는 경우 CLI를 사용하여 템플릿에서 새 앱을 생성할 수 있습니다.

```bash
langgraph new --template=new-langgraph-project-python my_new_project
```

LangGraph 프로젝트가 있으면 다음 앱 코드를 추가합니다:

```python
# ./src/agent/webapp.py
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

# highlight-next-line
app = FastAPI()

class CustomHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers['X-Custom-Header'] = 'Hello from middleware!'
        return response

# 앱에 미들웨어 추가
app.add_middleware(CustomHeaderMiddleware)
```

## `langgraph.json` 구성

`langgraph.json` 구성 파일에 다음을 추가합니다. 경로가 위에서 만든 `webapp.py` 파일을 가리키는지 확인하세요.

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "env": ".env",
  "http": {
    "app": "./src/agent/webapp.py:app"
  }
  // 인증, 저장소 등과 같은 기타 구성 옵션
}
```

## 서버 시작

로컬에서 서버를 테스트합니다:

```bash
langgraph dev --no-browser
```

이제 서버에 대한 모든 요청은 응답에 커스텀 헤더 `X-Custom-Header`를 포함합니다.

## 배포

이 앱을 그대로 LangGraph Platform 또는 self-hosted platform에 배포할 수 있습니다.

## 다음 단계

이제 배포에 커스텀 미들웨어를 추가했으므로 유사한 기술을 사용하여 [커스텀 라우트](./custom_routes.md)를 추가하거나 [커스텀 lifespan 이벤트](./custom_lifespan.md)를 정의하여 서버의 동작을 추가로 커스터마이징할 수 있습니다.
