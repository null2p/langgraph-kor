# 커스텀 lifespan 이벤트 추가 방법

에이전트를 LangGraph Platform에 배포할 때, 서버가 시작될 때 데이터베이스 연결과 같은 리소스를 초기화하고, 종료될 때 적절하게 닫히도록 해야 하는 경우가 많습니다. Lifespan 이벤트를 사용하면 서버의 시작 및 종료 시퀀스에 연결하여 이러한 중요한 설정 및 정리 작업을 처리할 수 있습니다.

이는 [커스텀 라우트 추가](./custom_routes.md)와 동일한 방식으로 작동합니다. 자신만의 [`Starlette`](https://www.starlette.io/applications/) 앱([`FastAPI`](https://fastapi.tiangolo.com/), [`FastHTML`](https://fastht.ml/) 및 기타 호환 가능한 앱 포함)을 제공하기만 하면 됩니다.

다음은 FastAPI를 사용한 예제입니다.

???+ note "Python만 지원"

    현재 `langgraph-api>=0.0.26`을 사용하는 Python 배포에서만 커스텀 lifespan 이벤트를 지원합니다.

## 앱 생성

**기존** LangGraph Platform 애플리케이션에서 시작하여 `webapp.py` 파일에 다음 lifespan 코드를 추가합니다. 처음부터 시작하는 경우 CLI를 사용하여 템플릿에서 새 앱을 생성할 수 있습니다.

```bash
langgraph new --template=new-langgraph-project-python my_new_project
```

LangGraph 프로젝트가 있으면 다음 앱 코드를 추가합니다:

```python
# ./src/agent/webapp.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 예를 들어...
    engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
    # 재사용 가능한 세션 팩토리 생성
    async_session = sessionmaker(engine, class_=AsyncSession)
    # 앱 상태에 저장
    app.state.db_session = async_session
    yield
    # 연결 정리
    await engine.dispose()

# highlight-next-line
app = FastAPI(lifespan=lifespan)

# ... 필요한 경우 커스텀 라우트를 추가할 수 있습니다.
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

서버가 시작될 때 시작 메시지가 출력되고, `Ctrl+C`로 중지하면 정리 메시지가 출력되는 것을 볼 수 있습니다.

## 배포

이 앱을 그대로 LangGraph Platform 또는 self-hosted platform에 배포할 수 있습니다.

## 다음 단계

이제 배포에 lifespan 이벤트를 추가했으므로 유사한 기술을 사용하여 [커스텀 라우트](./custom_routes.md)나 [커스텀 미들웨어](./custom_middleware.md)를 추가하여 서버의 동작을 추가로 커스터마이징할 수 있습니다.
