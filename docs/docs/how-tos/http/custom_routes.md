# 커스텀 라우트 추가 방법

에이전트를 LangGraph Platform에 배포할 때, 서버는 자동으로 run 및 thread 생성, 장기 메모리 저장소와의 상호 작용, 구성 가능한 assistant 관리 및 기타 핵심 기능을 위한 라우트를 노출합니다([모든 기본 API 엔드포인트 보기](../../cloud/reference/api/api_ref.md)).

자신만의 [`Starlette`](https://www.starlette.io/applications/) 앱([ `FastAPI`](https://fastapi.tiangolo.com/), [`FastHTML`](https://fastht.ml/) 및 기타 호환 가능한 앱 포함)을 제공하여 커스텀 라우트를 추가할 수 있습니다. `langgraph.json` 구성 파일에 앱 경로를 제공하여 LangGraph Platform에 이를 알립니다.

커스텀 앱 객체를 정의하면 원하는 라우트를 추가할 수 있으므로 `/login` 엔드포인트 추가부터 전체 풀스택 웹앱 작성까지 모든 것을 단일 LangGraph Server에 배포할 수 있습니다.

다음은 FastAPI를 사용한 예제입니다.

## 앱 생성

**기존** LangGraph Platform 애플리케이션에서 시작하여 webapp 파일에 다음 커스텀 라우트 코드를 추가합니다. 처음부터 시작하는 경우 CLI를 사용하여 템플릿에서 새 앱을 생성할 수 있습니다.

```bash
langgraph new --template=new-langgraph-project-python my_new_project
```

LangGraph 프로젝트가 있으면 다음 앱 코드를 추가합니다:

```python
# ./src/agent/webapp.py
from fastapi import FastAPI

# highlight-next-line
app = FastAPI()


@app.get("/hello")
def read_root():
    return {"Hello": "World"}

```

## `langgraph.json` 구성

`langgraph.json` 구성 파일에 다음을 추가합니다. 경로가 위에서 만든 `webapp.py` 파일의 FastAPI 애플리케이션 인스턴스 `app`을 가리키는지 확인하세요.

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

브라우저에서 `localhost:2024/hello`로 이동하면(`2024`는 기본 개발 포트) `/hello` 엔드포인트가 `{"Hello": "World"}`를 반환하는 것을 볼 수 있습니다.

!!! note "기본 엔드포인트 shadowing"

    앱에서 생성하는 라우트는 시스템 기본값보다 우선순위가 높습니다. 즉, 모든 기본 엔드포인트의 동작을 shadow하고 재정의할 수 있습니다.

## 배포

이 앱을 그대로 LangGraph Platform 또는 self-hosted platform에 배포할 수 있습니다.

## 다음 단계

이제 배포에 커스텀 라우트를 추가했으므로 동일한 기술을 사용하여 [커스텀 미들웨어](./custom_middleware.md) 및 [커스텀 lifespan 이벤트](./custom_lifespan.md) 정의와 같이 서버의 동작을 추가로 커스터마이징할 수 있습니다.
