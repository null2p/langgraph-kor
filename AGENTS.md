# AGENTS 지침

이 저장소는 모노레포입니다. 각 라이브러리는 `libs/` 하위의 하위 디렉토리에 있습니다.

어떤 라이브러리의 코드를 수정하든, pull request를 생성하기 전에 해당 라이브러리의 디렉토리에서 다음 명령을 실행하세요:

- `make format` – 코드 포매터 실행
- `make lint` – 린터 실행
- `make test` – 테스트 스위트 실행

특정 테스트 파일을 실행하거나 추가 pytest 옵션을 전달하려면 `TEST` 변수를 지정할 수 있습니다:

```
TEST=path/to/test.py make test
```

다른 pytest 인수도 `TEST` 변수 내에서 제공할 수 있습니다.

## 라이브러리

저장소에는 여러 Python 및 JavaScript/TypeScript 라이브러리가 포함되어 있습니다.
다음은 상위 수준 개요입니다:

- **checkpoint** – LangGraph 체크포인터를 위한 기본 인터페이스.
- **checkpoint-postgres** – 체크포인트 저장소의 Postgres 구현.
- **checkpoint-sqlite** – 체크포인트 저장소의 SQLite 구현.
- **cli** – LangGraph를 위한 공식 명령줄 인터페이스.
- **langgraph** – 상태 저장형 다중 액터 에이전트를 구축하기 위한 핵심 프레임워크.
- **prebuilt** – 에이전트와 도구를 생성하고 실행하기 위한 상위 수준 API.
- **sdk-js** – LangGraph REST API와 상호작용하기 위한 JS/TS SDK.
- **sdk-py** – LangGraph Server API를 위한 Python SDK.

### 의존성 맵

아래 다이어그램은 해당 라이브러리의 `pyproject.toml`(또는 `package.json`)에 선언된 각 프로덕션 의존성에 대한 다운스트림 라이브러리를 나열합니다.

```text
checkpoint
├── checkpoint-postgres
├── checkpoint-sqlite
├── prebuilt
└── langgraph

prebuilt
└── langgraph

sdk-py
├── langgraph
└── cli

sdk-js (독립형)
```

라이브러리에 대한 변경은 위에 표시된 모든 종속 항목에 영향을 줄 수 있습니다.
