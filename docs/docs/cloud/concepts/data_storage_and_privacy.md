# 데이터 저장 및 프라이버시

이 문서는 인메모리 서버(`langgraph dev`)와 로컬 Docker 서버(`langgraph up`) 모두에서 LangGraph CLI와 LangGraph Server의 데이터 처리 방식을 설명합니다. 또한 호스팅된 LangGraph Studio 프런트엔드와 상호 작용할 때 추적되는 데이터에 대해서도 설명합니다.

## CLI

LangGraph **CLI**는 LangGraph 애플리케이션을 구축하고 실행하기 위한 커맨드라인 인터페이스입니다. 자세한 내용은 [CLI 가이드](../../concepts/langgraph_cli.md)를 참조하세요.

기본적으로 대부분의 CLI 명령 호출은 호출 시 단일 분석 이벤트를 로깅합니다. 이는 CLI 경험 개선의 우선순위를 더 잘 정하는 데 도움이 됩니다. 각 텔레메트리 이벤트에는 호출 프로세스의 OS, OS 버전, Python 버전, CLI 버전, 명령 이름(`dev`, `up`, `run` 등), 그리고 플래그가 명령에 전달되었는지 여부를 나타내는 boolean이 포함됩니다. 전체 분석 로직은 [여기](https://github.com/langchain-ai/langgraph/blob/main/libs/cli/langgraph_cli/analytics.py)에서 확인할 수 있습니다.

`LANGGRAPH_CLI_NO_ANALYTICS=1`을 설정하여 모든 CLI 텔레메트리를 비활성화할 수 있습니다.

## LangGraph Server (인메모리 & docker)

[LangGraph Server](../../concepts/langgraph_server.md)는 애플리케이션 상태의 checkpoint, 장기 메모리, thread 메타데이터, assistant 및 유사한 리소스를 로컬 파일 시스템이나 데이터베이스에 지속적으로 유지하는 내구성 있는 실행 런타임을 제공합니다. 저장 위치를 의도적으로 사용자 지정하지 않는 한, 이 정보는 로컬 디스크(`langgraph dev`의 경우) 또는 PostgreSQL 데이터베이스(`langgraph up` 및 모든 배포의 경우)에 기록됩니다.

### LangSmith 추적

LangGraph 서버(인메모리 또는 Docker 모두)를 실행할 때 LangSmith 추적을 활성화하여 더 빠른 디버깅을 촉진하고 프로덕션에서 그래프 상태 및 LLM 프롬프트에 대한 관찰 가능성을 제공할 수 있습니다. 서버의 런타임 환경에서 `LANGSMITH_TRACING=false`를 설정하여 언제든지 추적을 비활성화할 수 있습니다.

### 인메모리 개발 서버 (`langgraph dev`)

`langgraph dev`는 빠른 개발과 테스트를 위해 설계된 단일 Python 프로세스로 [인메모리 개발 서버](../../tutorials/langgraph-platform/local-server.md)를 실행합니다. 모든 체크포인팅 및 메모리 데이터를 현재 작업 디렉토리의 `.langgraph_api` 디렉토리 내 디스크에 저장합니다. [CLI](#cli) 섹션에서 설명한 텔레메트리 데이터를 제외하고, 추적을 활성화했거나 그래프 코드가 외부 서비스에 명시적으로 연결하지 않는 한 머신 외부로 데이터가 전송되지 않습니다.

### Standalone Container (`langgraph up`)

`langgraph up`은 로컬 패키지를 Docker 이미지로 빌드하고 API 서버, PostgreSQL 컨테이너, Redis 컨테이너의 세 가지 컨테이너로 구성된 [standalone container](../../concepts/deployment_options.md#standalone-container)로 서버를 실행합니다. 모든 지속 데이터(checkpoint, assistant 등)는 PostgreSQL 데이터베이스에 저장됩니다. Redis는 이벤트의 실시간 스트리밍을 위한 pubsub 연결로 사용됩니다. 유효한 `LANGGRAPH_AES_KEY` 환경 변수를 설정하여 데이터베이스에 저장하기 전에 모든 checkpoint를 암호화할 수 있습니다. 또한 `langgraph.json`에서 checkpoint 및 cross-thread 메모리에 대한 [TTL](../../how-tos/ttl/configure_ttl.md)을 지정하여 데이터가 저장되는 기간을 제어할 수 있습니다. 지속된 모든 thread, 메모리 및 기타 데이터는 관련 API 엔드포인트를 통해 삭제할 수 있습니다.

서버에 유효한 라이선스가 있는지 확인하고 실행된 run 및 task 수를 추적하기 위해 추가 API 호출이 이루어집니다. 주기적으로 API 서버는 제공된 라이선스 키(또는 API 키)를 검증합니다.

[추적](#langsmith-tracing)을 비활성화한 경우, 그래프 코드가 외부 서비스에 명시적으로 연결하지 않는 한 사용자 데이터가 외부에 지속되지 않습니다.

## Studio

[LangGraph Studio](../../concepts/langgraph_studio.md)는 LangGraph 서버와 상호 작용하기 위한 그래픽 인터페이스입니다. 개인 데이터를 지속하지 않습니다(서버로 보내는 데이터는 LangSmith로 전송되지 않습니다). studio 인터페이스는 [smith.langchain.com](https://smith.langchain.com)에서 제공되지만, 브라우저에서 실행되며 로컬 LangGraph 서버에 직접 연결되므로 LangSmith로 데이터를 보낼 필요가 없습니다.

로그인한 경우 LangSmith는 studio의 사용자 경험을 개선하기 위해 일부 사용 분석을 수집합니다. 여기에는 다음이 포함됩니다:

- 페이지 방문 및 탐색 패턴
- 사용자 작업(버튼 클릭)
- 브라우저 유형 및 버전
- 화면 해상도 및 뷰포트 크기

중요한 점은 애플리케이션 데이터나 코드(또는 기타 민감한 구성 세부 정보)는 수집되지 않는다는 것입니다. 이 모든 것은 LangGraph 서버의 지속성 계층에 저장됩니다. Studio를 익명으로 사용할 때는 계정 생성이 필요하지 않으며 사용 분석도 수집되지 않습니다.

## 빠른 참조

요약하자면, CLI 분석을 끄고 추적을 비활성화하여 서버 측 텔레메트리를 opt-out할 수 있습니다.

| Variable                       | Purpose                   | Default                          |
| ------------------------------ | ------------------------- | -------------------------------- |
| `LANGGRAPH_CLI_NO_ANALYTICS=1` | CLI 분석 비활성화     | 분석 활성화                |
| `LANGSMITH_API_KEY`            | LangSmith 추적 활성화  | 추적 비활성화                 |
| `LANGSMITH_TRACING=false`      | LangSmith 추적 비활성화 | 환경에 따라 다름           |
