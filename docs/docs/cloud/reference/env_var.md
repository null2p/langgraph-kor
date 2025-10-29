# 환경 변수

LangGraph Server는 배포를 구성하기 위한 특정 환경 변수를 지원합니다.

## `BG_JOB_ISOLATED_LOOPS`

백그라운드 실행을 serving API 이벤트 루프와 분리된 격리된 이벤트 루프에서 실행하려면 `BG_JOB_ISOLATED_LOOPS`를 `True`로 설정합니다.

그래프/노드 구현에 동기 코드가 포함되어 있는 경우 이 환경 변수를 `True`로 설정해야 합니다. 이 상황에서 동기 코드는 serving API 이벤트 루프를 차단하여 API를 사용할 수 없게 만들 수 있습니다. 사용할 수 없는 API의 증상은 상태 확인 실패로 인한 지속적인 애플리케이션 재시작입니다.

기본값은 `False`입니다.

## `BG_JOB_SHUTDOWN_GRACE_PERIOD_SECS`

큐가 종료 신호를 받은 후 서버가 백그라운드 작업이 완료될 때까지 기다리는 시간(초)을 지정합니다. 이 기간이 지나면 서버가 강제 종료됩니다. 기본값은 `180`초입니다. 종료 시 작업이 깔끔하게 완료될 충분한 시간을 확보하도록 이를 설정하세요. `langgraph-api==0.2.16`에서 추가되었습니다.

## `BG_JOB_TIMEOUT_SECS`

백그라운드 실행의 타임아웃을 늘릴 수 있습니다. 그러나 Cloud SaaS 배포의 인프라는 API 요청에 대해 1시간 타임아웃 제한을 적용합니다. 이는 클라이언트와 서버 간의 연결이 1시간 후에 타임아웃된다는 것을 의미합니다. 이는 구성할 수 없습니다.

백그라운드 실행은 1시간 이상 실행될 수 있지만, 실행이 1시간 이상 걸리는 경우 클라이언트는 서버에 다시 연결해야 합니다(예: `POST /threads/{thread_id}/runs/{run_id}/stream`을 통해 스트림에 참여). 그래야 실행에서 출력을 검색할 수 있습니다.

기본값은 `3600`입니다.

## `DD_API_KEY`

배포에 대한 Datadog 추적을 자동으로 활성화하려면 `DD_API_KEY`([Datadog API 키](https://docs.datadoghq.com/account_management/api-app-keys/))를 지정하세요. 추적 계측을 구성하려면 다른 [`DD_*` 환경 변수](https://ddtrace.readthedocs.io/en/stable/configuration.html)를 지정하세요.

`DD_API_KEY`가 지정되면 애플리케이션 프로세스가 [`ddtrace-run` 명령](https://ddtrace.readthedocs.io/en/stable/installation_quickstart.html)으로 래핑됩니다. 추적 계측을 올바르게 구성하려면 일반적으로 다른 `DD_*` 환경 변수(예: `DD_SITE`, `DD_ENV`, `DD_SERVICE`, `DD_TRACE_ENABLED`)가 필요합니다. 자세한 내용은 [`DD_*` 환경 변수](https://ddtrace.readthedocs.io/en/stable/configuration.html)를 참조하세요.

!!! note
    `DD_API_KEY`를 활성화하면(따라서 `ddtrace-run`) 애플리케이션 코드에 계측한 다른 자동 계측 솔루션(예: OpenTelemetry)을 재정의하거나 간섭할 수 있습니다.

## `LANGCHAIN_TRACING_SAMPLING_RATE`

LangSmith로 전송되는 추적의 샘플링 비율입니다. 유효한 값: `0`과 `1` 사이의 모든 부동 소수점 숫자.

자세한 내용은 <a href="https://docs.smith.langchain.com/how_to_guides/tracing/sample_traces" target="_blank">LangSmith 문서</a>를 참조하세요.

## `LANGGRAPH_AUTH_TYPE`

LangGraph Server 배포에 대한 인증 유형입니다. 유효한 값: `langsmith`, `noop`.

LangGraph Platform 배포의 경우 이 환경 변수가 자동으로 설정됩니다. 로컬 개발이나 인증이 외부에서 처리되는 배포(예: 자체 호스팅)의 경우 이 환경 변수를 `noop`으로 설정하세요.

## `LANGGRAPH_POSTGRES_POOL_MAX_SIZE`

langgraph-api 버전 `0.2.12`부터 Postgres 연결 풀의 최대 크기(복제본당)를 `LANGGRAPH_POSTGRES_POOL_MAX_SIZE` 환경 변수를 사용하여 제어할 수 있습니다. 이 변수를 설정하면 서버가 Postgres 데이터베이스와 설정할 동시 연결 수의 상한을 결정할 수 있습니다.

예를 들어, 배포가 10개의 복제본으로 확장되고 `LANGGRAPH_POSTGRES_POOL_MAX_SIZE`가 `150`으로 구성된 경우 최대 `1500`개의 Postgres 연결을 설정할 수 있습니다. 이는 데이터베이스 리소스가 제한되거나(또는 더 사용 가능하거나) 성능 또는 확장 이유로 연결 동작을 조정해야 하는 배포에 특히 유용합니다.

기본값은 `150` 연결입니다.

## `LANGSMITH_RUNS_ENDPOINTS`

[자체 호스팅 LangSmith](https://docs.smith.langchain.com/self_hosting)가 있는 배포에만 해당합니다.

배포가 자체 호스팅 LangSmith 인스턴스로 추적을 보내도록 이 환경 변수를 설정하세요. `LANGSMITH_RUNS_ENDPOINTS`의 값은 JSON 문자열입니다: `{"<SELF_HOSTED_LANGSMITH_HOSTNAME>":"<LANGSMITH_API_KEY>"}`.

`SELF_HOSTED_LANGSMITH_HOSTNAME`은 자체 호스팅 LangSmith 인스턴스의 호스트 이름입니다. 배포에서 액세스할 수 있어야 합니다. `LANGSMITH_API_KEY`는 자체 호스팅 LangSmith 인스턴스에서 생성된 LangSmith API입니다.

## `LANGSMITH_TRACING`

LangSmith로의 추적을 비활성화하려면 `LANGSMITH_TRACING`을 `false`로 설정하세요.

기본값은 `true`입니다.

## `LOG_COLOR`

이는 주로 `langgraph dev` 명령을 통해 개발 서버를 사용하는 컨텍스트에서 관련이 있습니다. 기본 콘솔 렌더러를 사용할 때 ANSI 색상 콘솔 출력을 활성화하려면 `LOG_COLOR`를 `true`로 설정하세요. 이 변수를 `false`로 설정하여 색상 출력을 비활성화하면 흑백 로그가 생성됩니다. 기본값은 `true`입니다.

## `LOG_LEVEL`

[로그 레벨](https://docs.python.org/3/library/logging.html#logging-levels)을 구성합니다. 기본값은 `INFO`입니다.

## `LOG_JSON`

구성된 `JSONRenderer`를 사용하여 모든 로그 메시지를 JSON 객체로 렌더링하려면 `LOG_JSON`을 `true`로 설정하세요. 이렇게 하면 로그 관리 시스템에서 쉽게 구문 분석하거나 수집할 수 있는 구조화된 로그가 생성됩니다. 기본값은 `false`입니다.

## `MOUNT_PREFIX`

!!! info "자체 호스팅 배포에서만 허용됨"
    `MOUNT_PREFIX` 환경 변수는 자체 호스팅 배포 모델에서만 허용되며, LangGraph Platform SaaS는 이 환경 변수를 허용하지 않습니다.

특정 경로 접두사 아래에서 LangGraph Server를 제공하려면 `MOUNT_PREFIX`를 설정하세요. 이는 서버가 특정 경로 접두사가 필요한 역방향 프록시 또는 로드 밸런서 뒤에 있는 배포에 유용합니다.

예를 들어, 서버가 `https://example.com/langgraph` 아래에서 제공되도록 하려면 `MOUNT_PREFIX`를 `/langgraph`로 설정하세요.

## `N_JOBS_PER_WORKER`

LangGraph Server 작업 큐의 워커당 작업 수입니다. 기본값은 `10`입니다.

## `POSTGRES_URI_CUSTOM`

!!! info "자체 호스팅 Data Plane 및 자체 호스팅 Control Plane 전용"
    사용자 지정 Postgres 인스턴스는 [자체 호스팅 Data Plane](../../concepts/langgraph_self_hosted_data_plane.md) 및 [자체 호스팅 Control Plane](../../concepts/langgraph_self_hosted_control_plane.md) 배포에서만 사용할 수 있습니다.

사용자 지정 Postgres 인스턴스를 사용하려면 `POSTGRES_URI_CUSTOM`을 지정하세요. `POSTGRES_URI_CUSTOM`의 값은 유효한 [Postgres 연결 URI](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING-URIS)여야 합니다.

Postgres:

- 버전 15.8 이상.
- 초기 데이터베이스가 있어야 하며 연결 URI가 데이터베이스를 참조해야 합니다.

Control Plane 기능:

- `POSTGRES_URI_CUSTOM`이 지정되면 LangGraph Control Plane은 서버에 대한 데이터베이스를 프로비저닝하지 않습니다.
- `POSTGRES_URI_CUSTOM`이 제거되면 LangGraph Control Plane은 서버에 대한 데이터베이스를 프로비저닝하지 않으며 외부에서 관리되는 Postgres 인스턴스를 삭제하지 않습니다.
- `POSTGRES_URI_CUSTOM`이 제거되면 리비전 배포가 성공하지 않습니다. `POSTGRES_URI_CUSTOM`이 한 번 지정되면 배포의 수명 주기 동안 항상 설정되어야 합니다.
- 배포가 삭제되면 LangGraph Control Plane은 외부에서 관리되는 Postgres 인스턴스를 삭제하지 않습니다.
- `POSTGRES_URI_CUSTOM`의 값을 업데이트할 수 있습니다. 예를 들어 URI의 비밀번호를 업데이트할 수 있습니다.

데이터베이스 연결:

- 사용자 지정 Postgres 인스턴스는 LangGraph Server에서 액세스할 수 있어야 합니다. 사용자는 연결을 보장할 책임이 있습니다.

## `REDIS_CLUSTER`

!!! info "자체 호스팅 배포에서만 허용됨"
    Redis 클러스터 모드는 자체 호스팅 배포 모델에서만 사용할 수 있으며, LangGraph Platform SaaS는 기본적으로 redis 인스턴스를 프로비저닝합니다.

Redis 클러스터 모드를 활성화하려면 `REDIS_CLUSTER`를 `True`로 설정하세요. 활성화되면 시스템은 클러스터 모드를 사용하여 Redis에 연결합니다. 이는 Redis 클러스터 배포에 연결할 때 유용합니다.

기본값은 `False`입니다.

## `REDIS_KEY_PREFIX`

!!! info "API Server 버전 0.1.9 이상에서 사용 가능"
    이 환경 변수는 API Server 버전 0.1.9 이상에서 지원됩니다.

Redis 키의 접두사를 지정하세요. 이를 통해 여러 LangGraph Server 인스턴스가 다른 키 접두사를 사용하여 동일한 Redis 인스턴스를 공유할 수 있습니다.

기본값은 `''`입니다.

## `REDIS_URI_CUSTOM`

!!! info "자체 호스팅 Data Plane 및 자체 호스팅 Control Plane 전용"
    사용자 지정 Redis 인스턴스는 [자체 호스팅 Data Plane](../../concepts/langgraph_self_hosted_data_plane.md) 및 [자체 호스팅 Control Plane](../../concepts/langgraph_self_hosted_control_plane.md) 배포에서만 사용할 수 있습니다.

사용자 지정 Redis 인스턴스를 사용하려면 `REDIS_URI_CUSTOM`을 지정하세요. `REDIS_URI_CUSTOM`의 값은 유효한 [Redis 연결 URI](https://redis-py.readthedocs.io/en/stable/connections.html#redis.Redis.from_url)여야 합니다.

## `RESUMABLE_STREAM_TTL_SECONDS`

Redis의 재개 가능한 스트림 데이터에 대한 초 단위 TTL(time-to-live)입니다.

실행이 생성되고 출력이 스트리밍되면 스트림을 재개 가능하도록 구성할 수 있습니다(예: `stream_resumable=True`). 스트림이 재개 가능한 경우 스트림의 출력이 Redis에 임시로 저장됩니다. 이 데이터의 TTL은 `RESUMABLE_STREAM_TTL_SECONDS`를 설정하여 구성할 수 있습니다.

재개 가능한 스트림을 구현하는 방법에 대한 자세한 내용은 [Python](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.RunsClient.stream) 및 [JS/TS](https://langchain-ai.github.io/langgraphjs/reference/classes/sdk_client.RunsClient.html#stream) SDK를 참조하세요.

기본값은 `120`초입니다.
