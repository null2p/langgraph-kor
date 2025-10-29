# Standalone Container 배포 방법

배포하기 전에 [Standalone Container](../../concepts/langgraph_standalone_container.md) 배포 옵션에 대한 개념 가이드를 검토하세요.

## Prerequisites

1. [LangGraph CLI](../../concepts/langgraph_cli.md)를 사용하여 [애플리케이션을 로컬에서 테스트](../../tutorials/langgraph-platform/local-server.md)합니다.
1. [LangGraph CLI](../../concepts/langgraph_cli.md)를 사용하여 Docker 이미지를 빌드합니다 (예: `langgraph build`).
1. standalone container 배포에는 다음 환경 변수가 필요합니다.
    1. `REDIS_URI`: Redis 인스턴스에 대한 연결 세부 정보. Redis는 백그라운드 실행에서 실시간 출력을 스트리밍할 수 있도록 pub-sub 브로커로 사용됩니다. `REDIS_URI` 값은 유효한 [Redis 연결 URI](https://redis-py.readthedocs.io/en/stable/connections.html#redis.Redis.from_url)여야 합니다.

        !!! Note "Shared Redis Instance"
            여러 self-hosted 배포가 동일한 Redis 인스턴스를 공유할 수 있습니다. 예를 들어, `Deployment A`의 경우 `REDIS_URI`를 `redis://<hostname_1>:<port>/1`로 설정하고 `Deployment B`의 경우 `REDIS_URI`를 `redis://<hostname_1>:<port>/2`로 설정할 수 있습니다.

            `1`과 `2`는 동일한 인스턴스 내의 서로 다른 데이터베이스 번호이지만 `<hostname_1>`은 공유됩니다. **동일한 데이터베이스 번호를 별도의 배포에 사용할 수 없습니다**.

    1. `DATABASE_URI`: Postgres 연결 세부 정보. Postgres는 assistants, threads, runs를 저장하고, thread state 및 장기 메모리를 유지하며, '정확히 한 번' 시맨틱으로 백그라운드 작업 큐의 상태를 관리하는 데 사용됩니다. `DATABASE_URI` 값은 유효한 [Postgres 연결 URI](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING-URIS)여야 합니다.

        !!! Note "Shared Postgres Instance"
            여러 self-hosted 배포가 동일한 Postgres 인스턴스를 공유할 수 있습니다. 예를 들어, `Deployment A`의 경우 `DATABASE_URI`를 `postgres://<user>:<password>@/<database_name_1>?host=<hostname_1>`로 설정하고 `Deployment B`의 경우 `DATABASE_URI`를 `postgres://<user>:<password>@/<database_name_2>?host=<hostname_1>`로 설정할 수 있습니다.

            `<database_name_1>`과 `database_name_2`는 동일한 인스턴스 내의 서로 다른 데이터베이스이지만 `<hostname_1>`은 공유됩니다. **동일한 데이터베이스를 별도의 배포에 사용할 수 없습니다**.

    1. `LANGGRAPH_CLOUD_LICENSE_KEY`: ([Enterprise](../../concepts/langgraph_data_plane.md#licensing) 사용 시) LangGraph Platform 라이선스 키. 이는 서버 시작 시 한 번 인증하는 데 사용됩니다.
    1. `LANGSMITH_ENDPOINT`: [self-hosted LangSmith](https://docs.smith.langchain.com/self_hosting) 인스턴스로 추적을 보내려면 `LANGSMITH_ENDPOINT`를 self-hosted LangSmith 인스턴스의 호스트 이름으로 설정합니다.
1. 네트워크에서 `https://beacon.langchain.com`로의 Egress. 이는 air-gapped 모드에서 실행하지 않는 경우 라이선스 확인 및 사용량 보고에 필요합니다. 자세한 내용은 [Egress 문서](../../cloud/deployment/egress.md)를 참조하세요.

## Kubernetes (Helm)

이 [Helm chart](https://github.com/langchain-ai/helm/blob/main/charts/langgraph-cloud/README.md)를 사용하여 Kubernetes 클러스터에 LangGraph Server를 배포합니다.

## Docker

다음 `docker` 명령을 실행합니다:
```shell
docker run \
    --env-file .env \
    -p 8123:8000 \
    -e REDIS_URI="foo" \
    -e DATABASE_URI="bar" \
    -e LANGSMITH_API_KEY="baz" \
    my-image
```

!!! note

    * 사전 요구사항 단계(`langgraph build`)에서 빌드한 이미지의 이름으로 `my-image`를 교체해야 하며, `REDIS_URI`, `DATABASE_URI`, `LANGSMITH_API_KEY`에 적절한 값을 제공해야 합니다.
    * 애플리케이션에 추가 환경 변수가 필요한 경우 유사한 방식으로 전달할 수 있습니다.

## Docker Compose

Docker Compose YAML 파일:
```yml
volumes:
    langgraph-data:
        driver: local
services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-postgres:
        image: postgres:16
        ports:
            - "5433:5432"
        environment:
            POSTGRES_DB: postgres
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
        volumes:
            - langgraph-data:/var/lib/postgresql/data
        healthcheck:
            test: pg_isready -U postgres
            start_period: 10s
            timeout: 1s
            retries: 5
            interval: 5s
    langgraph-api:
        image: ${IMAGE_NAME}
        ports:
            - "8123:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
            langgraph-postgres:
                condition: service_healthy
        env_file:
            - .env
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            LANGSMITH_API_KEY: ${LANGSMITH_API_KEY}
            POSTGRES_URI: postgres://postgres:postgres@langgraph-postgres:5432/postgres?sslmode=disable
```

동일한 폴더에 이 Docker Compose 파일과 함께 `docker compose up` 명령을 실행할 수 있습니다.

이렇게 하면 포트 `8123`에서 LangGraph Server가 시작됩니다 (이를 변경하려면 `langgraph-api` 볼륨의 포트를 변경하면 됩니다). 다음을 실행하여 애플리케이션이 정상인지 테스트할 수 있습니다:

```shell
curl --request GET --url 0.0.0.0:8123/ok
```
모든 것이 올바르게 실행되고 있다면 다음과 같은 응답이 표시됩니다:

```shell
{"ok":true}
```
