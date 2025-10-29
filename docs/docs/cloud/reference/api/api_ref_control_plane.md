# LangGraph Control Plane API 레퍼런스

LangGraph Control Plane API는 프로그래밍 방식으로 LangGraph Server 배포를 생성하고 관리하는 데 사용됩니다. 예를 들어, API를 조율하여 커스텀 CI/CD 워크플로를 만들 수 있습니다.

API 레퍼런스를 보려면 <a href="https://api.host.langchain.com/docs" target="_blank">여기</a>를 클릭하세요.

## 호스트

Cloud SaaS 데이터 리전의 LangGraph Control Plane 호스트:

| US | EU |
|----|----|
| `https://api.host.langchain.com` | `https://eu.api.host.langchain.com` |

**참고**: LangGraph Platform의 셀프 호스팅 배포에는 LangGraph Control Plane에 대한 커스텀 호스트가 있습니다.

## 인증

LangGraph Control Plane API로 인증하려면 `X-Api-Key` 헤더를 유효한 LangSmith API 키로 설정하세요.

예제 `curl` 명령:
```shell
curl --request GET \
  --url http://localhost:8124/v2/deployments \
  --header 'X-Api-Key: LANGSMITH_API_KEY'
```

## 버전 관리

각 엔드포인트 경로에는 버전이 접두사로 붙습니다(예: `v1`, `v2`).

## 빠른 시작

1. `POST /v2/deployments`를 호출하여 새 배포를 생성합니다. 응답 본문에는 배포 ID(`id`)와 최신(첫 번째) 리비전의 ID(`latest_revision_id`)가 포함됩니다.
1. `GET /v2/deployments/{deployment_id}`를 호출하여 배포를 검색합니다. URL의 `deployment_id`를 배포 ID(`id`) 값으로 설정합니다.
1. `GET /v2/deployments/{deployment_id}/revisions/{latest_revision_id}`를 호출하여 리비전 `status`가 `DEPLOYED`가 될 때까지 폴링합니다.
1. `PATCH /v2/deployments/{deployment_id}`를 호출하여 배포를 업데이트합니다.

## 예제 코드
다음은 LangGraph Control Plane API를 조율하여 배포를 생성, 업데이트 및 삭제하는 방법을 보여주는 Python 예제 코드입니다.
```python
import os
import time

import requests
from dotenv import load_dotenv


load_dotenv()

# 필수 환경 변수
CONTROL_PLANE_HOST = os.getenv("CONTROL_PLANE_HOST")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
INTEGRATION_ID = os.getenv("INTEGRATION_ID")
MAX_WAIT_TIME = 1800  # 30분


def get_headers() -> dict:
    """LangGraph Control Plane API 요청에 대한 공통 헤더를 반환합니다."""
    return {
        "X-Api-Key": LANGSMITH_API_KEY,
    }


def create_deployment() -> str:
    """배포를 생성합니다. 배포 ID를 반환합니다."""
    headers = get_headers()
    headers["Content-Type"] = "application/json"

    deployment_name = "my_deployment"

    request_body = {
        "name": deployment_name,
        "source": "github",
        "source_config": {
            "integration_id": INTEGRATION_ID,
            "repo_url": "https://github.com/langchain-ai/langgraph-example",
            "deployment_type": "dev",
            "build_on_push": False,
            "custom_url": None,
            "resource_spec": None,
        },
        "source_revision_config": {
            "repo_ref": "main",
            "langgraph_config_path": "langgraph.json",
            "image_uri": None,
        },
        "secrets": [
            {
                "name": "OPENAI_API_KEY",
                "value": "test_openai_api_key",
            },
            {
                "name": "ANTHROPIC_API_KEY",
                "value": "test_anthropic_api_key",
            },
            {
                "name": "TAVILY_API_KEY",
                "value": "test_tavily_api_key",
            },
        ],
    }

    response = requests.post(
        url=f"{CONTROL_PLANE_HOST}/v2/deployments",
        headers=headers,
        json=request_body,
    )

    if response.status_code != 201:
        raise Exception(f"Failed to create deployment: {response.text}")

    deployment_id = response.json()["id"]
    print(f"Created deployment {deployment_name} ({deployment_id})")
    return deployment_id


def get_deployment(deployment_id: str) -> dict:
    """배포를 가져옵니다."""
    response = requests.get(
        url=f"{CONTROL_PLANE_HOST}/v2/deployments/{deployment_id}",
        headers=get_headers(),
    )

    if response.status_code != 200:
        raise Exception(f"Failed to get deployment ID {deployment_id}: {response.text}")

    return response.json()


def list_revisions(deployment_id: str) -> list[dict]:
    """리비전 목록을 가져옵니다.

    반환 목록은 created_at을 기준으로 내림차순(최신 순)으로 정렬됩니다.
    """
    response = requests.get(
        url=f"{CONTROL_PLANE_HOST}/v2/deployments/{deployment_id}/revisions",
        headers=get_headers(),
    )

    if response.status_code != 200:
        raise Exception(
            f"Failed to list revisions for deployment ID {deployment_id}: {response.text}"
        )

    return response.json()


def get_revision(
    deployment_id: str,
    revision_id: str,
) -> dict:
    """리비전을 가져옵니다."""
    response = requests.get(
        url=f"{CONTROL_PLANE_HOST}/v2/deployments/{deployment_id}/revisions/{revision_id}",
        headers=get_headers(),
    )

    if response.status_code != 200:
        raise Exception(f"Failed to get revision ID {revision_id}: {response.text}")

    return response.json()


def patch_deployment(deployment_id: str) -> None:
    """배포를 패치합니다."""
    headers = get_headers()
    headers["Content-Type"] = "application/json"

    response = requests.patch(
        url=f"{CONTROL_PLANE_HOST}/v2/deployments/{deployment_id}",
        headers=headers,
        json={
            "source_config": {
                "build_on_push": True,
            },
            "source_revision_config": {
                "repo_ref": "main",
                "langgraph_config_path": "langgraph.json",
            },
        },
    )

    if response.status_code != 200:
        raise Exception(f"Failed to patch deployment: {response.text}")

    print(f"Patched deployment ID {deployment_id}")


def wait_for_deployment(deployment_id: str, revision_id: str) -> None:
    """리비전 상태가 DEPLOYED가 될 때까지 기다립니다."""
    start_time = time.time()
    revision, status = None, None
    while time.time() - start_time < MAX_WAIT_TIME:
        revision = get_revision(deployment_id, revision_id)
        status = revision["status"]
        if status == "DEPLOYED":
            break
        elif "FAILED" in status:
            raise Exception(f"Revision ID {revision_id} failed: {revision}")

        print(f"Waiting for revision ID {revision_id} to be DEPLOYED...")
        time.sleep(60)

    if status != "DEPLOYED":
        raise Exception(
            f"Timeout waiting for revision ID {revision_id} to be DEPLOYED: {revision}"
        )


def delete_deployment(deployment_id: str) -> None:
    """배포를 삭제합니다."""
    response = requests.delete(
        url=f"{CONTROL_PLANE_HOST}/v2/deployments/{deployment_id}",
        headers=get_headers(),
    )

    if response.status_code != 204:
        raise Exception(
            f"Failed to delete deployment ID {deployment_id}: {response.text}"
        )

    print(f"Deployment ID {deployment_id} deleted")


if __name__ == "__main__":
    # 배포 생성 및 최신 리비전 가져오기
    deployment_id = create_deployment()
    revisions = list_revisions(deployment_id)
    latest_revision = revisions["resources"][0]
    latest_revision_id = latest_revision["id"]

    # 최신 리비전이 DEPLOYED 상태가 될 때까지 대기
    wait_for_deployment(deployment_id, latest_revision_id)

    # 배포 패치 및 최신 리비전 가져오기
    patch_deployment(deployment_id)
    revisions = list_revisions(deployment_id)
    latest_revision = revisions["resources"][0]
    latest_revision_id = latest_revision["id"]

    # 최신 리비전이 DEPLOYED 상태가 될 때까지 대기
    wait_for_deployment(deployment_id, latest_revision_id)

    # 배포 삭제
    delete_deployment(deployment_id)
```