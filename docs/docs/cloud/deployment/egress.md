# 구독 메트릭 및 운영 메타데이터를 위한 Egress

> **중요: Self Hosted 전용**
> 이 섹션은 오프라인 모드에서 실행하지 않는 고객에게만 적용되며, self-hosted LangGraph Platform 인스턴스를 사용한다고 가정합니다.
> 이는 SaaS 또는 Hybrid 배포에는 적용되지 않습니다.

Self-Hosted LangGraph Platform 인스턴스는 모든 정보를 로컬에 저장하며 네트워크 외부로 민감한 정보를 절대 전송하지 않습니다. 현재 주문서의 권한에 따라 청구 목적으로만 플랫폼 사용량을 추적합니다. 고객을 원격으로 더 잘 지원하기 위해 `https://beacon.langchain.com`로의 egress가 필요합니다.

향후 LangGraph Platform이 환경 내에서 최적의 수준으로 실행되고 있는지 확인하는 데 도움이 되는 지원 진단을 도입할 예정입니다.

> **경고**
> **네트워크에서 `https://beacon.langchain.com`로의 egress가 필요합니다.**
> **API 키를 사용하는 경우 API 키 확인을 위해 `https://api.smith.langchain.com` 또는 `https://eu.api.smith.langchain.com`로의 egress도 허용해야 합니다.**

일반적으로 Beacon에 전송하는 데이터는 다음과 같이 분류할 수 있습니다:

- **구독 메트릭**
  - 구독 메트릭은 LangSmith의 액세스 수준 및 활용도를 결정하는 데 사용됩니다. 여기에는 다음이 포함되지만 이에 국한되지 않습니다:
    - 실행된 노드 수
    - 실행된 Run 수
    - 라이선스 키 확인
- **운영 메타데이터**
  - 이 메타데이터는 원격 지원을 지원하기 위해 위의 구독 메트릭을 포함하고 수집하며, LangChain 팀이 성능 문제를 보다 효과적이고 사전에 진단 및 해결할 수 있도록 합니다.

## Example Payloads

투명성을 최대화하기 위해 여기에 샘플 페이로드를 제공합니다:

### License Verification (Enterprise 라이선스 사용 시)

**Endpoint:**

`POST beacon.langchain.com/v1/beacon/verify`

**Request:**

```json
{
  "license": "<YOUR_LICENSE_KEY>"
}
```

**Response:**

```json
{
  "token": "Valid JWT" // Short-lived JWT token to avoid repeated license checks
}
```

### Api Key Verification (LangSmith API 키 사용 시)

**Endpoint:**
`POST api.smith.langchain.com/auth`

**Request:**

```json
"Headers": {
  X-Api-Key: <YOUR_API_KEY>
}
```

**Response:**

```json
{
  "org_config": {
    "org_id": "3a1c2b6f-4430-4b92-8a5b-79b8b567bbc1",
    ... // Additional organization details
  }
}
```

### Usage Reporting

**Endpoint:**

`POST beacon.langchain.com/v1/metadata/submit`

**Request:**

```json
{
  "license": "<YOUR_LICENSE_KEY>",
  "from_timestamp": "2025-01-06T09:00:00Z",
  "to_timestamp": "2025-01-06T10:00:00Z",
  "tags": {
    "langgraph.python.version": "0.1.0",
    "langgraph_api.version": "0.2.0",
    "langgraph.platform.revision": "abc123",
    "langgraph.platform.variant": "standard",
    "langgraph.platform.host": "host-1",
    "langgraph.platform.tenant_id": "3a1c2b6f-4430-4b92-8a5b-79b8b567bbc1",
    "langgraph.platform.project_id": "c5b5f53a-4716-4326-8967-d4f7f7799735",
    "langgraph.platform.plan": "enterprise",
    "user_app.uses_indexing": "true",
    "user_app.uses_custom_app": "false",
    "user_app.uses_custom_auth": "true",
    "user_app.uses_thread_ttl": "true",
    "user_app.uses_store_ttl": "false"
  },
  "measures": {
    "langgraph.platform.runs": 150,
    "langgraph.platform.nodes": 450
  },
  "logs": []
}
```

**Response:**

```json
"204 No Content"
```

## Our Commitment

LangChain은 구독 메트릭 또는 운영 메타데이터에 민감한 정보를 저장하지 않습니다. 수집된 모든 데이터는 제3자와 공유되지 않습니다. 전송되는 데이터에 대해 우려가 있는 경우 계정 팀에 문의하세요.
