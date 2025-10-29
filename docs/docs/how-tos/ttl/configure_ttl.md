# LangGraph 애플리케이션에 TTL을 추가하는 방법

!!! tip "사전 요구사항"

    이 가이드는 [LangGraph Platform](../../concepts/langgraph_platform.md), [지속성](../../concepts/persistence.md), 및 [Cross-thread 지속성](../../concepts/persistence.md#memory-store) 개념에 익숙하다고 가정합니다.

???+ note "LangGraph platform 전용"

    TTL은 LangGraph Platform 배포에서만 지원됩니다. 이 가이드는 LangGraph OSS에는 적용되지 않습니다.

LangGraph Platform은 [checkpoint](../../concepts/persistence.md#checkpoints)(thread 상태)와 [cross-thread memories](../../concepts/persistence.md#memory-store)(store 항목) 모두를 지속합니다. `langgraph.json`에서 Time-to-Live(TTL) 정책을 구성하여 이 데이터의 라이프사이클을 자동으로 관리하고 무한정 축적을 방지할 수 있습니다.

## Checkpoint TTL 구성

Checkpoint는 대화 thread의 상태를 캡처합니다. TTL을 설정하면 오래된 checkpoint와 thread가 자동으로 삭제됩니다.

`langgraph.json` 파일에 `checkpointer.ttl` 구성을 추가합니다:

:::python
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "checkpointer": {
    "ttl": {
      "strategy": "delete",
      "sweep_interval_minutes": 60,
      "default_ttl": 43200
    }
  }
}
```
:::

:::js
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.ts:graph"
  },
  "checkpointer": {
    "ttl": {
      "strategy": "delete",
      "sweep_interval_minutes": 60,
      "default_ttl": 43200
    }
  }
}
```
:::

*   `strategy`: 만료 시 수행되는 작업을 지정합니다. 현재 `"delete"`만 지원되며, 만료 시 thread의 모든 checkpoint를 삭제합니다.
*   `sweep_interval_minutes`: 시스템이 만료된 checkpoint를 확인하는 빈도를 분 단위로 정의합니다.
*   `default_ttl`: checkpoint의 기본 수명을 분 단위로 설정합니다(예: 43200분 = 30일).

## Store Item TTL 구성

Store 항목은 cross-thread 데이터 지속성을 허용합니다. Store 항목에 대한 TTL을 구성하면 오래된 데이터를 제거하여 메모리를 관리하는 데 도움이 됩니다.

`langgraph.json` 파일에 `store.ttl` 구성을 추가합니다:

:::python
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 120,
      "default_ttl": 10080
    }
  }
}
```
:::

:::js
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.ts:graph"
  },
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 120,
      "default_ttl": 10080
    }
  }
}
```
:::

*   `refresh_on_read`: (선택 사항, 기본값 `true`) `true`인 경우, `get` 또는 `search`를 통해 항목에 액세스하면 만료 타이머가 재설정됩니다. `false`인 경우, TTL은 `put`에서만 새로 고쳐집니다.
*   `sweep_interval_minutes`: (선택 사항) 시스템이 만료된 항목을 확인하는 빈도를 분 단위로 정의합니다. 생략하면 스위핑이 발생하지 않습니다.
*   `default_ttl`: (선택 사항) store 항목의 기본 수명을 분 단위로 설정합니다(예: 10080분 = 7일). 생략하면 항목은 기본적으로 만료되지 않습니다.

## TTL 구성 결합

동일한 `langgraph.json` 파일에서 checkpoint와 store 항목 모두에 대한 TTL을 구성하여 각 데이터 유형에 대해 서로 다른 정책을 설정할 수 있습니다. 다음은 예제입니다:

:::python
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "checkpointer": {
    "ttl": {
      "strategy": "delete",
      "sweep_interval_minutes": 60,
      "default_ttl": 43200
    }
  },
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 120,
      "default_ttl": 10080
    }
  }
}
```
:::

:::js
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.ts:graph"
  },
  "checkpointer": {
    "ttl": {
      "strategy": "delete",
      "sweep_interval_minutes": 60,
      "default_ttl": 43200
    }
  },
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 120,
      "default_ttl": 10080
    }
  }
}
```
:::

## 런타임 재정의

`langgraph.json`의 기본 `store.ttl` 설정은 `get`, `put`, `search`와 같은 SDK 메서드 호출에서 특정 TTL 값을 제공하여 런타임에 재정의할 수 있습니다.

## 배포 프로세스

`langgraph.json`에서 TTL을 구성한 후, 변경 사항이 적용되도록 LangGraph 애플리케이션을 배포하거나 다시 시작합니다. 로컬 개발에는 `langgraph dev`를 사용하고, Docker 배포에는 `langgraph up`을 사용합니다.

다른 구성 가능한 옵션에 대한 자세한 내용은 @[langgraph.json CLI reference][langgraph.json]를 참조하세요.
