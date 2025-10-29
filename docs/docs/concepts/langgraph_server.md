---
search:
  boost: 2
---

# LangGraph Server

**LangGraph Server**는 에이전트 기반 애플리케이션을 생성하고 관리하기 위한 API를 제공합니다. 특정 작업에 대해 구성된 에이전트인 [어시스턴트](assistants.md) 개념을 기반으로 구축되었으며, 내장된 [지속성](persistence.md#memory-store)과 **작업 큐**를 포함합니다. 이 다용도 API는 백그라운드 처리부터 실시간 상호작용에 이르기까지 광범위한 에이전트 애플리케이션 사용 사례를 지원합니다.

LangGraph Server를 사용하여 [어시스턴트](assistants.md), [thread](./persistence.md#threads), [run](./assistants.md#execution), [cron 작업](../cloud/concepts/cron_jobs.md), [webhook](../cloud/concepts/webhooks.md) 등을 생성하고 관리합니다.

!!! tip "API 레퍼런스"

    API 엔드포인트 및 데이터 모델에 대한 자세한 정보는 [LangGraph Platform API 레퍼런스 문서](../cloud/reference/api/api_ref.html)를 참조하세요.

## 애플리케이션 구조

LangGraph Server 애플리케이션을 배포하려면 배포하려는 그래프와 의존성 및 환경 변수와 같은 관련 구성 설정을 지정해야 합니다.

배포를 위해 LangGraph 애플리케이션을 구조화하는 방법을 알아보려면 [애플리케이션 구조](./application_structure.md) 가이드를 읽어보세요.

## 배포의 구성 요소

LangGraph Server를 배포할 때 하나 이상의 [그래프](#graphs), [지속성](persistence.md)을 위한 데이터베이스 및 작업 큐를 배포합니다.

### 그래프

LangGraph Server로 그래프를 배포할 때 [어시스턴트](assistants.md)의 "blueprint"를 배포하는 것입니다.

[어시스턴트](assistants.md)는 특정 구성 설정과 쌍을 이루는 그래프입니다. 동일한 그래프로 제공할 수 있는 다양한 사용 사례를 수용하기 위해 각각 고유한 설정을 가진 여러 어시스턴트를 그래프당 생성할 수 있습니다.

배포 시 LangGraph Server는 그래프의 기본 구성 설정을 사용하여 각 그래프에 대한 기본 어시스턴트를 자동으로 생성합니다.

!!! note

    우리는 종종 그래프를 [에이전트](agentic_concepts.md)를 구현하는 것으로 생각하지만, 그래프가 반드시 에이전트를 구현할 필요는 없습니다. 예를 들어, 그래프는 애플리케이션 제어 흐름에 영향을 미칠 수 있는 능력 없이 앞뒤 대화만 지원하는 간단한 챗봇을 구현할 수 있습니다. 실제로 애플리케이션이 더 복잡해지면 그래프는 종종 함께 작동하는 [여러 에이전트](./multi_agent.md)를 사용하는 더 복잡한 흐름을 구현합니다.

### 지속성 및 작업 큐

LangGraph Server는 [지속성](persistence.md)을 위한 데이터베이스와 작업 큐를 활용합니다.

현재 LangGraph Server의 데이터베이스로는 [Postgres](https://www.postgresql.org/)만 지원되고 작업 큐로는 [Redis](https://redis.io/)만 지원됩니다.

[LangGraph Platform](./langgraph_cloud.md)을 사용하여 배포하는 경우 이러한 구성 요소가 자동으로 관리됩니다. 자체 인프라에 LangGraph Server를 배포하는 경우 이러한 구성 요소를 직접 설정하고 관리해야 합니다.

이러한 구성 요소가 설정되고 관리되는 방법에 대한 자세한 내용은 [배포 옵션](./deployment_options.md) 가이드를 검토하세요.

## 더 알아보기

* LangGraph [애플리케이션 구조](./application_structure.md) 가이드는 배포를 위해 LangGraph 애플리케이션을 구조화하는 방법을 설명합니다.
* [LangGraph Platform API 레퍼런스](../cloud/reference/api/api_ref.html)는 API 엔드포인트 및 데이터 모델에 대한 자세한 정보를 제공합니다.
