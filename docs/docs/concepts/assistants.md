# 어시스턴트

**어시스턴트**를 사용하면 그래프의 핵심 로직과 별도로 구성(프롬프트, LLM 선택, 도구 등)을 관리할 수 있어, 그래프 아키텍처를 변경하지 않고도 빠른 변경이 가능합니다. 이는 구조적 변경이 아닌 컨텍스트/구성 변형을 통해 서로 다른 사용 사례에 최적화된 동일한 그래프 아키텍처의 여러 특수 버전을 만드는 방법입니다.

예를 들어, 공통 그래프 아키텍처를 기반으로 구축된 범용 작성 에이전트를 상상해보세요. 구조는 동일하게 유지되지만, 블로그 게시물 및 트윗과 같은 다양한 작성 스타일은 성능을 최적화하기 위해 맞춤형 구성이 필요합니다. 이러한 변형을 지원하기 위해 기본 그래프를 공유하지만 모델 선택 및 시스템 프롬프트가 다른 여러 어시스턴트(예: 블로그용 하나, 트윗용 하나)를 만들 수 있습니다.

![assistant versions](img/assistants.png)

LangGraph Cloud API는 어시스턴트 및 해당 버전을 생성하고 관리하기 위한 여러 엔드포인트를 제공합니다. 자세한 내용은 [API 레퍼런스](../cloud/reference/api/api_ref.html#tag/assistants)를 참조하세요.

!!! info

    어시스턴트는 [LangGraph Platform](langgraph_platform.md) 개념입니다. 오픈 소스 LangGraph 라이브러리에서는 사용할 수 없습니다.

## 구성

:::python
어시스턴트는 구성 및 [런타임 컨텍스트](low_level.md#runtime-context)의 LangGraph 오픈 소스 개념을 기반으로 합니다.
:::

이러한 기능은 오픈 소스 LangGraph 라이브러리에서 사용할 수 있지만, 어시스턴트는 [LangGraph Platform](langgraph_platform.md)에만 존재합니다. 이는 어시스턴트가 배포된 그래프와 긴밀하게 결합되어 있기 때문입니다. 배포 시 LangGraph Server는 그래프의 기본 컨텍스트 및 구성 설정을 사용하여 각 그래프에 대한 기본 어시스턴트를 자동으로 생성합니다.

실제로 어시스턴트는 특정 구성을 가진 그래프의 _인스턴스_일 뿐입니다. 따라서 여러 어시스턴트가 동일한 그래프를 참조할 수 있지만 서로 다른 구성(예: 프롬프트, 모델, 도구)을 포함할 수 있습니다. LangGraph Server API는 어시스턴트를 생성하고 관리하기 위한 여러 엔드포인트를 제공합니다. 어시스턴트를 만드는 방법에 대한 자세한 내용은 [API 레퍼런스](../cloud/reference/api/api_ref.html) 및 [이 how-to](../cloud/how-tos/configuration_cloud.md)를 참조하세요.

## 버전 관리

어시스턴트는 시간이 지남에 따라 변경 사항을 추적하기 위한 버전 관리를 지원합니다.
어시스턴트를 만든 후 해당 어시스턴트에 대한 후속 편집은 새 버전을 생성합니다. 어시스턴트 버전을 관리하는 방법에 대한 자세한 내용은 [이 how-to](../cloud/how-tos/configuration_cloud.md#create-a-new-version-for-your-assistant)를 참조하세요.

## 실행

**run**은 어시스턴트의 호출입니다. 각 run은 고유한 입력, 구성, 컨텍스트 및 메타데이터를 가질 수 있으며, 이는 기본 그래프의 실행 및 출력에 영향을 줄 수 있습니다. run은 선택적으로 [thread](./persistence.md#threads)에서 실행될 수 있습니다.

LangGraph Platform API는 run을 생성하고 관리하기 위한 여러 엔드포인트를 제공합니다. 자세한 내용은 [API 레퍼런스](../cloud/reference/api/api_ref.html#tag/thread-runs/)를 참조하세요.
