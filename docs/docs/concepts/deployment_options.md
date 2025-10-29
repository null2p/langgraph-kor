---
search:
  boost: 2
---

# 배포 옵션

## 무료 배포

[로컬](../tutorials/langgraph-platform/local-server.md): 로컬 테스트 및 개발을 위한 배포

## 프로덕션 배포

[LangGraph Platform](langgraph_platform.md)을 통한 배포에는 4가지 주요 옵션이 있습니다:

1. [Cloud SaaS](#cloud-saas)

1. [Self-Hosted Data Plane](#self-hosted-data-plane)

1. [Self-Hosted Control Plane](#self-hosted-control-plane)

1. [Standalone Container](#standalone-container)


빠른 비교:

|                      | **Cloud SaaS** | **Self-Hosted Data Plane** | **Self-Hosted Control Plane** | **Standalone Container** |
|----------------------|----------------|----------------------------|-------------------------------|--------------------------|
| **[Control plane UI/API](../concepts/langgraph_control_plane.md)** | Yes | Yes | Yes | No |
| **CI/CD** | 플랫폼에서 내부 관리 | 귀하가 외부에서 관리 | 귀하가 외부에서 관리 | 귀하가 외부에서 관리 |
| **데이터/컴퓨팅 상주** | LangChain의 클라우드 | 귀하의 클라우드 | 귀하의 클라우드 | 귀하의 클라우드 |
| **LangSmith 호환성** | LangSmith SaaS로 추적 | LangSmith SaaS로 추적 | Self-Hosted LangSmith로 추적 | 선택적 추적 |
| **[가격 책정](https://www.langchain.com/pricing-langgraph-platform)** | Plus | Enterprise | Enterprise | Enterprise |

## Cloud SaaS

[Cloud SaaS](./langgraph_cloud.md) 배포 옵션은 저희가 [Control plane](./langgraph_control_plane.md)과 [Data plane](./langgraph_data_plane.md)을 클라우드에서 관리하는 완전 관리형 배포 모델입니다. 이 옵션은 LangGraph 서버를 배포하고 관리하는 간단한 방법을 제공합니다.

GitHub 저장소를 플랫폼에 연결하고 [Control plane UI](./langgraph_control_plane.md#control-plane-ui)에서 LangGraph 서버를 배포합니다. 빌드 프로세스(즉, CI/CD)는 플랫폼에서 내부적으로 관리됩니다.

자세한 내용은 다음을 참조하세요:

* [Cloud SaaS 개념 가이드](./langgraph_cloud.md)
* [Cloud SaaS 배포 방법](../cloud/deployment/cloud.md)

## Self-Hosted Data Plane

!!! info "중요"
    Self-Hosted Data Plane 배포 옵션에는 [Enterprise](../concepts/plans.md) 플랜이 필요합니다.

[Self-Hosted Data Plane](./langgraph_self_hosted_data_plane.md) 배포 옵션은 "하이브리드" 배포 모델로, 저희가 [Control plane](./langgraph_control_plane.md)을 클라우드에서 관리하고 귀하가 [Data plane](./langgraph_data_plane.md)을 귀하의 클라우드에서 관리합니다. 이 옵션은 Data plane 인프라를 안전하게 관리할 수 있는 방법을 제공하면서도 Control plane 관리를 저희에게 맡길 수 있습니다.

[LangGraph CLI](./langgraph_cli.md)를 사용하여 Docker 이미지를 빌드하고 [Control plane UI](./langgraph_control_plane.md#control-plane-ui)에서 LangGraph 서버를 배포합니다.

지원되는 컴퓨팅 플랫폼: [Kubernetes](https://kubernetes.io/), [Amazon ECS](https://aws.amazon.com/ecs/) (곧 출시!)

자세한 내용은 다음을 참조하세요:

* [Self-Hosted Data Plane 개념 가이드](./langgraph_self_hosted_data_plane.md)
* [Self-Hosted Data Plane 배포 방법](../cloud/deployment/self_hosted_data_plane.md)

## Self-Hosted Control Plane

!!! info "중요"
    Self-Hosted Control Plane 배포 옵션에는 [Enterprise](../concepts/plans.md) 플랜이 필요합니다.

[Self-Hosted Control Plane](./langgraph_self_hosted_control_plane.md) 배포 옵션은 귀하가 [Control plane](./langgraph_control_plane.md)과 [Data plane](./langgraph_data_plane.md)을 귀하의 클라우드에서 관리하는 완전 셀프 호스팅 배포 모델입니다. 이 옵션은 Control plane 및 Data plane 인프라에 대한 완전한 제어 및 책임을 제공합니다.

[LangGraph CLI](./langgraph_cli.md)를 사용하여 Docker 이미지를 빌드하고 [Control plane UI](./langgraph_control_plane.md#control-plane-ui)에서 LangGraph 서버를 배포합니다.

지원되는 컴퓨팅 플랫폼: [Kubernetes](https://kubernetes.io/)

자세한 내용은 다음을 참조하세요:

* [Self-Hosted Control Plane 개념 가이드](./langgraph_self_hosted_control_plane.md)
* [Self-Hosted Control Plane 배포 방법](../cloud/deployment/self_hosted_control_plane.md)

## Standalone Container

[Standalone Container](./langgraph_standalone_container.md) 배포 옵션은 가장 제한이 적은 배포 모델입니다. [사용 가능한](./plans.md) 라이선스 옵션을 사용하여 귀하의 클라우드에 LangGraph 서버의 독립 실행형 인스턴스를 배포합니다.

[LangGraph CLI](./langgraph_cli.md)를 사용하여 Docker 이미지를 빌드하고 선택한 컨테이너 배포 도구를 사용하여 LangGraph 서버를 배포합니다. 이미지는 모든 컴퓨팅 플랫폼에 배포할 수 있습니다.

자세한 내용은 다음을 참조하세요:

* [Standalone Container 개념 가이드](./langgraph_standalone_container.md)
* [Standalone Container 배포 방법](../cloud/deployment/standalone_container.md)

## 관련 정보

자세한 내용은 다음을 참조하세요:

* [LangGraph Platform 플랜](./plans.md)
* [LangGraph Platform 가격 책정](https://www.langchain.com/langgraph-platform-pricing)
