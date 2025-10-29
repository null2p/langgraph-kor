---
search:
  boost: 2
---

# Self-Hosted Data Plane

셀프 호스팅 배포에는 두 가지 버전이 있습니다: [Self-Hosted Data Plane](./deployment_options.md#self-hosted-data-plane)과 [Self-Hosted Control Plane](./deployment_options.md#self-hosted-control-plane).

!!! info "중요"

    Self-Hosted Data Plane 배포 옵션에는 [Enterprise](plans.md) 플랜이 필요합니다.

## 요구 사항

- `langgraph-cli` 및/또는 [LangGraph Studio](./langgraph_studio.md) 앱을 사용하여 로컬에서 그래프를 테스트합니다.
- `langgraph build` 명령을 사용하여 이미지를 빌드합니다.

## Self-Hosted Data Plane

[Self-Hosted Data Plane](../cloud/deployment/self_hosted_data_plane.md) 배포 옵션은 "하이브리드" 배포 모델로, 저희가 [Control plane](./langgraph_control_plane.md)을 클라우드에서 관리하고 귀하가 [Data plane](./langgraph_data_plane.md)을 귀하의 클라우드에서 관리합니다. 이 옵션은 Data plane 인프라를 안전하게 관리할 수 있는 방법을 제공하면서도 Control plane 관리를 저희에게 맡길 수 있습니다. Self-Hosted Data Plane 버전을 사용할 때는 [LangSmith](https://smith.langchain.com/) API 키로 인증합니다.

|                                    | [Control plane](../concepts/langgraph_control_plane.md)                                                                                     | [Data plane](../concepts/langgraph_data_plane.md)                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **무엇인가?**                    | <ul><li>배포 및 개정판 생성을 위한 Control plane UI</li><li>배포 및 개정판 생성을 위한 Control plane API</li></ul> | <ul><li>Control plane 상태와 배포를 조정하기 위한 Data plane "listener"</li><li>LangGraph 서버</li><li>Postgres, Redis 등</li></ul> |
| **어디에 호스팅되나?**            | LangChain의 클라우드                                                                                                                           | 귀하의 클라우드                                                                                                                                          |
| **누가 프로비저닝하고 관리하나?** | LangChain                                                                                                                                   | 귀하                                                                                                                                                 |

[LangGraph Server](../concepts/langgraph_server.md)를 Self-Hosted Data Plane에 배포하는 방법에 대한 정보는 [Self-Hosted Data Plane 배포 가이드](../cloud/deployment/self_hosted_data_plane.md)를 참조하세요.

### 아키텍처

![Self-Hosted Data Plane Architecture](./img/self_hosted_data_plane_architecture.png)

### 컴퓨팅 플랫폼

- **Kubernetes**: Self-Hosted Data Plane 배포 옵션은 모든 Kubernetes 클러스터에 Data plane 인프라를 배포하는 것을 지원합니다.
- **Amazon ECS**: 곧 출시!

!!! tip
Kubernetes에 배포하고 싶다면 [Self-Hosted Data Plane 배포 가이드](../cloud/deployment/self_hosted_data_plane.md)를 따르세요.
