# 셀프 호스팅 Control Plane

셀프 호스팅 배포에는 두 가지 버전이 있습니다: [Self-Hosted Data Plane](./deployment_options.md#self-hosted-data-plane)과 [Self-Hosted Control Plane](./deployment_options.md#self-hosted-control-plane).

!!! info "중요"

    Self-Hosted Control Plane 배포 옵션에는 [Enterprise](plans.md) 플랜이 필요합니다.

## 요구 사항

- [LangGraph CLI](./langgraph_cli.md) 및/또는 [LangGraph Studio](./langgraph_studio.md) 앱을 사용하여 로컬에서 그래프를 테스트합니다.
- `langgraph build` 명령을 사용하여 이미지를 빌드합니다.
- Self-Hosted LangSmith 인스턴스가 배포되어 있어야 합니다.
- LangSmith 인스턴스에 Ingress를 사용하고 있어야 합니다. 모든 에이전트는 이 Ingress 뒤의 Kubernetes 서비스로 배포됩니다.

## Self-Hosted Control Plane

[Self-Hosted Control Plane](./langgraph_self_hosted_control_plane.md) 배포 옵션은 귀하의 클라우드에서 [control plane](./langgraph_control_plane.md)과 [data plane](./langgraph_data_plane.md)을 관리하는 완전 셀프 호스팅 배포 모델입니다. 이 옵션은 Control plane 및 Data plane 인프라에 대한 완전한 제어 및 책임을 제공합니다.

|                                    | [Control plane](../concepts/langgraph_control_plane.md)                                                                                     | [Data plane](../concepts/langgraph_data_plane.md)                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **무엇인가?**                       | <ul><li>배포 및 개정판 생성을 위한 Control plane UI</li><li>배포 및 개정판 생성을 위한 Control plane API</li></ul> | <ul><li>Control plane 상태와 배포를 조정하기 위한 Data plane "listener"</li><li>LangGraph 서버</li><li>Postgres, Redis 등</li></ul> |
| **어디에 호스팅되나?**              | 귀하의 클라우드                                                                                                                                  | 귀하의 클라우드                                                                                                                                          |
| **누가 프로비저닝하고 관리하나?**   | 귀하                                                                                                                                         | 귀하                                                                                                                                                 |

### 아키텍처

![Self-Hosted Control Plane Architecture](./img/self_hosted_control_plane_architecture.png)

### 컴퓨팅 플랫폼

- **Kubernetes**: Self-Hosted Control Plane 배포 옵션은 모든 Kubernetes 클러스터에 Control plane 및 Data plane 인프라를 배포하는 것을 지원합니다.

!!! tip
LangSmith 인스턴스에서 이를 활성화하려면 [Self-Hosted Control Plane 배포 가이드](../cloud/deployment/self_hosted_control_plane.md)를 따르세요.
