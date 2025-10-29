---
search:
  boost: 2
---

# Standalone Container

[LangGraph Server](../concepts/langgraph_server.md)를 배포하려면 [Standalone Container 배포 방법](../cloud/deployment/standalone_container.md) 가이드를 참조하세요.

## 개요

Standalone Container 배포 옵션은 가장 제한이 적은 배포 모델입니다. [Control plane](./langgraph_control_plane.md)이 없으며, [Data plane](./langgraph_data_plane.md) 인프라는 귀하가 직접 관리합니다.

|                   | [Control plane](../concepts/langgraph_control_plane.md) | [Data plane](../concepts/langgraph_data_plane.md) |
|-------------------|-------------------|------------|
| **무엇인가?** | n/a | <ul><li>LangGraph 서버</li><li>Postgres, Redis 등</li></ul> |
| **어디에 호스팅되나?** | n/a | 귀하의 클라우드 |
| **누가 프로비저닝하고 관리하나?** | n/a | 귀하 |

!!! warning

      LangGraph Platform은 서버리스 환경에 배포해서는 안 됩니다. Scale to zero는 작업 손실을 초래할 수 있으며, 스케일 업이 안정적으로 작동하지 않을 수 있습니다.

## 아키텍처

![Standalone Container](./img/langgraph_platform_deployment_architecture.png)

## 컴퓨팅 플랫폼

### Kubernetes

Standalone Container 배포 옵션은 Kubernetes 클러스터에 Data plane 인프라를 배포하는 것을 지원합니다.

### Docker

Standalone Container 배포 옵션은 Docker를 지원하는 모든 컴퓨팅 플랫폼에 Data plane 인프라를 배포하는 것을 지원합니다.
