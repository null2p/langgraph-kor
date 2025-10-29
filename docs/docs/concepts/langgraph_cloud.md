---
search:
  boost: 2
---

# Cloud SaaS

[LangGraph Server](../concepts/langgraph_server.md)를 배포하려면 [Cloud SaaS에 배포하는 방법](../cloud/deployment/cloud.md)에 대한 how-to 가이드를 따르세요.

## 개요

Cloud SaaS 배포 옵션은 우리가 클라우드에서 [control plane](./langgraph_control_plane.md)과 [data plane](./langgraph_data_plane.md)을 관리하는 완전 관리형 배포 모델입니다.

|                                    | [Control plane](../concepts/langgraph_control_plane.md)                                                                                     | [Data plane](../concepts/langgraph_data_plane.md)                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **무엇인가?**                       | <ul><li>배포 및 개정판 생성을 위한 Control plane UI</li><li>배포 및 개정판 생성을 위한 Control plane API</li></ul> | <ul><li>Control plane 상태와 배포를 조정하기 위한 Data plane "listener"</li><li>LangGraph 서버</li><li>Postgres, Redis 등</li></ul> |
| **어디에 호스팅되나?**              | LangChain 클라우드                                                                                                                           | LangChain 클라우드                                                                                                                                   |
| **누가 프로비저닝하고 관리하나?**   | LangChain                                                                                                                                   | LangChain                                                                                                                                           |

## 아키텍처

![Cloud SaaS](./img/self_hosted_control_plane_architecture.png)
