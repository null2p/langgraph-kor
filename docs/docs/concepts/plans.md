---
search:
  boost: 2
---

# LangGraph Platform 플랜


## 개요
LangGraph Platform은 프로덕션 환경에서 에이전틱 애플리케이션을 배포하기 위한 솔루션입니다.
사용하기 위한 세 가지 다른 플랜이 있습니다.

- **Developer**: 모든 [LangSmith](https://smith.langchain.com/) 사용자가 이 플랜에 액세스할 수 있습니다. LangSmith 계정을 생성하기만 하면 이 플랜에 가입할 수 있습니다. 이를 통해 [로컬 배포](./deployment_options.md#free-deployment) 옵션에 액세스할 수 있습니다.
- **Plus**: [Plus 계정](https://docs.smith.langchain.com/administration/pricing)이 있는 모든 [LangSmith](https://smith.langchain.com/) 사용자가 이 플랜에 액세스할 수 있습니다. LangSmith 계정을 Plus 플랜 유형으로 업그레이드하기만 하면 이 플랜에 가입할 수 있습니다. 이를 통해 [Cloud](./deployment_options.md#cloud-saas) 배포 옵션에 액세스할 수 있습니다.
- **Enterprise**: 이것은 LangSmith 플랜과 별개입니다. [영업팀에 문의](https://www.langchain.com/contact-sales)하여 이 플랜에 가입할 수 있습니다. 이를 통해 모든 [배포 옵션](./deployment_options.md)에 액세스할 수 있습니다.


## 플랜 세부사항

|                                                                  | Developer                                   | Plus                                                  | Enterprise                                          |
|------------------------------------------------------------------|---------------------------------------------|-------------------------------------------------------|-----------------------------------------------------|
| 배포 옵션                                               | Local                          | Cloud SaaS                                         | <ul><li>Cloud SaaS</li><li>Self-Hosted Data Plane</li><li>Self-Hosted Control Plane</li><li>Standalone Container</li></ul> |
| 사용                                                            | 무료 | [가격 책정](https://www.langchain.com/langgraph-platform-pricing) 참조 | 커스텀                                              |
| 상태 및 대화 히스토리 검색 및 업데이트를 위한 API | ✅                                           | ✅                                                     | ✅                                                   |
| 장기 메모리 검색 및 업데이트를 위한 API                | ✅                                           | ✅                                                     | ✅                                                   |
| 수평 확장 가능한 작업 큐 및 서버                    | ✅                                           | ✅                                                     | ✅                                                   |
| 출력 및 중간 단계의 실시간 스트리밍            | ✅                                           | ✅                                                     | ✅                                                   |
| Assistants API (LangGraph 앱을 위한 구성 가능한 템플릿)       | ✅                                           | ✅                                                     | ✅                                                   |
| Cron 스케줄링                                                  | --                                          | ✅                                                     | ✅                                                   |
| 프로토타이핑을 위한 LangGraph Studio                                 | 	✅                                         | ✅                                                    | ✅                                                  |
| LangGraph API 호출을 위한 인증 및 권한 부여        | --                                          | 곧 출시!                                          | 곧 출시!                                        |
| LLM API 트래픽 감소를 위한 스마트 캐싱                       | --                                          | 곧 출시!                                          | 곧 출시!                                        |
| 상태를 위한 게시/구독 API                                  | --                                          | 곧 출시!                                          | 곧 출시!                                        |
| 스케줄링 우선순위 지정                                        | --                                          | 곧 출시!                                          | 곧 출시!                                        |

가격 정보는 [LangGraph Platform 가격 책정](https://www.langchain.com/langgraph-platform-pricing)을 참조하세요.

## 관련 정보

자세한 내용은 다음을 참조하세요:

* [배포 옵션 개념 가이드](./deployment_options.md)
* [LangGraph Platform 가격 책정](https://www.langchain.com/langgraph-platform-pricing)
* [LangSmith 플랜](https://docs.smith.langchain.com/administration/pricing)
