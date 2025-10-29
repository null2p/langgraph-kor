---
search:
  boost: 2
---

# FAQ

일반적인 질문과 답변입니다!

## LangGraph를 사용하려면 LangChain을 사용해야 하나요? 차이점은 무엇인가요?

아니요. LangGraph는 복잡한 에이전트 시스템을 위한 오케스트레이션 프레임워크이며 LangChain 에이전트보다 더 저수준이고 제어 가능합니다. LangChain은 모델 및 기타 구성 요소와 상호작용하기 위한 표준 인터페이스를 제공하며 간단한 체인 및 검색 흐름에 유용합니다.

## LangGraph는 다른 에이전트 프레임워크와 어떻게 다른가요?

다른 에이전트 프레임워크는 간단하고 일반적인 작업에는 작동할 수 있지만 복잡한 작업에는 부족합니다. LangGraph는 단일 블랙박스 인지 아키텍처로 제한하지 않고 고유한 작업을 처리할 수 있는 더 표현력이 풍부한 프레임워크를 제공합니다.

## LangGraph가 내 앱의 성능에 영향을 미치나요?

LangGraph는 코드에 어떤 오버헤드도 추가하지 않으며 스트리밍 워크플로를 염두에 두고 특별히 설계되었습니다.

## LangGraph는 오픈 소스인가요? 무료인가요?

예. LangGraph는 MIT 라이선스 오픈 소스 라이브러리이며 무료로 사용할 수 있습니다.

## LangGraph와 LangGraph Platform은 어떻게 다른가요?

LangGraph는 에이전트 워크플로에 향상된 제어를 제공하는 상태 기반 오케스트레이션 프레임워크입니다. LangGraph Platform은 LangGraph 애플리케이션을 배포하고 확장하기 위한 서비스로, 에이전트 UX를 구축하기 위한 의견이 반영된 API와 통합 개발자 스튜디오를 제공합니다.

| 기능            | LangGraph (오픈 소스)                                   | LangGraph Platform                                                                                     |
| ------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 설명         | 에이전트 애플리케이션을 위한 상태 기반 오케스트레이션 프레임워크 | LangGraph 애플리케이션 배포를 위한 확장 가능한 인프라                                           |
| SDK                | Python 및 JavaScript                                     | Python 및 JavaScript                                                                                  |
| HTTP API           | 없음                                                      | 있음 - 상태나 장기 메모리 검색 및 업데이트, 또는 구성 가능한 어시스턴트 생성에 유용 |
| 스트리밍           | 기본                                                     | 토큰별 메시지를 위한 전용 모드                                                             |
| Checkpointer        | 커뮤니티 기여                                     | 기본 지원                                                                               |
| 지속성 레이어   | 셀프 관리                                              | 효율적인 스토리지를 갖춘 관리형 Postgres                                                                |
| 배포          | 셀프 관리                                              | • Cloud SaaS <br> • 무료 셀프 호스팅 <br> • Enterprise (유료 셀프 호스팅)                              |
| 확장성         | 셀프 관리                                              | 작업 큐 및 서버의 자동 확장                                                                |
| 장애 허용     | 셀프 관리                                              | 자동 재시도                                                                                      |
| 동시성 제어 | 단순 스레딩                                          | Double-texting 지원                                                                                |
| 스케줄링          | 없음                                                      | Cron 스케줄링                                                                                        |
| 모니터링          | 없음                                                      | 관찰 가능성을 위한 LangSmith 통합                                                            |
| IDE 통합     | LangGraph Studio                                          | LangGraph Studio                                                                                       |

## LangGraph Platform은 오픈 소스인가요?

아니요. LangGraph Platform은 독점 소프트웨어입니다.

기본 기능에 액세스할 수 있는 무료 셀프 호스팅 버전의 LangGraph Platform이 있습니다. Cloud SaaS 배포 옵션과 Self-Hosted 배포 옵션은 유료 서비스입니다. 자세한 내용은 [영업팀에 문의](https://www.langchain.com/contact-sales)하세요.

자세한 내용은 [LangGraph Platform 가격 책정 페이지](https://www.langchain.com/pricing-langgraph-platform)를 참조하세요.

## LangGraph는 툴 호출을 지원하지 않는 LLM과 함께 작동하나요?

예! 모든 LLM과 함께 LangGraph를 사용할 수 있습니다. 우리가 툴 호출을 지원하는 LLM을 사용하는 주요 이유는 이것이 LLM이 무엇을 할지 결정하도록 하는 가장 편리한 방법이기 때문입니다. LLM이 툴 호출을 지원하지 않는 경우에도 사용할 수 있습니다 - 원시 LLM 문자열 응답을 무엇을 할지에 대한 결정으로 변환하는 약간의 로직을 작성하기만 하면 됩니다.

## LangGraph는 OSS LLM과 함께 작동하나요?

예! LangGraph는 내부에서 어떤 LLM이 사용되는지에 대해 완전히 중립적입니다. 대부분의 튜토리얼에서 폐쇄형 LLM을 사용하는 주요 이유는 이들이 툴 호출을 원활하게 지원하는 반면 OSS LLM은 종종 그렇지 않기 때문입니다. 하지만 툴 호출은 필수가 아니므로([이 섹션](#does-langgraph-work-with-llms-that-dont-support-tool-calling) 참조) OSS LLM과 함께 LangGraph를 완전히 사용할 수 있습니다.

## LangSmith에 로그인하지 않고 LangGraph Studio를 사용할 수 있나요

예! [개발 버전의 LangGraph Server](../tutorials/langgraph-platform/local-server.md)를 사용하여 백엔드를 로컬에서 실행할 수 있습니다.
이것은 LangSmith의 일부로 호스팅되는 스튜디오 프론트엔드에 연결됩니다.
`LANGSMITH_TRACING=false` 환경 변수를 설정하면 LangSmith에 추적이 전송되지 않습니다.

## LangGraph Platform 사용량에서 "nodes executed"는 무엇을 의미하나요?

**Nodes Executed**는 애플리케이션 호출 중에 호출되고 성공적으로 완료된 LangGraph 애플리케이션의 노드 수의 총합입니다. 실행 중에 노드가 호출되지 않거나 오류 상태로 끝나면 이러한 노드는 계산되지 않습니다. 노드가 여러 번 호출되고 성공적으로 완료되면 각 발생이 계산됩니다.
