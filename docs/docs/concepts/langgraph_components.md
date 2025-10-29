## 컴포넌트

LangGraph Platform은 LangGraph 애플리케이션의 개발, 배포, 디버깅 및 모니터링을 지원하기 위해 함께 작동하는 컴포넌트로 구성됩니다:

- [LangGraph Server](./langgraph_server.md): 서버는 에이전틱 애플리케이션 배포를 위한 모범 사례를 통합하는 의견이 반영된 API 및 아키텍처를 정의하여, 서버 인프라 개발이 아닌 에이전트 로직 구축에 집중할 수 있게 합니다.
- [LangGraph CLI](./langgraph_cli.md): LangGraph CLI는 로컬 LangGraph와 상호작용하는 데 도움이 되는 명령줄 인터페이스입니다.
- [LangGraph Studio](./langgraph_studio.md): LangGraph Studio는 LangGraph Server에 연결하여 로컬에서 애플리케이션의 시각화, 상호작용 및 디버깅을 가능하게 하는 전문 IDE입니다.
- [Python/JS SDK](./sdk.md): Python/JS SDK는 배포된 LangGraph 애플리케이션과 프로그래밍 방식으로 상호작용하는 방법을 제공합니다.
- [Remote Graph](../how-tos/use-remote-graph.md): RemoteGraph를 사용하면 배포된 LangGraph 애플리케이션과 로컬에서 실행되는 것처럼 상호작용할 수 있습니다.
- [LangGraph control plane](./langgraph_control_plane.md): LangGraph Control Plane은 사용자가 LangGraph Server를 생성하고 업데이트하는 Control Plane UI와 UI 경험을 지원하는 Control Plane API를 의미합니다.
- [LangGraph data plane](./langgraph_data_plane.md): LangGraph Data Plane은 LangGraph Server, 각 서버에 대한 해당 인프라 및 LangGraph Control Plane의 업데이트를 지속적으로 폴링하는 "listener" 애플리케이션을 의미합니다.

![LangGraph components](img/lg_platform.png)
