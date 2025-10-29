## Cron 작업

어시스턴트를 스케줄에 따라 실행하는 것이 유용한 상황이 많습니다.

예를 들어, 매일 실행되어 하루의 뉴스 요약을 이메일로 보내는 어시스턴트를 빌드한다고 가정해 봅시다. Cron 작업을 사용하여 매일 오후 8시에 어시스턴트를 실행할 수 있습니다.

LangGraph Platform은 사용자 정의 스케줄에 따라 실행되는 cron 작업을 지원합니다. 사용자는 스케줄, 어시스턴트 및 일부 입력을 지정합니다. 그런 다음 지정된 스케줄에 따라 서버는:

- 지정된 어시스턴트로 새 thread를 생성합니다
- 해당 thread에 지정된 입력을 보냅니다

매번 동일한 입력을 thread에 보냅니다. Cron 작업을 생성하는 방법은 [how-to 가이드](../../cloud/how-tos/cron_jobs.md)를 참조하세요.

LangGraph Platform API는 cron 작업을 생성하고 관리하기 위한 여러 엔드포인트를 제공합니다. 자세한 내용은 [API 레퍼런스](../../cloud/reference/api/api_ref.html#tag/runscreate/POST/threads/{thread_id}/runs/crons)를 참조하세요.
