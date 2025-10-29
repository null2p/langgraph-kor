# 웹훅

Webhook은 LangGraph Platform 애플리케이션에서 외부 서비스로의 이벤트 기반 통신을 가능하게 합니다. 예를 들어, LangGraph Platform에 대한 API 호출이 실행을 완료하면 별도의 서비스에 업데이트를 발행할 수 있습니다.

많은 LangGraph Platform 엔드포인트는 `webhook` 매개변수를 허용합니다. POST 요청을 수락할 수 있는 엔드포인트에서 이 매개변수가 지정되면 LangGraph Platform은 실행 완료 시 요청을 보냅니다.

자세한 내용은 해당 [how-to 가이드](../../cloud/how-tos/webhooks.md)를 참조하세요.
