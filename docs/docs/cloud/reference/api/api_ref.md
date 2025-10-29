# LangGraph Server API 레퍼런스

LangGraph Server API 레퍼런스는 각 배포의 `/docs` 엔드포인트(예: `http://localhost:8124/docs`)에서 사용할 수 있습니다.

API 레퍼런스를 보려면 <a href="/langgraph/cloud/reference/api/api_ref.html" target="_blank">여기</a>를 클릭하세요.

## 인증

LangGraph Platform에 배포하는 경우 인증이 필요합니다. LangGraph Server에 대한 각 요청에 `X-Api-Key` 헤더를 전달합니다. 헤더 값은 LangGraph Server가 배포된 조직의 유효한 LangSmith API 키로 설정해야 합니다.

예제 `curl` 명령:
```shell
curl --request POST \
  --url http://localhost:8124/assistants/search \
  --header 'Content-Type: application/json' \
  --header 'X-Api-Key: LANGSMITH_API_KEY' \
  --data '{
  "metadata": {},
  "limit": 10,
  "offset": 0
}'
```
