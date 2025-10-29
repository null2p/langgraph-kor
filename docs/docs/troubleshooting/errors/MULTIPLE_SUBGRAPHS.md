# MULTIPLE_SUBGRAPHS

각 서브그래프에 대해 체크포인팅이 활성화된 상태에서 단일 LangGraph 노드 내에서 서브그래프를 여러 번 호출하고 있습니다.

이는 서브그래프의 체크포인트 네임스페이싱이 작동하는 방식에 대한 내부 제한으로 인해 현재 허용되지 않습니다.

## 문제 해결

다음은 이 오류를 해결하는 데 도움이 될 수 있습니다:

:::python

- 서브그래프에서 중단/재개가 필요하지 않은 경우 다음과 같이 컴파일할 때 `checkpointer=False`를 전달합니다: `.compile(checkpointer=False)`
  :::

:::js

- 서브그래프에서 중단/재개가 필요하지 않은 경우 다음과 같이 컴파일할 때 `checkpointer: false`를 전달합니다: `.compile({ checkpointer: false })`
  :::

- 동일한 노드에서 그래프를 명령적으로 여러 번 호출하지 말고 [`Send`](https://langchain-ai.github.io/langgraph/concepts/low_level/#send) API를 사용합니다.
