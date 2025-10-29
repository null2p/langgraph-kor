# LangGraph Studio 문제 해결

## :fontawesome-brands-safari:{ .safari } Safari 연결 문제

Safari는 localhost에서 평문 HTTP 트래픽을 차단합니다. `langgraph dev`로 Studio를 실행할 때 "Failed to load assistants" 오류가 표시될 수 있습니다.

### 해결 방법 1: Cloudflare 터널 사용

:::python

```shell
pip install -U langgraph-cli>=0.2.6
langgraph dev --tunnel
```

:::

:::js

```shell
npx @langchain/langgraph-cli dev
```

:::

명령은 다음 형식의 URL을 출력합니다:

```shell
https://smith.langchain.com/studio/?baseUrl=https://hamilton-praise-heart-costumes.trycloudflare.com
```

Safari에서 이 URL을 사용하여 Studio를 로드하세요. 여기서 `baseUrl` 파라미터는 에이전트 서버 엔드포인트를 지정합니다.

### 해결 방법 2: Chromium 브라우저 사용

Chrome 및 기타 Chromium 브라우저는 localhost에서 HTTP를 허용합니다. 추가 구성 없이 `langgraph dev`를 사용하세요.

## :fontawesome-brands-brave:{ .brave } Brave 연결 문제

Brave는 Brave Shields가 활성화되어 있을 때 localhost에서 평문 HTTP 트래픽을 차단합니다. `langgraph dev`로 Studio를 실행할 때 "Failed to load assistants" 오류가 표시될 수 있습니다.

### 해결 방법 1: Brave Shields 비활성화

URL 표시줄의 Brave 아이콘을 사용하여 LangSmith에 대한 Brave Shields를 비활성화하세요.

![Brave Shields](./img/brave-shields.png)

### 해결 방법 2: Cloudflare 터널 사용

:::python

```shell
pip install -U langgraph-cli>=0.2.6
langgraph dev --tunnel
```

:::

:::js

```shell
npx @langchain/langgraph-cli dev
```

:::

명령은 다음 형식의 URL을 출력합니다:

```shell
https://smith.langchain.com/studio/?baseUrl=https://hamilton-praise-heart-costumes.trycloudflare.com
```

Brave에서 이 URL을 사용하여 Studio를 로드하세요. 여기서 `baseUrl` 파라미터는 에이전트 서버 엔드포인트를 지정합니다.

## 그래프 엣지 문제

:::python
정의되지 않은 조건부 엣지는 그래프에서 예상치 못한 연결을 표시할 수 있습니다. 이는 적절한 정의가 없으면 LangGraph Studio가 조건부 엣지가 다른 모든 노드에 액세스할 수 있다고 가정하기 때문입니다. 이를 해결하려면 다음 방법 중 하나를 사용하여 라우팅 경로를 명시적으로 정의하세요:

### 해결 방법 1: 경로 맵

라우터 출력과 대상 노드 간의 매핑을 정의하세요:

```python
graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
```

### 해결 방법 2: 라우터 타입 정의 (Python)

Python의 `Literal` 타입을 사용하여 가능한 라우팅 목적지를 지정하세요:

```python
def routing_function(state: GraphState) -> Literal["node_b","node_c"]:
    if state['some_condition'] == True:
        return "node_b"
    else:
        return "node_c"
```

:::

:::js
정의되지 않은 조건부 엣지는 그래프에서 예상치 못한 연결을 표시할 수 있습니다. 이는 적절한 정의가 없으면 LangGraph Studio가 조건부 엣지가 다른 모든 노드에 액세스할 수 있다고 가정하기 때문입니다.
이를 해결하려면 라우터 출력과 대상 노드 간의 매핑을 명시적으로 정의하세요:

```typescript
graph.addConditionalEdges("node_a", routingFunction, {
  true: "node_b",
  false: "node_c",
});
```

:::
