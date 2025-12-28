# LangGraph.js 애플리케이션 설정 방법

[LangGraph.js](https://langchain-ai.github.io/langgraphjs/) 애플리케이션을 LangGraph Platform에 배포하거나 셀프 호스팅하려면 [LangGraph 구성 파일](../reference/cli.md#configuration-file)로 구성해야 합니다. 이 가이드는 `package.json`을 사용하여 프로젝트 의존성을 지정하고 배포를 위한 LangGraph.js 애플리케이션을 설정하는 기본 단계를 다룹니다.

이 워크스루는 [이 리포지토리](https://github.com/langchain-ai/langgraphjs-studio-starter)를 기반으로 하며, LangGraph 애플리케이션을 배포용으로 설정하는 방법을 자세히 알아보기 위해 사용해볼 수 있습니다.

The final repository structure will look something like this:

```bash
my-app/
├── src # all project code lies within here
│   ├── utils # optional utilities for your graph
│   │   ├── tools.ts # tools for your graph
│   │   ├── nodes.ts # node functions for you graph
│   │   └── state.ts # state definition of your graph
│   └── agent.ts # code for constructing your graph
├── package.json # package dependencies
├── .env # environment variables
└── langgraph.json # configuration file for LangGraph
```

각 단계 후에는 코드를 어떻게 구성할 수 있는지 보여주는 예제 파일 디렉토리가 제공됩니다.

## 의존성 지정

의존성은 `package.json`에 지정할 수 있습니다. 이러한 파일이 생성되지 않은 경우 나중에 [LangGraph 구성 파일](#create-langgraph-api-config)에서 의존성을 지정할 수 있습니다.

Example `package.json` file:

```json
{
  "name": "langgraphjs-studio-starter",
  "packageManager": "yarn@1.22.22",
  "dependencies": {
    "@langchain/community": "^0.2.31",
    "@langchain/core": "^0.2.31",
    "@langchain/langgraph": "^0.2.0",
    "@langchain/openai": "^0.2.8"
  }
}
```

앱을 배포할 때, 의존성은 선택한 패키지 관리자를 사용하여 설치되며, 아래 나열된 호환 가능한 버전 범위를 준수해야 합니다:

```
"@langchain/core": "^0.3.42",
"@langchain/langgraph": "^0.2.57",
"@langchain/langgraph-checkpoint": "~0.0.16",
```

Example file directory:

```bash
my-app/
└── package.json # package dependencies
```

## 환경 변수 지정

환경 변수는 선택적으로 파일(예: `.env`)에 지정할 수 있습니다. 배포를 위한 추가 변수 구성은 [환경 변수 레퍼런스](../reference/env_var.md)를 참조하세요.

Example `.env` file:

```
MY_ENV_VAR_1=foo
MY_ENV_VAR_2=bar
OPENAI_API_KEY=key
TAVILY_API_KEY=key_2
```

Example file directory:

```bash
my-app/
├── package.json
└── .env # environment variables
```

## 그래프 정의

그래프를 구현하세요! 그래프는 단일 파일 또는 여러 파일로 정의할 수 있습니다. LangGraph 애플리케이션에 포함될 각 컴파일된 그래프의 변수 이름을 기록해두세요. 변수 이름은 나중에 [LangGraph 구성 파일](../reference/cli.md#configuration-file)을 생성할 때 사용됩니다.

Here is an example `agent.ts`:

```ts
import type { AIMessage } from "@langchain/core/messages";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI } from "@langchain/openai";

import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";

const tools = [new TavilySearchResults({ maxResults: 3 })];

// Define the function that calls the model
async function callModel(state: typeof MessagesAnnotation.State) {
  /**
   * Call the LLM powering our agent.
   * Feel free to customize the prompt, model, and other logic!
   */
  const model = new ChatOpenAI({
    model: "gpt-4o",
  }).bindTools(tools);

  const response = await model.invoke([
    {
      role: "system",
      content: `You are a helpful assistant. The current date is ${new Date().getTime()}.`,
    },
    ...state.messages,
  ]);

  // MessagesAnnotation supports returning a single message or array of messages
  return { messages: response };
}

// Define the function that determines whether to continue or not
function routeModelOutput(state: typeof MessagesAnnotation.State) {
  const messages = state.messages;
  const lastMessage: AIMessage = messages[messages.length - 1];
  // If the LLM is invoking tools, route there.
  if ((lastMessage?.tool_calls?.length ?? 0) > 0) {
    return "tools";
  }
  // Otherwise end the graph.
  return "__end__";
}

// Define a new graph.
// See https://langchain-ai.github.io/langgraphjs/how-tos/define-state/#getting-started for
// more on defining custom graph states.
const workflow = new StateGraph(MessagesAnnotation)
  // Define the two nodes we will cycle between
  .addNode("callModel", callModel)
  .addNode("tools", new ToolNode(tools))
  // Set the entrypoint as `callModel`
  // This means that this node is the first one called
  .addEdge("__start__", "callModel")
  .addConditionalEdges(
    // First, we define the edges' source node. We use `callModel`.
    // This means these are the edges taken after the `callModel` node is called.
    "callModel",
    // Next, we pass in the function that will determine the sink node(s), which
    // will be called after the source node is called.
    routeModelOutput,
    // List of the possible destinations the conditional edge can route to.
    // Required for conditional edges to properly render the graph in Studio
    ["tools", "__end__"]
  )
  // This means that after `tools` is called, `callModel` node is called next.
  .addEdge("tools", "callModel");

// Finally, we compile it!
// This compiles it into a graph you can invoke and deploy.
export const graph = workflow.compile();
```

Example file directory:

```bash
my-app/
├── src # all project code lies within here
│   ├── utils # optional utilities for your graph
│   │   ├── tools.ts # tools for your graph
│   │   ├── nodes.ts # node functions for you graph
│   │   └── state.ts # state definition of your graph
│   └── agent.ts # code for constructing your graph
├── package.json # package dependencies
├── .env # environment variables
└── langgraph.json # configuration file for LangGraph
```

## LangGraph API 구성 생성 {#create-langgraph-api-config}

`langgraph.json`이라는 [LangGraph 구성 파일](../reference/cli.md#configuration-file)을 생성합니다. 구성 파일의 JSON 객체에서 각 키에 대한 자세한 설명은 [LangGraph 구성 파일 레퍼런스](../reference/cli.md#configuration-file)를 참조하세요.

Example `langgraph.json` file:

```json
{
  "node_version": "20",
  "dockerfile_lines": [],
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent.ts:graph"
  },
  "env": ".env"
}
```

`CompiledGraph`의 변수 이름이 최상위 `graphs` 키의 각 하위 키 값 끝에 나타납니다(즉, `:<variable_name>`).

!!! info "구성 위치"

    LangGraph 구성 파일은 컴파일된 그래프와 관련 의존성을 포함하는 TypeScript 파일과 같은 레벨 또는 더 높은 레벨의 디렉토리에 배치되어야 합니다.

## 다음 단계

프로젝트를 설정하고 GitHub 리포지토리에 배치한 후에는 [앱을 배포](./cloud.md)할 차례입니다.
