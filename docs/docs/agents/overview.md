---
title: 개요
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# 사전 구축된 컴포넌트를 사용한 에이전트 개발

LangGraph는 에이전트 기반 애플리케이션을 구축하기 위한 저수준 프리미티브와 고수준 사전 구축 컴포넌트를 모두 제공합니다. 이 섹션은 처음부터 오케스트레이션, 메모리 또는 인간 피드백 처리를 구현할 필요 없이 에이전틱 시스템을 빠르고 안정적으로 구축할 수 있도록 설계된 사전 구축된 즉시 사용 가능한 컴포넌트에 중점을 둡니다.

## 에이전트란 무엇인가?

_에이전트_는 세 가지 구성 요소로 이루어져 있습니다: **대형 언어 모델(LLM)**, 사용할 수 있는 **도구** 세트, 그리고 지침을 제공하는 **프롬프트**.

LLM은 루프에서 작동합니다. 각 반복에서 호출할 도구를 선택하고, 입력을 제공하며, 결과(관찰)를 받고, 그 관찰을 사용하여 다음 작업을 알립니다. 루프는 중지 조건이 충족될 때까지 계속됩니다 — 일반적으로 에이전트가 사용자에게 응답할 충분한 정보를 수집했을 때입니다.

<figure markdown="1">
![image](./assets/agent.png){: style="max-height:400px"}
<figcaption>에이전트 루프: LLM이 도구를 선택하고 그 출력을 사용하여 사용자 요청을 충족합니다.</figcaption>
</figure>

## 주요 기능

LangGraph는 강력한 프로덕션 준비 에이전틱 시스템을 구축하는 데 필수적인 여러 기능을 포함합니다:

- [**메모리 통합**](../how-tos/memory/add-memory.md): _단기_(세션 기반) 및 _장기_(세션 간 지속) 메모리에 대한 네이티브 지원으로 챗봇과 어시스턴트에서 상태 저장 동작을 가능하게 합니다.
- [**Human-in-the-loop 제어**](../concepts/human_in_the_loop.md): 실시간 상호작용으로 제한되는 웹소켓 기반 솔루션과 달리 인간 피드백을 기다리기 위해 실행을 _무한정_ 일시 중지할 수 있습니다. 이를 통해 워크플로우의 어느 시점에서든 비동기 승인, 수정 또는 개입이 가능합니다.
- [**스트리밍 지원**](../how-tos/streaming.md): 에이전트 상태, 모델 토큰, 도구 출력 또는 결합된 스트림의 실시간 스트리밍.
- [**배포 도구**](../tutorials/langgraph-platform/local-server.md): 인프라 없는 배포 도구를 포함합니다. [**LangGraph Platform**](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/)은 테스트, 디버깅 및 배포를 지원합니다.
  - **[Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)**: 워크플로우를 검사하고 디버그하기 위한 시각적 IDE.
  - 프로덕션을 위한 여러 [**배포 옵션**](https://langchain-ai.github.io/langgraph/concepts/deployment_options.md)을 지원합니다.

## 고수준 빌딩 블록

LangGraph는 일반적인 에이전트 동작과 워크플로우를 구현하는 사전 구축된 컴포넌트 세트를 제공합니다. 이러한 추상화는 LangGraph 프레임워크 위에 구축되어 고급 커스터마이징을 위한 유연성을 유지하면서도 프로덕션으로의 더 빠른 경로를 제공합니다.

LangGraph를 에이전트 개발에 사용하면 상태, 메모리 및 인간 피드백을 위한 지원 인프라를 구축하고 유지하는 대신 애플리케이션의 로직과 동작에 집중할 수 있습니다.

:::python

## 패키지 생태계

고수준 컴포넌트는 각각 특정 초점을 가진 여러 패키지로 구성되어 있습니다.

| 패키지                                      | 설명                                                                                | 설치                                    |
| ------------------------------------------ | ----------------------------------------------------------------------------------- | --------------------------------------- |
| `langgraph-prebuilt` (`langgraph`의 일부) | [**에이전트 생성**](./agents.md)을 위한 사전 구축 컴포넌트                               | `pip install -U langgraph langchain`    |
| `langgraph-supervisor`                     | [**supervisor**](./multi-agent.md#supervisor) 에이전트 구축 도구                      | `pip install -U langgraph-supervisor`   |
| `langgraph-swarm`                          | [**swarm**](./multi-agent.md#swarm) 다중 에이전트 시스템 구축 도구                    | `pip install -U langgraph-swarm`        |
| `langchain-mcp-adapters`                   | 도구 및 리소스 통합을 위한 [**MCP 서버**](./mcp.md) 인터페이스                        | `pip install -U langchain-mcp-adapters` |
| `langmem`                                  | 에이전트 메모리 관리: [**단기 및 장기**](../how-tos/memory/add-memory.md)            | `pip install -U langmem`                |
| `agentevals`                               | [**에이전트 성능 평가**](./evals.md)를 위한 유틸리티                                   | `pip install -U agentevals`             |

## 에이전트 그래프 시각화

@[`create_react_agent`][create_react_agent]에 의해 생성된 그래프를 시각화하고 해당 코드의 개요를 보려면 다음 도구를 사용하세요.
다음의 존재 유무에 따라 정의되는 에이전트의 인프라를 탐색할 수 있습니다:

- [`tools`](../how-tos/tool-calling.md): 에이전트가 작업을 수행하는 데 사용할 수 있는 도구(함수, API 또는 기타 호출 가능한 객체) 목록.
- [`pre_model_hook`](../how-tos/create-react-agent-manage-message-history.ipynb): 모델이 호출되기 전에 호출되는 함수. 메시지를 압축하거나 다른 전처리 작업을 수행하는 데 사용할 수 있습니다.
- `post_model_hook`: 모델이 호출된 후 호출되는 함수. 가드레일, human-in-the-loop 플로우 또는 다른 후처리 작업을 구현하는 데 사용할 수 있습니다.
- [`response_format`](../agents/agents.md#6-configure-structured-output): 최종 출력 타입을 제한하는 데 사용되는 데이터 구조, 예: `pydantic` `BaseModel`.

<div class="agent-layout">
  <div class="agent-graph-features-container">
    <div class="agent-graph-features">
      <h3 class="agent-section-title">Features</h3>
      <label><input type="checkbox" id="tools" checked> <code>tools</code></label>
      <label><input type="checkbox" id="pre_model_hook"> <code>pre_model_hook</code></label>
      <label><input type="checkbox" id="post_model_hook"> <code>post_model_hook</code></label>
      <label><input type="checkbox" id="response_format"> <code>response_format</code></label>
    </div>
  </div>

  <div class="agent-graph-container">
    <h3 class="agent-section-title">Graph</h3>
    <img id="agent-graph-img" src="../assets/react_agent_graphs/0001.svg" alt="graph image" style="max-width: 100%;"/>
  </div>
</div>

다음 코드 스니펫은 @[`create_react_agent`][create_react_agent]로 위의 에이전트(및 기본 그래프)를 생성하는 방법을 보여줍니다:

<div class="language-python">
  <pre><code id="agent-code" class="language-python"></code></pre>
</div>

<script>
function getCheckedValue(id) {
  return document.getElementById(id).checked ? "1" : "0";
}

function getKey() {
  return [
    getCheckedValue("response_format"),
    getCheckedValue("post_model_hook"),
    getCheckedValue("pre_model_hook"),
    getCheckedValue("tools")
  ].join("");
}

function generateCodeSnippet({ tools, pre, post, response }) {
  const lines = [
    "from langgraph.prebuilt import create_react_agent",
    "from langchain_openai import ChatOpenAI"
  ];

  if (response) lines.push("from pydantic import BaseModel");

  lines.push("", 'model = ChatOpenAI("o4-mini")', "");

  if (tools) {
    lines.push(
      "def tool() -> None:",
      '    """Testing tool."""',
      "    ...",
      ""
    );
  }

  if (pre) {
    lines.push(
      "def pre_model_hook() -> None:",
      '    """Pre-model hook."""',
      "    ...",
      ""
    );
  }

  if (post) {
    lines.push(
      "def post_model_hook() -> None:",
      '    """Post-model hook."""',
      "    ...",
      ""
    );
  }

  if (response) {
    lines.push(
      "class ResponseFormat(BaseModel):",
      '    """Response format for the agent."""',
      "    result: str",
      ""
    );
  }

  lines.push("agent = create_react_agent(");
  lines.push("    model,");

  if (tools) lines.push("    tools=[tool],");
  if (pre) lines.push("    pre_model_hook=pre_model_hook,");
  if (post) lines.push("    post_model_hook=post_model_hook,");
  if (response) lines.push("    response_format=ResponseFormat,");

  lines.push(")", "", "# Visualize the graph", "# For Jupyter or GUI environments:", "agent.get_graph().draw_mermaid_png()", "", "# To save PNG to file:", "png_data = agent.get_graph().draw_mermaid_png()", "with open(\"graph.png\", \"wb\") as f:", "    f.write(png_data)", "", "# For terminal/ASCII output:", "agent.get_graph().draw_ascii()");

  return lines.join("\n");
}

async function render() {
  const key = getKey();
  document.getElementById("agent-graph-img").src = `../assets/react_agent_graphs/${key}.svg`;

  const state = {
    tools: document.getElementById("tools").checked,
    pre: document.getElementById("pre_model_hook").checked,
    post: document.getElementById("post_model_hook").checked,
    response: document.getElementById("response_format").checked
  };

  document.getElementById("agent-code").textContent = generateCodeSnippet(state);
}

function initializeWidget() {
  render(); // no need for `await` here
  document.querySelectorAll(".agent-graph-features input").forEach((input) => {
    input.addEventListener("change", render);
  });
}

// Init for both full reload and SPA nav (used by MkDocs Material)
window.addEventListener("DOMContentLoaded", initializeWidget);
document$.subscribe(initializeWidget);
</script>

:::

:::js

## 패키지 생태계

고수준 컴포넌트는 각각 특정 초점을 가진 여러 패키지로 구성되어 있습니다.

| 패키지                   | 설명                                                                       | 설치                                               |
| ------------------------ | -------------------------------------------------------------------------- | -------------------------------------------------- |
| `langgraph`              | [**에이전트 생성**](./agents.md)을 위한 사전 구축 컴포넌트                    | `npm install @langchain/langgraph @langchain/core` |
| `langgraph-supervisor`   | [**supervisor**](./multi-agent.md#supervisor) 에이전트 구축 도구            | `npm install @langchain/langgraph-supervisor`      |
| `langgraph-swarm`        | [**swarm**](./multi-agent.md#swarm) 다중 에이전트 시스템 구축 도구          | `npm install @langchain/langgraph-swarm`           |
| `langchain-mcp-adapters` | 도구 및 리소스 통합을 위한 [**MCP 서버**](./mcp.md) 인터페이스              | `npm install @langchain/mcp-adapters`              |
| `agentevals`             | [**에이전트 성능 평가**](./evals.md)를 위한 유틸리티                          | `npm install agentevals`                           |

## 에이전트 그래프 시각화

@[`createReactAgent`][create_react_agent]에 의해 생성된 그래프를 시각화하고 해당 코드의 개요를 보려면 다음 도구를 사용하세요. 다음의 존재 유무에 따라 정의되는 에이전트의 인프라를 탐색할 수 있습니다:

- [`tools`](./tools.md): 에이전트가 작업을 수행하는 데 사용할 수 있는 도구(함수, API 또는 기타 호출 가능한 객체) 목록.
- `preModelHook`: 모델이 호출되기 전에 호출되는 함수. 메시지를 압축하거나 다른 전처리 작업을 수행하는 데 사용할 수 있습니다.
- `postModelHook`: 모델이 호출된 후 호출되는 함수. 가드레일, human-in-the-loop 플로우 또는 다른 후처리 작업을 구현하는 데 사용할 수 있습니다.
- [`responseFormat`](./agents.md#6-configure-structured-output): 최종 출력 타입을 제한하는 데 사용되는 데이터 구조(Zod 스키마를 통해).

<div class="agent-layout">
  <div class="agent-graph-features-container">
    <div class="agent-graph-features">
      <h3 class="agent-section-title">Features</h3>
      <label><input type="checkbox" id="tools" checked> <code>tools</code></label>
      <label><input type="checkbox" id="preModelHook"> <code>preModelHook</code></label>
      <label><input type="checkbox" id="postModelHook"> <code>postModelHook</code></label>
      <label><input type="checkbox" id="responseFormat"> <code>responseFormat</code></label>
    </div>
  </div>

  <div class="agent-graph-container">
    <h3 class="agent-section-title">Graph</h3>
    <img id="agent-graph-img" src="../assets/react_agent_graphs/0001.svg" alt="graph image" style="max-width: 100%;"/>
  </div>
</div>

다음 코드 스니펫은 @[`createReactAgent`][create_react_agent]로 위의 에이전트(및 기본 그래프)를 생성하는 방법을 보여줍니다:

<div class="language-typescript">
  <pre><code id="agent-code" class="language-typescript"></code></pre>
</div>

<script>
function getCheckedValue(id) {
  return document.getElementById(id).checked ? "1" : "0";
}

function getKey() {
  return [
    getCheckedValue("responseFormat"),
    getCheckedValue("postModelHook"),
    getCheckedValue("preModelHook"),
    getCheckedValue("tools")
  ].join("");
}

function dedent(strings, ...values) {
  const str = String.raw({ raw: strings }, ...values)
  const [space] = str.split("\n").filter(Boolean).at(0).match(/^(\s*)/)
  const spaceLen = space.length
  return str.split("\n").map(line => line.slice(spaceLen)).join("\n").trim()
}

Object.assign(dedent, {
  offset: (size) => (strings, ...values) => {
    return dedent(strings, ...values).split("\n").map(line => " ".repeat(size) + line).join("\n")
  }
})




function generateCodeSnippet({ tools, pre, post, response }) {
  const lines = []

  lines.push(dedent`
    import { createReactAgent } from "@langchain/langgraph/prebuilt";
    import { ChatOpenAI } from "@langchain/openai";
  `)

  if (tools) lines.push(`import { tool } from "@langchain/core/tools";`);
  if (response || tools) lines.push(`import { z } from "zod";`);

  lines.push("", dedent`
    const agent = createReactAgent({
      llm: new ChatOpenAI({ model: "o4-mini" }),
  `)

  if (tools) {
    lines.push(dedent.offset(2)`
      tools: [
        tool(() => "Sample tool output", {
          name: "sampleTool",
          schema: z.object({}),
        }),
      ],
    `)
  }

  if (pre) {
    lines.push(dedent.offset(2)`
      preModelHook: (state) => ({ llmInputMessages: state.messages }),
    `)
  }

  if (post) {
    lines.push(dedent.offset(2)`
      postModelHook: (state) => state,
    `)
  }

  if (response) {
    lines.push(dedent.offset(2)`
      responseFormat: z.object({ result: z.string() }),
    `)
  }

  lines.push(`});`);

  return lines.join("\n");
}

function render() {
  const key = getKey();
  document.getElementById("agent-graph-img").src = `../assets/react_agent_graphs/${key}.svg`;

  const state = {
    tools: document.getElementById("tools").checked,
    pre: document.getElementById("preModelHook").checked,
    post: document.getElementById("postModelHook").checked,
    response: document.getElementById("responseFormat").checked
  };

  document.getElementById("agent-code").textContent = generateCodeSnippet(state);
}

function initializeWidget() {
  render(); // no need for `await` here
  document.querySelectorAll(".agent-graph-features input").forEach((input) => {
    input.addEventListener("change", render);
  });
}

// Init for both full reload and SPA nav (used by MkDocs Material)
window.addEventListener("DOMContentLoaded", initializeWidget);
document$.subscribe(initializeWidget);
</script>

:::
