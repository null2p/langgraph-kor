---
search:
  boost: 2
---

# 워크플로우와 에이전트

이 가이드는 에이전트 시스템의 일반적인 패턴을 검토합니다. 이러한 시스템을 설명할 때 "워크플로우"와 "에이전트"를 구분하는 것이 유용할 수 있습니다. 이 차이에 대해 생각하는 한 가지 방법은 Anthropic의 `Building Effective Agents` 블로그 게시물에 잘 설명되어 있습니다:

> 워크플로우는 LLM과 도구가 미리 정의된 코드 경로를 통해 오케스트레이션되는 시스템입니다.
> 반면 에이전트는 LLM이 자체 프로세스와 도구 사용을 동적으로 지시하여 작업을 수행하는 방법을 제어하는 시스템입니다.

다음은 이러한 차이를 시각화하는 간단한 방법입니다:

![Agent Workflow](../concepts/img/agent_workflow.png)

에이전트와 워크플로우를 구축할 때 LangGraph는 지속성, 스트리밍, 디버깅 지원 및 배포를 포함한 여러 이점을 제공합니다.

## 설정

:::python
구조화된 출력과 도구 호출을 지원하는 [모든 채팅 모델](https://python.langchain.com/docs/integrations/chat/)을 사용할 수 있습니다. 아래에서는 Anthropic에 대한 패키지 설치, API 키 설정 및 구조화된 출력 / 도구 호출 테스트 프로세스를 보여줍니다.

??? "Install dependencies"

    ```bash
    pip install langchain_core langchain-anthropic langgraph
    ```

Initialize an LLM

```python
import os
import getpass

from langchain_anthropic import ChatAnthropic

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
```

:::

:::js
구조화된 출력과 도구 호출을 지원하는 [모든 채팅 모델](https://js.langchain.com/docs/integrations/chat/)을 사용할 수 있습니다. 아래에서는 Anthropic에 대한 패키지 설치, API 키 설정 및 구조화된 출력 / 도구 호출 테스트 프로세스를 보여줍니다.

??? "Install dependencies"

    ```bash
    npm install @langchain/core @langchain/anthropic @langchain/langgraph
    ```

Initialize an LLM

```typescript
import { ChatAnthropic } from "@langchain/anthropic";

process.env.ANTHROPIC_API_KEY = "YOUR_API_KEY";

const llm = new ChatAnthropic({ model: "claude-3-5-sonnet-latest" });
```

:::

## 구성 요소: 증강된 LLM

LLM에는 워크플로우와 에이전트 구축을 지원하는 증강 기능이 있습니다. 여기에는 `Building Effective Agents`에 대한 Anthropic 블로그의 이 이미지에 표시된 것처럼 구조화된 출력과 도구 호출이 포함됩니다:

![augmented_llm.png](./workflows/img/augmented_llm.png)

:::python

```python
# Schema for structured output
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )


# Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)

# Invoke the augmented LLM
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")

# Define a tool
def multiply(a: int, b: int) -> int:
    return a * b

# Augment the LLM with tools
llm_with_tools = llm.bind_tools([multiply])

# Invoke the LLM with input that triggers the tool call
msg = llm_with_tools.invoke("What is 2 times 3?")

# Get the tool call
msg.tool_calls
```

:::

:::js

```typescript
import { z } from "zod";
import { tool } from "@langchain/core/tools";

// Schema for structured output
const SearchQuery = z.object({
  search_query: z.string().describe("Query that is optimized web search."),
  justification: z
    .string()
    .describe("Why this query is relevant to the user's request."),
});

// Augment the LLM with schema for structured output
const structuredLlm = llm.withStructuredOutput(SearchQuery);

// Invoke the augmented LLM
const output = await structuredLlm.invoke(
  "How does Calcium CT score relate to high cholesterol?"
);

// Define a tool
const multiply = tool(
  async ({ a, b }: { a: number; b: number }) => {
    return a * b;
  },
  {
    name: "multiply",
    description: "Multiply two numbers",
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
  }
);

// Augment the LLM with tools
const llmWithTools = llm.bindTools([multiply]);

// Invoke the LLM with input that triggers the tool call
const msg = await llmWithTools.invoke("What is 2 times 3?");

// Get the tool call
console.log(msg.tool_calls);
```

:::

## 프롬프트 체이닝

프롬프트 체이닝에서는 각 LLM 호출이 이전 호출의 출력을 처리합니다.

`Building Effective Agents`에 대한 Anthropic 블로그에서 언급한 것처럼:

> 프롬프트 체이닝은 작업을 일련의 단계로 분해하며, 각 LLM 호출은 이전 호출의 출력을 처리합니다. 프로세스가 여전히 정상 궤도에 있는지 확인하기 위해 중간 단계에 프로그래밍 방식 검사(아래 다이어그램의 "gate" 참조)를 추가할 수 있습니다.

> 이 워크플로우를 사용하는 경우: 이 워크플로우는 작업을 고정된 하위 작업으로 쉽고 깔끔하게 분해할 수 있는 상황에 이상적입니다. 주요 목표는 각 LLM 호출을 더 쉬운 작업으로 만들어 대기 시간을 더 높은 정확도로 절충하는 것입니다.

![prompt_chain.png](./workflows/img/prompt_chain.png)

=== "Graph API"

    :::python
    ```python
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END
    from IPython.display import Image, display


    # Graph state
    class State(TypedDict):
        topic: str
        joke: str
        improved_joke: str
        final_joke: str


    # Nodes
    def generate_joke(state: State):
        """First LLM call to generate initial joke"""

        msg = llm.invoke(f"Write a short joke about {state['topic']}")
        return {"joke": msg.content}


    def check_punchline(state: State):
        """Gate function to check if the joke has a punchline"""

        # Simple check - does the joke contain "?" or "!"
        if "?" in state["joke"] or "!" in state["joke"]:
            return "Pass"
        return "Fail"


    def improve_joke(state: State):
        """Second LLM call to improve the joke"""

        msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
        return {"improved_joke": msg.content}


    def polish_joke(state: State):
        """Third LLM call for final polish"""

        msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
        return {"final_joke": msg.content}


    # Build workflow
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("generate_joke", generate_joke)
    workflow.add_node("improve_joke", improve_joke)
    workflow.add_node("polish_joke", polish_joke)

    # Add edges to connect nodes
    workflow.add_edge(START, "generate_joke")
    workflow.add_conditional_edges(
        "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
    )
    workflow.add_edge("improve_joke", "polish_joke")
    workflow.add_edge("polish_joke", END)

    # Compile
    chain = workflow.compile()

    # Show workflow
    display(Image(chain.get_graph().draw_mermaid_png()))

    # Invoke
    state = chain.invoke({"topic": "cats"})
    print("Initial joke:")
    print(state["joke"])
    print("\n--- --- ---\n")
    if "improved_joke" in state:
        print("Improved joke:")
        print(state["improved_joke"])
        print("\n--- --- ---\n")

        print("Final joke:")
        print(state["final_joke"])
    else:
        print("Joke failed quality gate - no punchline detected!")
    ```

    **LangSmith Trace**

    https://smith.langchain.com/public/a0281fca-3a71-46de-beee-791468607b75/r

    **Resources:**

    **LangChain Academy**

    See our lesson on Prompt Chaining [here](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/chain.ipynb).
    :::

    :::js
    ```typescript
    import { StateGraph, START, END } from "@langchain/langgraph";
    import { z } from "zod";

    // Graph state
    const State = z.object({
      topic: z.string(),
      joke: z.string().optional(),
      improved_joke: z.string().optional(),
      final_joke: z.string().optional(),
    });

    // Nodes
    const generateJoke = async (state: z.infer<typeof State>) => {
      // First LLM call to generate initial joke
      const msg = await llm.invoke(`Write a short joke about ${state.topic}`);
      return { joke: msg.content };
    };

    const checkPunchline = (state: z.infer<typeof State>) => {
      // Gate function to check if the joke has a punchline
      // Simple check - does the joke contain "?" or "!"
      if (state.joke && (state.joke.includes("?") || state.joke.includes("!"))) {
        return "Pass";
      }
      return "Fail";
    };

    const improveJoke = async (state: z.infer<typeof State>) => {
      // Second LLM call to improve the joke
      const msg = await llm.invoke(`Make this joke funnier by adding wordplay: ${state.joke}`);
      return { improved_joke: msg.content };
    };

    const polishJoke = async (state: z.infer<typeof State>) => {
      // Third LLM call for final polish
      const msg = await llm.invoke(`Add a surprising twist to this joke: ${state.improved_joke}`);
      return { final_joke: msg.content };
    };

    // Build workflow
    const workflow = new StateGraph(State)
      .addNode("generate_joke", generateJoke)
      .addNode("improve_joke", improveJoke)
      .addNode("polish_joke", polishJoke)
      .addEdge(START, "generate_joke")
      .addConditionalEdges(
        "generate_joke",
        checkPunchline,
        { "Fail": "improve_joke", "Pass": END }
      )
      .addEdge("improve_joke", "polish_joke")
      .addEdge("polish_joke", END);

    // Compile
    const chain = workflow.compile();

    // Show workflow
    import * as fs from "node:fs/promises";
    const drawableGraph = await chain.getGraphAsync();
    const image = await drawableGraph.drawMermaidPng();
    const imageBuffer = new Uint8Array(await image.arrayBuffer());
    await fs.writeFile("workflow.png", imageBuffer);

    // Invoke
    const state = await chain.invoke({ topic: "cats" });
    console.log("Initial joke:");
    console.log(state.joke);
    console.log("\n--- --- ---\n");
    if (state.improved_joke) {
      console.log("Improved joke:");
      console.log(state.improved_joke);
      console.log("\n--- --- ---\n");

      console.log("Final joke:");
      console.log(state.final_joke);
    } else {
      console.log("Joke failed quality gate - no punchline detected!");
    }
    ```
    :::

=== "Functional API"

    :::python
    ```python
    from langgraph.func import entrypoint, task


    # Tasks
    @task
    def generate_joke(topic: str):
        """First LLM call to generate initial joke"""
        msg = llm.invoke(f"Write a short joke about {topic}")
        return msg.content


    def check_punchline(joke: str):
        """Gate function to check if the joke has a punchline"""
        # Simple check - does the joke contain "?" or "!"
        if "?" in joke or "!" in joke:
            return "Fail"

        return "Pass"


    @task
    def improve_joke(joke: str):
        """Second LLM call to improve the joke"""
        msg = llm.invoke(f"Make this joke funnier by adding wordplay: {joke}")
        return msg.content


    @task
    def polish_joke(joke: str):
        """Third LLM call for final polish"""
        msg = llm.invoke(f"Add a surprising twist to this joke: {joke}")
        return msg.content


    @entrypoint()
    def prompt_chaining_workflow(topic: str):
        original_joke = generate_joke(topic).result()
        if check_punchline(original_joke) == "Pass":
            return original_joke

        improved_joke = improve_joke(original_joke).result()
        return polish_joke(improved_joke).result()

    # Invoke
    for step in prompt_chaining_workflow.stream("cats", stream_mode="updates"):
        print(step)
        print("\n")
    ```

    **LangSmith Trace**

    https://smith.langchain.com/public/332fa4fc-b6ca-416e-baa3-161625e69163/r
    :::

    :::js
    ```typescript
    import { entrypoint, task } from "@langchain/langgraph";

    // Tasks
    const generateJoke = task("generate_joke", async (topic: string) => {
      // First LLM call to generate initial joke
      const msg = await llm.invoke(`Write a short joke about ${topic}`);
      return msg.content;
    });

    const checkPunchline = (joke: string) => {
      // Gate function to check if the joke has a punchline
      // Simple check - does the joke contain "?" or "!"
      if (joke.includes("?") || joke.includes("!")) {
        return "Pass";
      }
      return "Fail";
    };

    const improveJoke = task("improve_joke", async (joke: string) => {
      // Second LLM call to improve the joke
      const msg = await llm.invoke(`Make this joke funnier by adding wordplay: ${joke}`);
      return msg.content;
    });

    const polishJoke = task("polish_joke", async (joke: string) => {
      // Third LLM call for final polish
      const msg = await llm.invoke(`Add a surprising twist to this joke: ${joke}`);
      return msg.content;
    });

    const promptChainingWorkflow = entrypoint("promptChainingWorkflow", async (topic: string) => {
      const originalJoke = await generateJoke(topic);
      if (checkPunchline(originalJoke) === "Pass") {
        return originalJoke;
      }

      const improvedJoke = await improveJoke(originalJoke);
      return await polishJoke(improvedJoke);
    });

    // Invoke
    const stream = await promptChainingWorkflow.stream("cats", { streamMode: "updates" });
    for await (const step of stream) {
      console.log(step);
      console.log("\n");
    }
    ```
    :::

## 병렬화

병렬화를 사용하면 LLM이 작업을 동시에 수행합니다:

> LLM은 때때로 작업을 동시에 수행하고 출력을 프로그래밍 방식으로 집계할 수 있습니다. 이 워크플로우인 병렬화는 두 가지 주요 변형으로 나타납니다: 섹셔닝: 작업을 병렬로 실행되는 독립적인 하위 작업으로 나눕니다. 투표: 다양한 출력을 얻기 위해 동일한 작업을 여러 번 실행합니다.

> 이 워크플로우를 사용하는 경우: 병렬화는 분할된 하위 작업을 속도를 위해 병렬화할 수 있을 때 또는 더 높은 신뢰도 결과를 위해 여러 관점이나 시도가 필요할 때 효과적입니다. 여러 고려 사항이 있는 복잡한 작업의 경우 일반적으로 각 고려 사항이 별도의 LLM 호출로 처리될 때 LLM이 더 나은 성능을 발휘하며, 각 특정 측면에 집중된 주의를 기울일 수 있습니다.

![parallelization.png](./workflows/img/parallelization.png)

=== "Graph API"

    :::python
    ```python
    # Graph state
    class State(TypedDict):
        topic: str
        joke: str
        story: str
        poem: str
        combined_output: str


    # Nodes
    def call_llm_1(state: State):
        """First LLM call to generate initial joke"""

        msg = llm.invoke(f"Write a joke about {state['topic']}")
        return {"joke": msg.content}


    def call_llm_2(state: State):
        """Second LLM call to generate story"""

        msg = llm.invoke(f"Write a story about {state['topic']}")
        return {"story": msg.content}


    def call_llm_3(state: State):
        """Third LLM call to generate poem"""

        msg = llm.invoke(f"Write a poem about {state['topic']}")
        return {"poem": msg.content}


    def aggregator(state: State):
        """Combine the joke and story into a single output"""

        combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
        combined += f"STORY:\n{state['story']}\n\n"
        combined += f"JOKE:\n{state['joke']}\n\n"
        combined += f"POEM:\n{state['poem']}"
        return {"combined_output": combined}


    # Build workflow
    parallel_builder = StateGraph(State)

    # Add nodes
    parallel_builder.add_node("call_llm_1", call_llm_1)
    parallel_builder.add_node("call_llm_2", call_llm_2)
    parallel_builder.add_node("call_llm_3", call_llm_3)
    parallel_builder.add_node("aggregator", aggregator)

    # Add edges to connect nodes
    parallel_builder.add_edge(START, "call_llm_1")
    parallel_builder.add_edge(START, "call_llm_2")
    parallel_builder.add_edge(START, "call_llm_3")
    parallel_builder.add_edge("call_llm_1", "aggregator")
    parallel_builder.add_edge("call_llm_2", "aggregator")
    parallel_builder.add_edge("call_llm_3", "aggregator")
    parallel_builder.add_edge("aggregator", END)
    parallel_workflow = parallel_builder.compile()

    # Show workflow
    display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

    # Invoke
    state = parallel_workflow.invoke({"topic": "cats"})
    print(state["combined_output"])
    ```

    **LangSmith Trace**

    https://smith.langchain.com/public/3be2e53c-ca94-40dd-934f-82ff87fac277/r

    **Resources:**

    **Documentation**

    See our documentation on parallelization [here](https://langchain-ai.github.io/langgraph/how-tos/branching/).

    **LangChain Academy**

    See our lesson on parallelization [here](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/simple-graph.ipynb).
    :::

    :::js
    ```typescript
    // Graph state
    const State = z.object({
      topic: z.string(),
      joke: z.string().optional(),
      story: z.string().optional(),
      poem: z.string().optional(),
      combined_output: z.string().optional(),
    });

    // Nodes
    const callLlm1 = async (state: z.infer<typeof State>) => {
      // First LLM call to generate initial joke
      const msg = await llm.invoke(`Write a joke about ${state.topic}`);
      return { joke: msg.content };
    };

    const callLlm2 = async (state: z.infer<typeof State>) => {
      // Second LLM call to generate story
      const msg = await llm.invoke(`Write a story about ${state.topic}`);
      return { story: msg.content };
    };

    const callLlm3 = async (state: z.infer<typeof State>) => {
      // Third LLM call to generate poem
      const msg = await llm.invoke(`Write a poem about ${state.topic}`);
      return { poem: msg.content };
    };

    const aggregator = (state: z.infer<typeof State>) => {
      // Combine the joke and story into a single output
      let combined = `Here's a story, joke, and poem about ${state.topic}!\n\n`;
      combined += `STORY:\n${state.story}\n\n`;
      combined += `JOKE:\n${state.joke}\n\n`;
      combined += `POEM:\n${state.poem}`;
      return { combined_output: combined };
    };

    // Build workflow
    const parallelBuilder = new StateGraph(State)
      .addNode("call_llm_1", callLlm1)
      .addNode("call_llm_2", callLlm2)
      .addNode("call_llm_3", callLlm3)
      .addNode("aggregator", aggregator)
      .addEdge(START, "call_llm_1")
      .addEdge(START, "call_llm_2")
      .addEdge(START, "call_llm_3")
      .addEdge("call_llm_1", "aggregator")
      .addEdge("call_llm_2", "aggregator")
      .addEdge("call_llm_3", "aggregator")
      .addEdge("aggregator", END);

    const parallelWorkflow = parallelBuilder.compile();

    // Invoke
    const state = await parallelWorkflow.invoke({ topic: "cats" });
    console.log(state.combined_output);
    ```
    :::

=== "Functional API"

    :::python
    ```python
    @task
    def call_llm_1(topic: str):
        """First LLM call to generate initial joke"""
        msg = llm.invoke(f"Write a joke about {topic}")
        return msg.content


    @task
    def call_llm_2(topic: str):
        """Second LLM call to generate story"""
        msg = llm.invoke(f"Write a story about {topic}")
        return msg.content


    @task
    def call_llm_3(topic):
        """Third LLM call to generate poem"""
        msg = llm.invoke(f"Write a poem about {topic}")
        return msg.content


    @task
    def aggregator(topic, joke, story, poem):
        """Combine the joke and story into a single output"""

        combined = f"Here's a story, joke, and poem about {topic}!\n\n"
        combined += f"STORY:\n{story}\n\n"
        combined += f"JOKE:\n{joke}\n\n"
        combined += f"POEM:\n{poem}"
        return combined


    # Build workflow
    @entrypoint()
    def parallel_workflow(topic: str):
        joke_fut = call_llm_1(topic)
        story_fut = call_llm_2(topic)
        poem_fut = call_llm_3(topic)
        return aggregator(
            topic, joke_fut.result(), story_fut.result(), poem_fut.result()
        ).result()

    # Invoke
    for step in parallel_workflow.stream("cats", stream_mode="updates"):
        print(step)
        print("\n")
    ```

    **LangSmith Trace**

    https://smith.langchain.com/public/623d033f-e814-41e9-80b1-75e6abb67801/r
    :::

    :::js
    ```typescript
    const callLlm1 = task("call_llm_1", async (topic: string) => {
      // First LLM call to generate initial joke
      const msg = await llm.invoke(`Write a joke about ${topic}`);
      return msg.content;
    });

    const callLlm2 = task("call_llm_2", async (topic: string) => {
      // Second LLM call to generate story
      const msg = await llm.invoke(`Write a story about ${topic}`);
      return msg.content;
    });

    const callLlm3 = task("call_llm_3", async (topic: string) => {
      // Third LLM call to generate poem
      const msg = await llm.invoke(`Write a poem about ${topic}`);
      return msg.content;
    });

    const aggregator = task("aggregator", (topic: string, joke: string, story: string, poem: string) => {
      // Combine the joke and story into a single output
      let combined = `Here's a story, joke, and poem about ${topic}!\n\n`;
      combined += `STORY:\n${story}\n\n`;
      combined += `JOKE:\n${joke}\n\n`;
      combined += `POEM:\n${poem}`;
      return combined;
    });

    // Build workflow
    const parallelWorkflow = entrypoint("parallelWorkflow", async (topic: string) => {
      const jokeFut = callLlm1(topic);
      const storyFut = callLlm2(topic);
      const poemFut = callLlm3(topic);

      return await aggregator(
        topic,
        await jokeFut,
        await storyFut,
        await poemFut
      );
    });

    // Invoke
    const stream = await parallelWorkflow.stream("cats", { streamMode: "updates" });
    for await (const step of stream) {
      console.log(step);
      console.log("\n");
    }
    ```
    :::

## 라우팅

라우팅은 입력을 분류하고 후속 작업으로 안내합니다. `Building Effective Agents`에 대한 Anthropic 블로그에서 언급한 것처럼:

> 라우팅은 입력을 분류하고 전문화된 후속 작업으로 안내합니다. 이 워크플로우를 사용하면 관심사를 분리하고 더 전문화된 프롬프트를 구축할 수 있습니다. 이 워크플로우가 없으면 한 종류의 입력에 최적화하면 다른 입력의 성능이 저하될 수 있습니다.

> 이 워크플로우를 사용하는 경우: 라우팅은 별도로 처리하는 것이 더 나은 뚜렷한 범주가 있고 LLM 또는 더 전통적인 분류 모델/알고리즘에 의해 분류를 정확하게 처리할 수 있는 복잡한 작업에 적합합니다.

![routing.png](./workflows/img/routing.png)

=== "Graph API"

    :::python
    ```python
    from typing_extensions import Literal
    from langchain_core.messages import HumanMessage, SystemMessage


    # Schema for structured output to use as routing logic
    class Route(BaseModel):
        step: Literal["poem", "story", "joke"] = Field(
            None, description="The next step in the routing process"
        )


    # Augment the LLM with schema for structured output
    router = llm.with_structured_output(Route)


    # State
    class State(TypedDict):
        input: str
        decision: str
        output: str


    # Nodes
    def llm_call_1(state: State):
        """Write a story"""

        result = llm.invoke(state["input"])
        return {"output": result.content}


    def llm_call_2(state: State):
        """Write a joke"""

        result = llm.invoke(state["input"])
        return {"output": result.content}


    def llm_call_3(state: State):
        """Write a poem"""

        result = llm.invoke(state["input"])
        return {"output": result.content}


    def llm_call_router(state: State):
        """Route the input to the appropriate node"""

        # Run the augmented LLM with structured output to serve as routing logic
        decision = router.invoke(
            [
                SystemMessage(
                    content="Route the input to story, joke, or poem based on the user's request."
                ),
                HumanMessage(content=state["input"]),
            ]
        )

        return {"decision": decision.step}


    # Conditional edge function to route to the appropriate node
    def route_decision(state: State):
        # Return the node name you want to visit next
        if state["decision"] == "story":
            return "llm_call_1"
        elif state["decision"] == "joke":
            return "llm_call_2"
        elif state["decision"] == "poem":
            return "llm_call_3"


    # Build workflow
    router_builder = StateGraph(State)

    # Add nodes
    router_builder.add_node("llm_call_1", llm_call_1)
    router_builder.add_node("llm_call_2", llm_call_2)
    router_builder.add_node("llm_call_3", llm_call_3)
    router_builder.add_node("llm_call_router", llm_call_router)

    # Add edges to connect nodes
    router_builder.add_edge(START, "llm_call_router")
    router_builder.add_conditional_edges(
        "llm_call_router",
        route_decision,
        {  # Name returned by route_decision : Name of next node to visit
            "llm_call_1": "llm_call_1",
            "llm_call_2": "llm_call_2",
            "llm_call_3": "llm_call_3",
        },
    )
    router_builder.add_edge("llm_call_1", END)
    router_builder.add_edge("llm_call_2", END)
    router_builder.add_edge("llm_call_3", END)

    # Compile workflow
    router_workflow = router_builder.compile()

    # Show the workflow
    display(Image(router_workflow.get_graph().draw_mermaid_png()))

    # Invoke
    state = router_workflow.invoke({"input": "Write me a joke about cats"})
    print(state["output"])
    ```

    **LangSmith Trace**

    https://smith.langchain.com/public/c4580b74-fe91-47e4-96fe-7fac598d509c/r

    **Resources:**

    **LangChain Academy**

    See our lesson on routing [here](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/router.ipynb).

    **Examples**

    [Here](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/) is RAG workflow that routes questions. See our video [here](https://www.youtube.com/watch?v=bq1Plo2RhYI).
    :::

    :::js
    ```typescript
    import { SystemMessage, HumanMessage } from "@langchain/core/messages";

    // Schema for structured output to use as routing logic
    const Route = z.object({
      step: z.enum(["poem", "story", "joke"]).describe("The next step in the routing process"),
    });

    // Augment the LLM with schema for structured output
    const router = llm.withStructuredOutput(Route);

    // State
    const State = z.object({
      input: z.string(),
      decision: z.string().optional(),
      output: z.string().optional(),
    });

    // Nodes
    const llmCall1 = async (state: z.infer<typeof State>) => {
      // Write a story
      const result = await llm.invoke(state.input);
      return { output: result.content };
    };

    const llmCall2 = async (state: z.infer<typeof State>) => {
      // Write a joke
      const result = await llm.invoke(state.input);
      return { output: result.content };
    };

    const llmCall3 = async (state: z.infer<typeof State>) => {
      // Write a poem
      const result = await llm.invoke(state.input);
      return { output: result.content };
    };

    const llmCallRouter = async (state: z.infer<typeof State>) => {
      // Route the input to the appropriate node
      const decision = await router.invoke([
        new SystemMessage("Route the input to story, joke, or poem based on the user's request."),
        new HumanMessage(state.input),
      ]);

      return { decision: decision.step };
    };

    // Conditional edge function to route to the appropriate node
    const routeDecision = (state: z.infer<typeof State>) => {
      // Return the node name you want to visit next
      if (state.decision === "story") {
        return "llm_call_1";
      } else if (state.decision === "joke") {
        return "llm_call_2";
      } else if (state.decision === "poem") {
        return "llm_call_3";
      }
    };

    // Build workflow
    const routerBuilder = new StateGraph(State)
      .addNode("llm_call_1", llmCall1)
      .addNode("llm_call_2", llmCall2)
      .addNode("llm_call_3", llmCall3)
      .addNode("llm_call_router", llmCallRouter)
      .addEdge(START, "llm_call_router")
      .addConditionalEdges(
        "llm_call_router",
        routeDecision,
        {
          "llm_call_1": "llm_call_1",
          "llm_call_2": "llm_call_2",
          "llm_call_3": "llm_call_3",
        }
      )
      .addEdge("llm_call_1", END)
      .addEdge("llm_call_2", END)
      .addEdge("llm_call_3", END);

    const routerWorkflow = routerBuilder.compile();

    // Invoke
    const state = await routerWorkflow.invoke({ input: "Write me a joke about cats" });
    console.log(state.output);
    ```
    :::

=== "Functional API"

    :::python
    ```python
    from typing_extensions import Literal
    from pydantic import BaseModel
    from langchain_core.messages import HumanMessage, SystemMessage


    # Schema for structured output to use as routing logic
    class Route(BaseModel):
        step: Literal["poem", "story", "joke"] = Field(
            None, description="The next step in the routing process"
        )


    # Augment the LLM with schema for structured output
    router = llm.with_structured_output(Route)


    @task
    def llm_call_1(input_: str):
        """Write a story"""
        result = llm.invoke(input_)
        return result.content


    @task
    def llm_call_2(input_: str):
        """Write a joke"""
        result = llm.invoke(input_)
        return result.content


    @task
    def llm_call_3(input_: str):
        """Write a poem"""
        result = llm.invoke(input_)
        return result.content


    def llm_call_router(input_: str):
        """Route the input to the appropriate node"""
        # Run the augmented LLM with structured output to serve as routing logic
        decision = router.invoke(
            [
                SystemMessage(
                    content="Route the input to story, joke, or poem based on the user's request."
                ),
                HumanMessage(content=input_),
            ]
        )
        return decision.step


    # Create workflow
    @entrypoint()
    def router_workflow(input_: str):
        next_step = llm_call_router(input_)
        if next_step == "story":
            llm_call = llm_call_1
        elif next_step == "joke":
            llm_call = llm_call_2
        elif next_step == "poem":
            llm_call = llm_call_3

        return llm_call(input_).result()

    # Invoke
    for step in router_workflow.stream("Write me a joke about cats", stream_mode="updates"):
        print(step)
        print("\n")
    ```

    **LangSmith Trace**

    https://smith.langchain.com/public/5e2eb979-82dd-402c-b1a0-a8cceaf2a28a/r
    :::

    :::js
    ```typescript
    import { SystemMessage, HumanMessage } from "@langchain/core/messages";

    // Schema for structured output to use as routing logic
    const Route = z.object({
      step: z.enum(["poem", "story", "joke"]).describe(
        "The next step in the routing process"
      ),
    });

    // Augment the LLM with schema for structured output
    const router = llm.withStructuredOutput(Route);

    const llmCall1 = task("llm_call_1", async (input: string) => {
      // Write a story
      const result = await llm.invoke(input);
      return result.content;
    });

    const llmCall2 = task("llm_call_2", async (input: string) => {
      // Write a joke
      const result = await llm.invoke(input);
      return result.content;
    });

    const llmCall3 = task("llm_call_3", async (input: string) => {
      // Write a poem
      const result = await llm.invoke(input);
      return result.content;
    });

    const llmCallRouter = async (input: string) => {
      // Route the input to the appropriate node
      const decision = await router.invoke([
        new SystemMessage("Route the input to story, joke, or poem based on the user's request."),
        new HumanMessage(input),
      ]);
      return decision.step;
    };

    // Create workflow
    const routerWorkflow = entrypoint("routerWorkflow", async (input: string) => {
      const nextStep = await llmCallRouter(input);

      let llmCall: typeof llmCall1;
      if (nextStep === "story") {
        llmCall = llmCall1;
      } else if (nextStep === "joke") {
        llmCall = llmCall2;
      } else if (nextStep === "poem") {
        llmCall = llmCall3;
      }

      return await llmCall(input);
    });

    // Invoke
    const stream = await routerWorkflow.stream("Write me a joke about cats", { streamMode: "updates" });
    for await (const step of stream) {
      console.log(step);
      console.log("\n");
    }
    ```
    :::

## 오케스트레이터-워커

오케스트레이터-워커에서는 오케스트레이터가 작업을 분해하고 각 하위 작업을 워커에게 위임합니다. `Building Effective Agents`에 대한 Anthropic 블로그에서 언급한 것처럼:

> 오케스트레이터-워커 워크플로우에서는 중앙 LLM이 작업을 동적으로 분해하고 워커 LLM에 위임한 다음 결과를 종합합니다.

> 이 워크플로우를 사용하는 경우: 이 워크플로우는 필요한 하위 작업을 예측할 수 없는 복잡한 작업에 적합합니다(예를 들어 코딩에서 변경해야 하는 파일의 수와 각 파일의 변경 특성은 작업에 따라 달라질 수 있습니다). 토폴로지상 유사하지만 병렬화와의 주요 차이점은 유연성입니다 — 하위 작업이 미리 정의되지 않고 특정 입력을 기반으로 오케스트레이터가 결정합니다.

![worker.png](./workflows/img/worker.png)

=== "Graph API"

    :::python
    ```python
    from typing import Annotated, List
    import operator


    # Schema for structured output to use in planning
    class Section(BaseModel):
        name: str = Field(
            description="Name for this section of the report.",
        )
        description: str = Field(
            description="Brief overview of the main topics and concepts to be covered in this section.",
        )


    class Sections(BaseModel):
        sections: List[Section] = Field(
            description="Sections of the report.",
        )


    # Augment the LLM with schema for structured output
    planner = llm.with_structured_output(Sections)
    ```

    **LangGraph에서 워커 생성**

    오케스트레이터-워커 워크플로우가 일반적이기 때문에 LangGraph는 **이를 지원하는 `Send` API를 제공합니다**. 이를 통해 워커 노드를 동적으로 생성하고 각 노드에 특정 입력을 보낼 수 있습니다. 각 워커에는 자체 상태가 있으며 모든 워커 출력은 오케스트레이터 그래프가 액세스할 수 있는 *공유 상태 키*에 기록됩니다. 이를 통해 오케스트레이터는 모든 워커 출력에 액세스하고 이를 최종 출력으로 합성할 수 있습니다. 아래에서 볼 수 있듯이 섹션 목록을 반복하고 각각을 워커 노드에 `Send`합니다. 추가 문서는 [여기](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)와 [여기](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)를 참조하세요.

    ```python
    from langgraph.types import Send


    # Graph state
    class State(TypedDict):
        topic: str  # Report topic
        sections: list[Section]  # List of report sections
        completed_sections: Annotated[
            list, operator.add
        ]  # All workers write to this key in parallel
        final_report: str  # Final report


    # Worker state
    class WorkerState(TypedDict):
        section: Section
        completed_sections: Annotated[list, operator.add]


    # Nodes
    def orchestrator(state: State):
        """Orchestrator that generates a plan for the report"""

        # Generate queries
        report_sections = planner.invoke(
            [
                SystemMessage(content="Generate a plan for the report."),
                HumanMessage(content=f"Here is the report topic: {state['topic']}"),
            ]
        )

        return {"sections": report_sections.sections}


    def llm_call(state: WorkerState):
        """Worker writes a section of the report"""

        # Generate section
        section = llm.invoke(
            [
                SystemMessage(
                    content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
                ),
                HumanMessage(
                    content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
                ),
            ]
        )

        # Write the updated section to completed sections
        return {"completed_sections": [section.content]}


    def synthesizer(state: State):
        """Synthesize full report from sections"""

        # List of completed sections
        completed_sections = state["completed_sections"]

        # Format completed section to str to use as context for final sections
        completed_report_sections = "\n\n---\n\n".join(completed_sections)

        return {"final_report": completed_report_sections}


    # Conditional edge function to create llm_call workers that each write a section of the report
    def assign_workers(state: State):
        """Assign a worker to each section in the plan"""

        # Kick off section writing in parallel via Send() API
        return [Send("llm_call", {"section": s}) for s in state["sections"]]


    # Build workflow
    orchestrator_worker_builder = StateGraph(State)

    # Add the nodes
    orchestrator_worker_builder.add_node("orchestrator", orchestrator)
    orchestrator_worker_builder.add_node("llm_call", llm_call)
    orchestrator_worker_builder.add_node("synthesizer", synthesizer)

    # Add edges to connect nodes
    orchestrator_worker_builder.add_edge(START, "orchestrator")
    orchestrator_worker_builder.add_conditional_edges(
        "orchestrator", assign_workers, ["llm_call"]
    )
    orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
    orchestrator_worker_builder.add_edge("synthesizer", END)

    # Compile the workflow
    orchestrator_worker = orchestrator_worker_builder.compile()

    # Show the workflow
    display(Image(orchestrator_worker.get_graph().draw_mermaid_png()))

    # Invoke
    state = orchestrator_worker.invoke({"topic": "Create a report on LLM scaling laws"})

    from IPython.display import Markdown
    Markdown(state["final_report"])
    ```

    **LangSmith Trace**

    https://smith.langchain.com/public/78cbcfc3-38bf-471d-b62a-b299b144237d/r

    **Resources:**

    **LangChain Academy**

    See our lesson on orchestrator-worker [here](https://github.com/langchain-ai/langchain-academy/blob/main/module-4/map-reduce.ipynb).

    **Examples**

    [Here](https://github.com/langchain-ai/report-mAIstro) is a project that uses orchestrator-worker for report planning and writing. See our video [here](https://www.youtube.com/watch?v=wSxZ7yFbbas).
    :::

    :::js
    ```typescript
    import "@langchain/langgraph/zod";

    // Schema for structured output to use in planning
    const Section = z.object({
      name: z.string().describe("Name for this section of the report."),
      description: z.string().describe("Brief overview of the main topics and concepts to be covered in this section."),
    });

    const Sections = z.object({
      sections: z.array(Section).describe("Sections of the report."),
    });

    // Augment the LLM with schema for structured output
    const planner = llm.withStructuredOutput(Sections);
    ```

    **LangGraph에서 워커 생성**

    오케스트레이터-워커 워크플로우가 일반적이기 때문에 LangGraph는 **이를 지원하는 `Send` API를 제공합니다**. 이를 통해 워커 노드를 동적으로 생성하고 각 노드에 특정 입력을 보낼 수 있습니다. 각 워커에는 자체 상태가 있으며 모든 워커 출력은 오케스트레이터 그래프가 액세스할 수 있는 *공유 상태 키*에 기록됩니다. 이를 통해 오케스트레이터는 모든 워커 출력에 액세스하고 이를 최종 출력으로 합성할 수 있습니다. 아래에서 볼 수 있듯이 섹션 목록을 반복하고 각각을 워커 노드에 `Send`합니다. 추가 문서는 [여기](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)와 [여기](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)를 참조하세요.

    ```typescript
    import { withLangGraph } from "@langchain/langgraph/zod";
    import { Send } from "@langchain/langgraph";

    // Graph state
    const State = z.object({
      topic: z.string(), // Report topic
      sections: z.array(Section).optional(), // List of report sections
      // All workers write to this key
      completed_sections: withLangGraph(z.array(z.string()), {
        reducer: {
          fn: (x, y) => x.concat(y),
        },
        default: () => [],
      }),
      final_report: z.string().optional(), // Final report
    });

    // Worker state
    const WorkerState = z.object({
      section: Section,
      completed_sections: withLangGraph(z.array(z.string()), {
        reducer: {
          fn: (x, y) => x.concat(y),
        },
        default: () => [],
      }),
    });

    // Nodes
    const orchestrator = async (state: z.infer<typeof State>) => {
      // Orchestrator that generates a plan for the report
      const reportSections = await planner.invoke([
        new SystemMessage("Generate a plan for the report."),
        new HumanMessage(`Here is the report topic: ${state.topic}`),
      ]);

      return { sections: reportSections.sections };
    };

    const llmCall = async (state: z.infer<typeof WorkerState>) => {
      // Worker writes a section of the report
      const section = await llm.invoke([
        new SystemMessage(
          "Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
        ),
        new HumanMessage(
          `Here is the section name: ${state.section.name} and description: ${state.section.description}`
        ),
      ]);

      // Write the updated section to completed sections
      return { completed_sections: [section.content] };
    };

    const synthesizer = (state: z.infer<typeof State>) => {
      // Synthesize full report from sections
      const completedSections = state.completed_sections;
      const completedReportSections = completedSections.join("\n\n---\n\n");
      return { final_report: completedReportSections };
    };

    // Conditional edge function to create llm_call workers
    const assignWorkers = (state: z.infer<typeof State>) => {
      // Assign a worker to each section in the plan
      return state.sections!.map((s) => new Send("llm_call", { section: s }));
    };

    // Build workflow
    const orchestratorWorkerBuilder = new StateGraph(State)
      .addNode("orchestrator", orchestrator)
      .addNode("llm_call", llmCall)
      .addNode("synthesizer", synthesizer)
      .addEdge(START, "orchestrator")
      .addConditionalEdges("orchestrator", assignWorkers, ["llm_call"])
      .addEdge("llm_call", "synthesizer")
      .addEdge("synthesizer", END);

    // Compile the workflow
    const orchestratorWorker = orchestratorWorkerBuilder.compile();

    // Invoke
    const state = await orchestratorWorker.invoke({ topic: "Create a report on LLM scaling laws" });
    console.log(state.final_report);
    ```
    :::

=== "Functional API"

    :::python
    ```python
    from typing import List


    # Schema for structured output to use in planning
    class Section(BaseModel):
        name: str = Field(
            description="Name for this section of the report.",
        )
        description: str = Field(
            description="Brief overview of the main topics and concepts to be covered in this section.",
        )


    class Sections(BaseModel):
        sections: List[Section] = Field(
            description="Sections of the report.",
        )


    # Augment the LLM with schema for structured output
    planner = llm.with_structured_output(Sections)


    @task
    def orchestrator(topic: str):
        """Orchestrator that generates a plan for the report"""
        # Generate queries
        report_sections = planner.invoke(
            [
                SystemMessage(content="Generate a plan for the report."),
                HumanMessage(content=f"Here is the report topic: {topic}"),
            ]
        )

        return report_sections.sections


    @task
    def llm_call(section: Section):
        """Worker writes a section of the report"""

        # Generate section
        result = llm.invoke(
            [
                SystemMessage(content="Write a report section."),
                HumanMessage(
                    content=f"Here is the section name: {section.name} and description: {section.description}"
                ),
            ]
        )

        # Write the updated section to completed sections
        return result.content


    @task
    def synthesizer(completed_sections: list[str]):
        """Synthesize full report from sections"""
        final_report = "\n\n---\n\n".join(completed_sections)
        return final_report


    @entrypoint()
    def orchestrator_worker(topic: str):
        sections = orchestrator(topic).result()
        section_futures = [llm_call(section) for section in sections]
        final_report = synthesizer(
            [section_fut.result() for section_fut in section_futures]
        ).result()
        return final_report

    # Invoke
    report = orchestrator_worker.invoke("Create a report on LLM scaling laws")
    from IPython.display import Markdown
    Markdown(report)
    ```

    **LangSmith Trace**

    https://smith.langchain.com/public/75a636d0-6179-4a12-9836-e0aa571e87c5/r
    :::

    :::js
    ```typescript
    // Schema for structured output to use in planning
    const Section = z.object({
      name: z.string().describe("Name for this section of the report."),
      description: z.string().describe("Brief overview of the main topics and concepts to be covered in this section."),
    });

    const Sections = z.object({
      sections: z.array(Section).describe("Sections of the report."),
    });

    // Augment the LLM with schema for structured output
    const planner = llm.withStructuredOutput(Sections);

    const orchestrator = task("orchestrator", async (topic: string) => {
      // Orchestrator that generates a plan for the report
      const reportSections = await planner.invoke([
        new SystemMessage("Generate a plan for the report."),
        new HumanMessage(`Here is the report topic: ${topic}`),
      ]);
      return reportSections.sections;
    });

    const llmCall = task("llm_call", async (section: z.infer<typeof Section>) => {
      // Worker writes a section of the report
      const result = await llm.invoke([
        new SystemMessage("Write a report section."),
        new HumanMessage(
          `Here is the section name: ${section.name} and description: ${section.description}`
        ),
      ]);
      return result.content;
    });

    const synthesizer = task("synthesizer", (completedSections: string[]) => {
      // Synthesize full report from sections
      const finalReport = completedSections.join("\n\n---\n\n");
      return finalReport;
    });

    const orchestratorWorker = entrypoint("orchestratorWorker", async (topic: string) => {
      const sections = await orchestrator(topic);
      const sectionFutures = sections.map((section) => llmCall(section));
      const finalReport = await synthesizer(
        await Promise.all(sectionFutures)
      );
      return finalReport;
    });

    // Invoke
    const report = await orchestratorWorker.invoke("Create a report on LLM scaling laws");
    console.log(report);
    ```
    :::

## 평가자-최적화자

평가자-최적화자 워크플로우에서는 한 LLM 호출이 응답을 생성하고 다른 호출이 루프에서 평가와 피드백을 제공합니다:

> 이 워크플로우를 사용하는 경우: 이 워크플로우는 명확한 평가 기준이 있고 반복적인 개선이 측정 가능한 가치를 제공할 때 특히 효과적입니다. 적합성의 두 가지 징후는 첫째, 사람이 피드백을 명확하게 표현할 때 LLM 응답을 입증 가능하게 개선할 수 있다는 것이고, 둘째, LLM이 그러한 피드백을 제공할 수 있다는 것입니다. 이것은 인간 작가가 세련된 문서를 작성할 때 거치는 반복적인 작성 프로세스와 유사합니다.

![evaluator_optimizer.png](./workflows/img/evaluator_optimizer.png)

=== "Graph API"

    :::python
    ```python
    # Graph state
    class State(TypedDict):
        joke: str
        topic: str
        feedback: str
        funny_or_not: str


    # Schema for structured output to use in evaluation
    class Feedback(BaseModel):
        grade: Literal["funny", "not funny"] = Field(
            description="Decide if the joke is funny or not.",
        )
        feedback: str = Field(
            description="If the joke is not funny, provide feedback on how to improve it.",
        )


    # Augment the LLM with schema for structured output
    evaluator = llm.with_structured_output(Feedback)


    # Nodes
    def llm_call_generator(state: State):
        """LLM generates a joke"""

        if state.get("feedback"):
            msg = llm.invoke(
                f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
            )
        else:
            msg = llm.invoke(f"Write a joke about {state['topic']}")
        return {"joke": msg.content}


    def llm_call_evaluator(state: State):
        """LLM evaluates the joke"""

        grade = evaluator.invoke(f"Grade the joke {state['joke']}")
        return {"funny_or_not": grade.grade, "feedback": grade.feedback}


    # Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
    def route_joke(state: State):
        """Route back to joke generator or end based upon feedback from the evaluator"""

        if state["funny_or_not"] == "funny":
            return "Accepted"
        elif state["funny_or_not"] == "not funny":
            return "Rejected + Feedback"


    # Build workflow
    optimizer_builder = StateGraph(State)

    # Add the nodes
    optimizer_builder.add_node("llm_call_generator", llm_call_generator)
    optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

    # Add edges to connect nodes
    optimizer_builder.add_edge(START, "llm_call_generator")
    optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
    optimizer_builder.add_conditional_edges(
        "llm_call_evaluator",
        route_joke,
        {  # Name returned by route_joke : Name of next node to visit
            "Accepted": END,
            "Rejected + Feedback": "llm_call_generator",
        },
    )

    # Compile the workflow
    optimizer_workflow = optimizer_builder.compile()

    # Show the workflow
    display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))

    # Invoke
    state = optimizer_workflow.invoke({"topic": "Cats"})
    print(state["joke"])
    ```

    **LangSmith Trace**

    https://smith.langchain.com/public/86ab3e60-2000-4bff-b988-9b89a3269789/r

    **Resources:**

    **Examples**

    [Here](https://github.com/langchain-ai/local-deep-researcher) is an assistant that uses evaluator-optimizer to improve a report. See our video [here](https://www.youtube.com/watch?v=XGuTzHoqlj8).

    [Here](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/) is a RAG workflow that grades answers for hallucinations or errors. See our video [here](https://www.youtube.com/watch?v=bq1Plo2RhYI).
    :::

    :::js
    ```typescript
    // Graph state
    const State = z.object({
      joke: z.string().optional(),
      topic: z.string(),
      feedback: z.string().optional(),
      funny_or_not: z.string().optional(),
    });

    // Schema for structured output to use in evaluation
    const Feedback = z.object({
      grade: z.enum(["funny", "not funny"]).describe("Decide if the joke is funny or not."),
      feedback: z.string().describe("If the joke is not funny, provide feedback on how to improve it."),
    });

    // Augment the LLM with schema for structured output
    const evaluator = llm.withStructuredOutput(Feedback);

    // Nodes
    const llmCallGenerator = async (state: z.infer<typeof State>) => {
      // LLM generates a joke
      let msg;
      if (state.feedback) {
        msg = await llm.invoke(
          `Write a joke about ${state.topic} but take into account the feedback: ${state.feedback}`
        );
      } else {
        msg = await llm.invoke(`Write a joke about ${state.topic}`);
      }
      return { joke: msg.content };
    };

    const llmCallEvaluator = async (state: z.infer<typeof State>) => {
      // LLM evaluates the joke
      const grade = await evaluator.invoke(`Grade the joke ${state.joke}`);
      return { funny_or_not: grade.grade, feedback: grade.feedback };
    };

    // Conditional edge function to route back to joke generator or end
    const routeJoke = (state: z.infer<typeof State>) => {
      // Route back to joke generator or end based upon feedback from the evaluator
      if (state.funny_or_not === "funny") {
        return "Accepted";
      } else if (state.funny_or_not === "not funny") {
        return "Rejected + Feedback";
      }
    };

    // Build workflow
    const optimizerBuilder = new StateGraph(State)
      .addNode("llm_call_generator", llmCallGenerator)
      .addNode("llm_call_evaluator", llmCallEvaluator)
      .addEdge(START, "llm_call_generator")
      .addEdge("llm_call_generator", "llm_call_evaluator")
      .addConditionalEdges(
        "llm_call_evaluator",
        routeJoke,
        {
          "Accepted": END,
          "Rejected + Feedback": "llm_call_generator",
        }
      );

    // Compile the workflow
    const optimizerWorkflow = optimizerBuilder.compile();

    // Invoke
    const state = await optimizerWorkflow.invoke({ topic: "Cats" });
    console.log(state.joke);
    ```
    :::

=== "Functional API"

    :::python
    ```python
    # Schema for structured output to use in evaluation
    class Feedback(BaseModel):
        grade: Literal["funny", "not funny"] = Field(
            description="Decide if the joke is funny or not.",
        )
        feedback: str = Field(
            description="If the joke is not funny, provide feedback on how to improve it.",
        )


    # Augment the LLM with schema for structured output
    evaluator = llm.with_structured_output(Feedback)


    # Nodes
    @task
    def llm_call_generator(topic: str, feedback: Feedback):
        """LLM generates a joke"""
        if feedback:
            msg = llm.invoke(
                f"Write a joke about {topic} but take into account the feedback: {feedback}"
            )
        else:
            msg = llm.invoke(f"Write a joke about {topic}")
        return msg.content


    @task
    def llm_call_evaluator(joke: str):
        """LLM evaluates the joke"""
        feedback = evaluator.invoke(f"Grade the joke {joke}")
        return feedback


    @entrypoint()
    def optimizer_workflow(topic: str):
        feedback = None
        while True:
            joke = llm_call_generator(topic, feedback).result()
            feedback = llm_call_evaluator(joke).result()
            if feedback.grade == "funny":
                break

        return joke

    # Invoke
    for step in optimizer_workflow.stream("Cats", stream_mode="updates"):
        print(step)
        print("\n")
    ```

    **LangSmith Trace**

    https://smith.langchain.com/public/f66830be-4339-4a6b-8a93-389ce5ae27b4/r
    :::

    :::js
    ```typescript
    // Schema for structured output to use in evaluation
    const Feedback = z.object({
      grade: z.enum(["funny", "not funny"]).describe("Decide if the joke is funny or not."),
      feedback: z.string().describe("If the joke is not funny, provide feedback on how to improve it."),
    });

    // Augment the LLM with schema for structured output
    const evaluator = llm.withStructuredOutput(Feedback);

    // Nodes
    const llmCallGenerator = task("llm_call_generator", async (topic: string, feedback?: string) => {
      // LLM generates a joke
      if (feedback) {
        const msg = await llm.invoke(
          `Write a joke about ${topic} but take into account the feedback: ${feedback}`
        );
        return msg.content;
      } else {
        const msg = await llm.invoke(`Write a joke about ${topic}`);
        return msg.content;
      }
    });

    const llmCallEvaluator = task("llm_call_evaluator", async (joke: string) => {
      // LLM evaluates the joke
      const feedback = await evaluator.invoke(`Grade the joke ${joke}`);
      return feedback;
    });

    const optimizerWorkflow = entrypoint("optimizerWorkflow", async (topic: string) => {
      let feedback;
      while (true) {
        const joke = await llmCallGenerator(topic, feedback?.feedback);
        feedback = await llmCallEvaluator(joke);
        if (feedback.grade === "funny") {
          return joke;
        }
      }
    });

    // Invoke
    const stream = await optimizerWorkflow.stream("Cats", { streamMode: "updates" });
    for await (const step of stream) {
      console.log(step);
      console.log("\n");
    }
    ```
    :::

## 에이전트

에이전트는 일반적으로 루프에서 환경 피드백을 기반으로 (도구 호출을 통해) 작업을 수행하는 LLM으로 구현됩니다. `Building Effective Agents`에 대한 Anthropic 블로그에서 언급한 것처럼:

> 에이전트는 정교한 작업을 처리할 수 있지만 구현은 종종 간단합니다. 일반적으로 루프에서 환경 피드백을 기반으로 도구를 사용하는 LLM일 뿐입니다. 따라서 도구 세트와 문서를 명확하고 신중하게 설계하는 것이 중요합니다.

> 에이전트를 사용하는 경우: 에이전트는 필요한 단계 수를 예측하기 어렵거나 불가능하고 고정된 경로를 하드코딩할 수 없는 개방형 문제에 사용할 수 있습니다. LLM은 잠재적으로 많은 턴 동안 작동하며 의사 결정에 대해 어느 정도 신뢰가 있어야 합니다. 에이전트의 자율성은 신뢰할 수 있는 환경에서 작업을 확장하는 데 이상적입니다.

![agent.png](./workflows/img/agent.png)

:::python

```python
from langchain_core.tools import tool


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)
```

:::

:::js

```typescript
import { tool } from "@langchain/core/tools";

// Define tools
const multiply = tool(
  async ({ a, b }: { a: number; b: number }) => {
    return a * b;
  },
  {
    name: "multiply",
    description: "Multiply a and b.",
    schema: z.object({
      a: z.number().describe("first int"),
      b: z.number().describe("second int"),
    }),
  }
);

const add = tool(
  async ({ a, b }: { a: number; b: number }) => {
    return a + b;
  },
  {
    name: "add",
    description: "Adds a and b.",
    schema: z.object({
      a: z.number().describe("first int"),
      b: z.number().describe("second int"),
    }),
  }
);

const divide = tool(
  async ({ a, b }: { a: number; b: number }) => {
    return a / b;
  },
  {
    name: "divide",
    description: "Divide a and b.",
    schema: z.object({
      a: z.number().describe("first int"),
      b: z.number().describe("second int"),
    }),
  }
);

// Augment the LLM with tools
const tools = [add, multiply, divide];
const toolsByName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));
const llmWithTools = llm.bindTools(tools);
```

:::

=== "Graph API"

    :::python
    ```python
    from langgraph.graph import MessagesState
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage


    # Nodes
    def llm_call(state: MessagesState):
        """LLM decides whether to call a tool or not"""

        return {
            "messages": [
                llm_with_tools.invoke(
                    [
                        SystemMessage(
                            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                        )
                    ]
                    + state["messages"]
                )
            ]
        }


    def tool_node(state: dict):
        """Performs the tool call"""

        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}


    # Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
    def should_continue(state: MessagesState) -> Literal["Action", END]:
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]
        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "Action"
        # Otherwise, we stop (reply to the user)
        return END


    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("environment", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {
            # Name returned by should_continue : Name of next node to visit
            "Action": "environment",
            END: END,
        },
    )
    agent_builder.add_edge("environment", "llm_call")

    # Compile the agent
    agent = agent_builder.compile()

    # Show the agent
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

    # Invoke
    messages = [HumanMessage(content="Add 3 and 4.")]
    messages = agent.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
    ```

    **LangSmith Trace**

    https://smith.langchain.com/public/051f0391-6761-4f8c-a53b-22231b016690/r

    **Resources:**

    **LangChain Academy**

    See our lesson on agents [here](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/agent.ipynb).

    **Examples**

    [Here](https://github.com/langchain-ai/memory-agent) is a project that uses a tool calling agent to create / store long-term memories.
    :::

    :::js
    ```typescript
    import { MessagesZodState, ToolNode } from "@langchain/langgraph/prebuilt";
    import { SystemMessage, HumanMessage, ToolMessage, isAIMessage } from "@langchain/core/messages";

    // Nodes
    const llmCall = async (state: z.infer<typeof MessagesZodState>) => {
      // LLM decides whether to call a tool or not
      const response = await llmWithTools.invoke([
        new SystemMessage(
          "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
        ),
        ...state.messages,
      ]);
      return { messages: [response] };
    };

    const toolNode = new ToolNode(tools);

    // Conditional edge function to route to the tool node or end
    const shouldContinue = (state: z.infer<typeof MessagesZodState>) => {
      // Decide if we should continue the loop or stop
      const messages = state.messages;
      const lastMessage = messages[messages.length - 1];
      // If the LLM makes a tool call, then perform an action
      if (isAIMessage(lastMessage) && lastMessage.tool_calls?.length) {
        return "Action";
      }
      // Otherwise, we stop (reply to the user)
      return END;
    };

    // Build workflow
    const agentBuilder = new StateGraph(MessagesZodState)
      .addNode("llm_call", llmCall)
      .addNode("environment", toolNode)
      .addEdge(START, "llm_call")
      .addConditionalEdges(
        "llm_call",
        shouldContinue,
        {
          "Action": "environment",
          [END]: END,
        }
      )
      .addEdge("environment", "llm_call");

    // Compile the agent
    const agent = agentBuilder.compile();

    // Invoke
    const messages = [new HumanMessage("Add 3 and 4.")];
    const result = await agent.invoke({ messages });
    for (const m of result.messages) {
      console.log(`${m.getType()}: ${m.content}`);
    }
    ```
    :::

=== "Functional API"

    :::python
    ```python
    from langgraph.graph import add_messages
    from langchain_core.messages import (
        SystemMessage,
        HumanMessage,
        BaseMessage,
        ToolCall,
    )


    @task
    def call_llm(messages: list[BaseMessage]):
        """LLM decides whether to call a tool or not"""
        return llm_with_tools.invoke(
            [
                SystemMessage(
                    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                )
            ]
            + messages
        )


    @task
    def call_tool(tool_call: ToolCall):
        """Performs the tool call"""
        tool = tools_by_name[tool_call["name"]]
        return tool.invoke(tool_call)


    @entrypoint()
    def agent(messages: list[BaseMessage]):
        llm_response = call_llm(messages).result()

        while True:
            if not llm_response.tool_calls:
                break

            # Execute tools
            tool_result_futures = [
                call_tool(tool_call) for tool_call in llm_response.tool_calls
            ]
            tool_results = [fut.result() for fut in tool_result_futures]
            messages = add_messages(messages, [llm_response, *tool_results])
            llm_response = call_llm(messages).result()

        messages = add_messages(messages, llm_response)
        return messages

    # Invoke
    messages = [HumanMessage(content="Add 3 and 4.")]
    for chunk in agent.stream(messages, stream_mode="updates"):
        print(chunk)
        print("\n")
    ```

    **LangSmith Trace**

    https://smith.langchain.com/public/42ae8bf9-3935-4504-a081-8ddbcbfc8b2e/r
    :::

    :::js
    ```typescript
    import { addMessages } from "@langchain/langgraph";
    import {
      SystemMessage,
      HumanMessage,
      BaseMessage,
      ToolCall,
    } from "@langchain/core/messages";

    const callLlm = task("call_llm", async (messages: BaseMessage[]) => {
      // LLM decides whether to call a tool or not
      return await llmWithTools.invoke([
        new SystemMessage(
          "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
        ),
        ...messages,
      ]);
    });

    const callTool = task("call_tool", async (toolCall: ToolCall) => {
      // Performs the tool call
      const tool = toolsByName[toolCall.name];
      return await tool.invoke(toolCall);
    });

    const agent = entrypoint("agent", async (messages: BaseMessage[]) => {
      let currentMessages = messages;
      let llmResponse = await callLlm(currentMessages);

      while (true) {
        if (!llmResponse.tool_calls?.length) {
          break;
        }

        // Execute tools
        const toolResults = await Promise.all(
          llmResponse.tool_calls.map((toolCall) => callTool(toolCall))
        );

        // Append to message list
        currentMessages = addMessages(currentMessages, [
          llmResponse,
          ...toolResults,
        ]);

        // Call model again
        llmResponse = await callLlm(currentMessages);
      }

      return llmResponse;
    });

    // Invoke
    const messages = [new HumanMessage("Add 3 and 4.")];
    const stream = await agent.stream(messages, { streamMode: "updates" });
    for await (const chunk of stream) {
      console.log(chunk);
      console.log("\n");
    }
    ```
    :::

#### 사전 구축

:::python
LangGraph는 위에서 정의한 에이전트를 생성하기 위한 **사전 구축 방법**도 제공합니다(@[`create_react_agent`][create_react_agent] 함수 사용):

https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/

```python
from langgraph.prebuilt import create_react_agent

# Pass in:
# (1) the augmented LLM with tools
# (2) the tools list (which is used to create the tool node)
pre_built_agent = create_react_agent(llm, tools=tools)

# Show the agent
display(Image(pre_built_agent.get_graph().draw_mermaid_png()))

# Invoke
messages = [HumanMessage(content="Add 3 and 4.")]
messages = pre_built_agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
```

**LangSmith Trace**

https://smith.langchain.com/public/abab6a44-29f6-4b97-8164-af77413e494d/r
:::

:::js
LangGraph는 위에서 정의한 에이전트를 생성하기 위한 **사전 구축 방법**도 제공합니다(@[`createReactAgent`][create_react_agent] 함수 사용):

```typescript
import { createReactAgent } from "@langchain/langgraph/prebuilt";

// Pass in:
// (1) the augmented LLM with tools
// (2) the tools list (which is used to create the tool node)
const preBuiltAgent = createReactAgent({ llm, tools });

// Invoke
const messages = [new HumanMessage("Add 3 and 4.")];
const result = await preBuiltAgent.invoke({ messages });
for (const m of result.messages) {
  console.log(`${m.getType()}: ${m.content}`);
}
```

:::

## LangGraph가 제공하는 것

LangGraph에서 위의 각 항목을 구성하면 다음과 같은 몇 가지 이점을 얻을 수 있습니다:

### 지속성: 휴먼 인 더 루프

LangGraph 지속성 레이어는 작업의 중단 및 승인(예: Human In The Loop)을 지원합니다. [LangChain Academy의 모듈 3](https://github.com/langchain-ai/langchain-academy/tree/main/module-3)을 참조하세요.

### 지속성: 메모리

LangGraph 지속성 레이어는 대화형(단기) 메모리와 장기 메모리를 지원합니다. LangChain Academy의 [모듈 2](https://github.com/langchain-ai/langchain-academy/tree/main/module-2) [및 5](https://github.com/langchain-ai/langchain-academy/tree/main/module-5)를 참조하세요:

### 스트리밍

LangGraph는 워크플로우 / 에이전트 출력 또는 중간 상태를 스트리밍하는 여러 방법을 제공합니다. [LangChain Academy의 모듈 3](https://github.com/langchain-ai/langchain-academy/blob/main/module-3/streaming-interruption.ipynb)을 참조하세요.

### 배포

LangGraph는 배포, 관찰 가능성 및 평가를 위한 쉬운 진입로를 제공합니다. LangChain Academy의 [모듈 6](https://github.com/langchain-ai/langchain-academy/tree/main/module-6)을 참조하세요.
