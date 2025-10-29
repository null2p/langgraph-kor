---
search:
  boost: 2
---

# LangGraph 런타임

:::python
@[Pregel]은 LangGraph의 런타임을 구현하여 LangGraph 애플리케이션의 실행을 관리합니다.

@[StateGraph][StateGraph]를 컴파일하거나 @[entrypoint][entrypoint]를 생성하면 입력과 함께 호출할 수 있는 @[Pregel] 인스턴스가 생성됩니다.
:::

:::js
@[Pregel]은 LangGraph의 런타임을 구현하여 LangGraph 애플리케이션의 실행을 관리합니다.

@[StateGraph][StateGraph]를 컴파일하거나 @[entrypoint][entrypoint]를 생성하면 입력과 함께 호출할 수 있는 @[Pregel] 인스턴스가 생성됩니다.
:::

이 가이드는 런타임을 높은 수준에서 설명하고 Pregel로 애플리케이션을 직접 구현하는 지침을 제공합니다.

:::python

> **참고:** @[Pregel] 런타임은 그래프를 사용한 대규모 병렬 계산의 효율적인 방법을 설명하는 [Google's Pregel algorithm](https://research.google/pubs/pub37252/)의 이름을 따서 명명되었습니다.

:::

:::js

> **참고:** @[Pregel] 런타임은 그래프를 사용한 대규모 병렬 계산의 효율적인 방법을 설명하는 [Google's Pregel algorithm](https://research.google/pubs/pub37252/)의 이름을 따서 명명되었습니다.

:::

## 개요

LangGraph에서 Pregel은 [**actors**](https://en.wikipedia.org/wiki/Actor_model)와 **channels**를 단일 애플리케이션으로 결합합니다. **Actors**는 채널에서 데이터를 읽고 채널에 데이터를 씁니다. Pregel은 **Pregel Algorithm**/**Bulk Synchronous Parallel** 모델을 따라 애플리케이션의 실행을 여러 단계로 구성합니다.

각 단계는 세 가지 단계로 구성됩니다:

- **계획(Plan)**: 이 단계에서 실행할 **actors**를 결정합니다. 예를 들어, 첫 번째 단계에서는 특수 **input** 채널을 구독하는 **actors**를 선택하고, 후속 단계에서는 이전 단계에서 업데이트된 채널을 구독하는 **actors**를 선택합니다.
- **실행(Execution)**: 모두 완료되거나, 하나가 실패하거나, 시간 초과에 도달할 때까지 선택된 모든 **actors**를 병렬로 실행합니다. 이 단계에서 채널 업데이트는 다음 단계까지 actors에게 보이지 않습니다.
- **업데이트(Update)**: 이 단계에서 **actors**가 작성한 값으로 채널을 업데이트합니다.

실행할 **actors**가 선택되지 않거나 최대 단계 수에 도달할 때까지 반복합니다.

## Actors

**actor**는 `PregelNode`입니다. 채널을 구독하고, 채널에서 데이터를 읽고, 채널에 데이터를 씁니다. Pregel 알고리즘의 **actor**로 생각할 수 있습니다. `PregelNodes`는 LangChain의 Runnable 인터페이스를 구현합니다.

## Channels

채널은 actors(PregelNodes) 간에 통신하는 데 사용됩니다. 각 채널에는 값 타입, 업데이트 타입 및 업데이트 함수가 있으며, 이는 업데이트 시퀀스를 가져와 저장된 값을 수정합니다. 채널은 한 체인에서 다른 체인으로 데이터를 보내거나 체인에서 미래 단계의 자신에게 데이터를 보내는 데 사용할 수 있습니다. LangGraph는 여러 내장 채널을 제공합니다:

:::python

- @[LastValue][LastValue]: 기본 채널로, 채널에 보낸 마지막 값을 저장하며, 입력 및 출력 값에 유용하거나 한 단계에서 다음 단계로 데이터를 보내는 데 유용합니다.
- @[Topic][Topic]: 구성 가능한 PubSub Topic으로, **actors** 간에 여러 값을 보내거나 출력을 축적하는 데 유용합니다. 값을 중복 제거하거나 여러 단계에 걸쳐 값을 축적하도록 구성할 수 있습니다.
- @[BinaryOperatorAggregate][BinaryOperatorAggregate]: 지속적인 값을 저장하고 현재 값과 채널에 보낸 각 업데이트에 이진 연산자를 적용하여 업데이트하며, 여러 단계에 걸쳐 집계를 계산하는 데 유용합니다. 예: `total = BinaryOperatorAggregate(int, operator.add)`
  :::

:::js

- @[LastValue]: 기본 채널로, 채널에 보낸 마지막 값을 저장하며, 입력 및 출력 값에 유용하거나 한 단계에서 다음 단계로 데이터를 보내는 데 유용합니다.
- @[Topic]: 구성 가능한 PubSub Topic으로, **actors** 간에 여러 값을 보내거나 출력을 축적하는 데 유용합니다. 값을 중복 제거하거나 여러 단계에 걸쳐 값을 축적하도록 구성할 수 있습니다.
- @[BinaryOperatorAggregate]: 지속적인 값을 저장하고 현재 값과 채널에 보낸 각 업데이트에 이진 연산자를 적용하여 업데이트하며, 여러 단계에 걸쳐 집계를 계산하는 데 유용합니다. 예: `total = BinaryOperatorAggregate(int, operator.add)`
  :::

## 예제

:::python
대부분의 사용자는 @[StateGraph][StateGraph] API 또는 @[entrypoint][entrypoint] 데코레이터를 통해 Pregel과 상호 작용하지만 Pregel과 직접 상호 작용하는 것도 가능합니다.
:::

:::js
대부분의 사용자는 @[StateGraph] API 또는 @[entrypoint] 데코레이터를 통해 Pregel과 상호 작용하지만 Pregel과 직접 상호 작용하는 것도 가능합니다.
:::

다음은 Pregel API의 감각을 제공하는 몇 가지 다른 예제입니다.

=== "Single node"

    :::python
    ```python
    from langgraph.channels import EphemeralValue
    from langgraph.pregel import Pregel, NodeBuilder

    node1 = (
        NodeBuilder().subscribe_only("a")
        .do(lambda x: x + x)
        .write_to("b")
    )

    app = Pregel(
        nodes={"node1": node1},
        channels={
            "a": EphemeralValue(str),
            "b": EphemeralValue(str),
        },
        input_channels=["a"],
        output_channels=["b"],
    )

    app.invoke({"a": "foo"})
    ```

    ```con
    {'b': 'foofoo'}
    ```
    :::

    :::js
    ```typescript
    import { EphemeralValue } from "@langchain/langgraph/channels";
    import { Pregel, NodeBuilder } from "@langchain/langgraph/pregel";

    const node1 = new NodeBuilder()
      .subscribeOnly("a")
      .do((x: string) => x + x)
      .writeTo("b");

    const app = new Pregel({
      nodes: { node1 },
      channels: {
        a: new EphemeralValue<string>(),
        b: new EphemeralValue<string>(),
      },
      inputChannels: ["a"],
      outputChannels: ["b"],
    });

    await app.invoke({ a: "foo" });
    ```

    ```console
    { b: 'foofoo' }
    ```
    :::

=== "Multiple nodes"

    :::python
    ```python
    from langgraph.channels import LastValue, EphemeralValue
    from langgraph.pregel import Pregel, NodeBuilder

    node1 = (
        NodeBuilder().subscribe_only("a")
        .do(lambda x: x + x)
        .write_to("b")
    )

    node2 = (
        NodeBuilder().subscribe_only("b")
        .do(lambda x: x + x)
        .write_to("c")
    )


    app = Pregel(
        nodes={"node1": node1, "node2": node2},
        channels={
            "a": EphemeralValue(str),
            "b": LastValue(str),
            "c": EphemeralValue(str),
        },
        input_channels=["a"],
        output_channels=["b", "c"],
    )

    app.invoke({"a": "foo"})
    ```

    ```con
    {'b': 'foofoo', 'c': 'foofoofoofoo'}
    ```
    :::

    :::js
    ```typescript
    import { LastValue, EphemeralValue } from "@langchain/langgraph/channels";
    import { Pregel, NodeBuilder } from "@langchain/langgraph/pregel";

    const node1 = new NodeBuilder()
      .subscribeOnly("a")
      .do((x: string) => x + x)
      .writeTo("b");

    const node2 = new NodeBuilder()
      .subscribeOnly("b")
      .do((x: string) => x + x)
      .writeTo("c");

    const app = new Pregel({
      nodes: { node1, node2 },
      channels: {
        a: new EphemeralValue<string>(),
        b: new LastValue<string>(),
        c: new EphemeralValue<string>(),
      },
      inputChannels: ["a"],
      outputChannels: ["b", "c"],
    });

    await app.invoke({ a: "foo" });
    ```

    ```console
    { b: 'foofoo', c: 'foofoofoofoo' }
    ```
    :::

=== "Topic"

    :::python
    ```python
    from langgraph.channels import EphemeralValue, Topic
    from langgraph.pregel import Pregel, NodeBuilder

    node1 = (
        NodeBuilder().subscribe_only("a")
        .do(lambda x: x + x)
        .write_to("b", "c")
    )

    node2 = (
        NodeBuilder().subscribe_to("b")
        .do(lambda x: x["b"] + x["b"])
        .write_to("c")
    )

    app = Pregel(
        nodes={"node1": node1, "node2": node2},
        channels={
            "a": EphemeralValue(str),
            "b": EphemeralValue(str),
            "c": Topic(str, accumulate=True),
        },
        input_channels=["a"],
        output_channels=["c"],
    )

    app.invoke({"a": "foo"})
    ```

    ```pycon
    {'c': ['foofoo', 'foofoofoofoo']}
    ```
    :::

    :::js
    ```typescript
    import { EphemeralValue, Topic } from "@langchain/langgraph/channels";
    import { Pregel, NodeBuilder } from "@langchain/langgraph/pregel";

    const node1 = new NodeBuilder()
      .subscribeOnly("a")
      .do((x: string) => x + x)
      .writeTo("b", "c");

    const node2 = new NodeBuilder()
      .subscribeTo("b")
      .do((x: { b: string }) => x.b + x.b)
      .writeTo("c");

    const app = new Pregel({
      nodes: { node1, node2 },
      channels: {
        a: new EphemeralValue<string>(),
        b: new EphemeralValue<string>(),
        c: new Topic<string>({ accumulate: true }),
      },
      inputChannels: ["a"],
      outputChannels: ["c"],
    });

    await app.invoke({ a: "foo" });
    ```

    ```console
    { c: ['foofoo', 'foofoofoofoo'] }
    ```
    :::

=== "BinaryOperatorAggregate"

    This examples demonstrates how to use the BinaryOperatorAggregate channel to implement a reducer.

    :::python
    ```python
    from langgraph.channels import EphemeralValue, BinaryOperatorAggregate
    from langgraph.pregel import Pregel, NodeBuilder


    node1 = (
        NodeBuilder().subscribe_only("a")
        .do(lambda x: x + x)
        .write_to("b", "c")
    )

    node2 = (
        NodeBuilder().subscribe_only("b")
        .do(lambda x: x + x)
        .write_to("c")
    )

    def reducer(current, update):
        if current:
            return current + " | " + update
        else:
            return update

    app = Pregel(
        nodes={"node1": node1, "node2": node2},
        channels={
            "a": EphemeralValue(str),
            "b": EphemeralValue(str),
            "c": BinaryOperatorAggregate(str, operator=reducer),
        },
        input_channels=["a"],
        output_channels=["c"],
    )

    app.invoke({"a": "foo"})
    ```
    :::

    :::js
    ```typescript
    import { EphemeralValue, BinaryOperatorAggregate } from "@langchain/langgraph/channels";
    import { Pregel, NodeBuilder } from "@langchain/langgraph/pregel";

    const node1 = new NodeBuilder()
      .subscribeOnly("a")
      .do((x: string) => x + x)
      .writeTo("b", "c");

    const node2 = new NodeBuilder()
      .subscribeOnly("b")
      .do((x: string) => x + x)
      .writeTo("c");

    const reducer = (current: string, update: string) => {
      if (current) {
        return current + " | " + update;
      } else {
        return update;
      }
    };

    const app = new Pregel({
      nodes: { node1, node2 },
      channels: {
        a: new EphemeralValue<string>(),
        b: new EphemeralValue<string>(),
        c: new BinaryOperatorAggregate<string>({ operator: reducer }),
      },
      inputChannels: ["a"],
      outputChannels: ["c"],
    });

    await app.invoke({ a: "foo" });
    ```
    :::

=== "Cycle"

    :::python

    This example demonstrates how to introduce a cycle in the graph, by having
    a chain write to a channel it subscribes to. Execution will continue
    until a `None` value is written to the channel.

    ```python
    from langgraph.channels import EphemeralValue
    from langgraph.pregel import Pregel, NodeBuilder, ChannelWriteEntry

    example_node = (
        NodeBuilder().subscribe_only("value")
        .do(lambda x: x + x if len(x) < 10 else None)
        .write_to(ChannelWriteEntry("value", skip_none=True))
    )

    app = Pregel(
        nodes={"example_node": example_node},
        channels={
            "value": EphemeralValue(str),
        },
        input_channels=["value"],
        output_channels=["value"],
    )

    app.invoke({"value": "a"})
    ```

    ```pycon
    {'value': 'aaaaaaaaaaaaaaaa'}
    ```
    :::

    :::js

    This example demonstrates how to introduce a cycle in the graph, by having
    a chain write to a channel it subscribes to. Execution will continue
    until a `null` value is written to the channel.

    ```typescript
    import { EphemeralValue } from "@langchain/langgraph/channels";
    import { Pregel, NodeBuilder, ChannelWriteEntry } from "@langchain/langgraph/pregel";

    const exampleNode = new NodeBuilder()
      .subscribeOnly("value")
      .do((x: string) => x.length < 10 ? x + x : null)
      .writeTo(new ChannelWriteEntry("value", { skipNone: true }));

    const app = new Pregel({
      nodes: { exampleNode },
      channels: {
        value: new EphemeralValue<string>(),
      },
      inputChannels: ["value"],
      outputChannels: ["value"],
    });

    await app.invoke({ value: "a" });
    ```

    ```console
    { value: 'aaaaaaaaaaaaaaaa' }
    ```
    :::

## 고수준 API

LangGraph는 Pregel 애플리케이션을 생성하기 위한 두 가지 고수준 API를 제공합니다: [StateGraph (Graph API)](./low_level.md)와 [Functional API](functional_api.md)입니다.

=== "StateGraph (Graph API)"

    :::python

    The @[StateGraph (Graph API)][StateGraph] is a higher-level abstraction that simplifies the creation of Pregel applications. It allows you to define a graph of nodes and edges. When you compile the graph, the StateGraph API automatically creates the Pregel application for you.

    ```python
    from typing import TypedDict, Optional

    from langgraph.constants import START
    from langgraph.graph import StateGraph

    class Essay(TypedDict):
        topic: str
        content: Optional[str]
        score: Optional[float]

    def write_essay(essay: Essay):
        return {
            "content": f"Essay about {essay['topic']}",
        }

    def score_essay(essay: Essay):
        return {
            "score": 10
        }

    builder = StateGraph(Essay)
    builder.add_node(write_essay)
    builder.add_node(score_essay)
    builder.add_edge(START, "write_essay")

    # Compile the graph.
    # This will return a Pregel instance.
    graph = builder.compile()
    ```
    :::

    :::js

    The @[StateGraph (Graph API)][StateGraph] is a higher-level abstraction that simplifies the creation of Pregel applications. It allows you to define a graph of nodes and edges. When you compile the graph, the StateGraph API automatically creates the Pregel application for you.

    ```typescript
    import { START, StateGraph } from "@langchain/langgraph";

    interface Essay {
      topic: string;
      content?: string;
      score?: number;
    }

    const writeEssay = (essay: Essay) => {
      return {
        content: `Essay about ${essay.topic}`,
      };
    };

    const scoreEssay = (essay: Essay) => {
      return {
        score: 10
      };
    };

    const builder = new StateGraph<Essay>({
      channels: {
        topic: null,
        content: null,
        score: null,
      }
    })
      .addNode("writeEssay", writeEssay)
      .addNode("scoreEssay", scoreEssay)
      .addEdge(START, "writeEssay");

    // Compile the graph.
    // This will return a Pregel instance.
    const graph = builder.compile();
    ```
    :::

    The compiled Pregel instance will be associated with a list of nodes and channels. You can inspect the nodes and channels by printing them.

    :::python
    ```python
    print(graph.nodes)
    ```

    You will see something like this:

    ```pycon
    {'__start__': <langgraph.pregel.read.PregelNode at 0x7d05e3ba1810>,
     'write_essay': <langgraph.pregel.read.PregelNode at 0x7d05e3ba14d0>,
     'score_essay': <langgraph.pregel.read.PregelNode at 0x7d05e3ba1710>}
    ```

    ```python
    print(graph.channels)
    ```

    You should see something like this

    ```pycon
    {'topic': <langgraph.channels.last_value.LastValue at 0x7d05e3294d80>,
     'content': <langgraph.channels.last_value.LastValue at 0x7d05e3295040>,
     'score': <langgraph.channels.last_value.LastValue at 0x7d05e3295980>,
     '__start__': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e3297e00>,
     'write_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e32960c0>,
     'score_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e2d8ab80>,
     'branch:__start__:__self__:write_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e32941c0>,
     'branch:__start__:__self__:score_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e2d88800>,
     'branch:write_essay:__self__:write_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e3295ec0>,
     'branch:write_essay:__self__:score_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e2d8ac00>,
     'branch:score_essay:__self__:write_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e2d89700>,
     'branch:score_essay:__self__:score_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e2d8b400>,
     'start:write_essay': <langgraph.channels.ephemeral_value.EphemeralValue at 0x7d05e2d8b280>}
    ```
    :::

    :::js
    ```typescript
    console.log(graph.nodes);
    ```

    You will see something like this:

    ```console
    {
      __start__: PregelNode { ... },
      writeEssay: PregelNode { ... },
      scoreEssay: PregelNode { ... }
    }
    ```

    ```typescript
    console.log(graph.channels);
    ```

    You should see something like this

    ```console
    {
      topic: LastValue { ... },
      content: LastValue { ... },
      score: LastValue { ... },
      __start__: EphemeralValue { ... },
      writeEssay: EphemeralValue { ... },
      scoreEssay: EphemeralValue { ... },
      'branch:__start__:__self__:writeEssay': EphemeralValue { ... },
      'branch:__start__:__self__:scoreEssay': EphemeralValue { ... },
      'branch:writeEssay:__self__:writeEssay': EphemeralValue { ... },
      'branch:writeEssay:__self__:scoreEssay': EphemeralValue { ... },
      'branch:scoreEssay:__self__:writeEssay': EphemeralValue { ... },
      'branch:scoreEssay:__self__:scoreEssay': EphemeralValue { ... },
      'start:writeEssay': EphemeralValue { ... }
    }
    ```
    :::

=== "Functional API"

    :::python

    In the [Functional API](functional_api.md), you can use an @[`entrypoint`][entrypoint] to create a Pregel application. The `entrypoint` decorator allows you to define a function that takes input and returns output.

    ```python
    from typing import TypedDict, Optional

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.func import entrypoint

    class Essay(TypedDict):
        topic: str
        content: Optional[str]
        score: Optional[float]


    checkpointer = InMemorySaver()

    @entrypoint(checkpointer=checkpointer)
    def write_essay(essay: Essay):
        return {
            "content": f"Essay about {essay['topic']}",
        }

    print("Nodes: ")
    print(write_essay.nodes)
    print("Channels: ")
    print(write_essay.channels)
    ```

    ```pycon
    Nodes:
    {'write_essay': <langgraph.pregel.read.PregelNode object at 0x7d05e2f9aad0>}
    Channels:
    {'__start__': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7d05e2c906c0>, '__end__': <langgraph.channels.last_value.LastValue object at 0x7d05e2c90c40>, '__previous__': <langgraph.channels.last_value.LastValue object at 0x7d05e1007280>}
    ```
    :::

    :::js

    In the [Functional API](functional_api.md), you can use an @[`entrypoint`][entrypoint] to create a Pregel application. The `entrypoint` decorator allows you to define a function that takes input and returns output.

    ```typescript
    import { MemorySaver } from "@langchain/langgraph";
    import { entrypoint } from "@langchain/langgraph/func";

    interface Essay {
      topic: string;
      content?: string;
      score?: number;
    }

    const checkpointer = new MemorySaver();

    const writeEssay = entrypoint(
      { checkpointer, name: "writeEssay" },
      async (essay: Essay) => {
        return {
          content: `Essay about ${essay.topic}`,
        };
      }
    );

    console.log("Nodes: ");
    console.log(writeEssay.nodes);
    console.log("Channels: ");
    console.log(writeEssay.channels);
    ```

    ```console
    Nodes:
    { writeEssay: PregelNode { ... } }
    Channels:
    {
      __start__: EphemeralValue { ... },
      __end__: LastValue { ... },
      __previous__: LastValue { ... }
    }
    ```
    :::
