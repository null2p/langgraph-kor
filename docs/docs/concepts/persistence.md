---
search:
  boost: 2
---

# 지속성 (Persistence)

LangGraph에는 체크포인터를 통해 구현된 내장 지속성 레이어가 있습니다. 체크포인터와 함께 그래프를 컴파일하면, 체크포인터는 모든 super-step마다 그래프 상태의 `checkpoint`를 저장합니다. 이러한 체크포인트는 `thread`에 저장되며, 그래프 실행 후에 액세스할 수 있습니다. `thread`를 통해 실행 후 그래프 상태에 액세스할 수 있기 때문에, Human-in-the-loop, 메모리, 타임 트래블, 장애 허용 등 여러 강력한 기능이 모두 가능합니다. 아래에서 이러한 각 개념에 대해 자세히 설명하겠습니다.

![Checkpoints](img/persistence/checkpoints.jpg)

!!! info "LangGraph API는 체크포인팅을 자동으로 처리합니다"

    LangGraph API를 사용할 때는 체크포인터를 수동으로 구현하거나 구성할 필요가 없습니다. API가 백그라운드에서 모든 지속성 인프라를 처리합니다.

## Thread {#threads}

Thread는 체크포인터가 저장하는 각 체크포인트에 할당된 고유 ID 또는 thread 식별자입니다. [Run](./assistants.md#execution) 시퀀스의 누적된 상태를 포함합니다. Run이 실행되면 어시스턴트의 기본 그래프 [상태](../concepts/low_level.md#state)가 thread에 지속됩니다.

체크포인터와 함께 그래프를 호출할 때는 config의 `configurable` 부분에 `thread_id`를 **반드시** 지정해야 합니다:

:::python

```python
{"configurable": {"thread_id": "1"}}
```

:::

:::js

```typescript
{
  configurable: {
    thread_id: "1";
  }
}
```

:::

Thread의 현재 및 과거 상태를 조회할 수 있습니다. 상태를 지속하려면 run을 실행하기 전에 thread를 생성해야 합니다. LangGraph Platform API는 thread와 thread 상태를 생성하고 관리하기 위한 여러 엔드포인트를 제공합니다. 자세한 내용은 [API 레퍼런스](../cloud/reference/api/api_ref.html#tag/threads)를 참조하세요.

## Checkpoint {#checkpoints}

특정 시점의 thread 상태를 checkpoint라고 합니다. Checkpoint는 각 super-step마다 저장된 그래프 상태의 스냅샷이며, 다음과 같은 주요 속성을 가진 `StateSnapshot` 객체로 표현됩니다:

- `config`: 이 체크포인트와 연결된 config
- `metadata`: 이 체크포인트와 연결된 메타데이터
- `values`: 이 시점의 상태 채널 값
- `next`: 그래프에서 다음에 실행할 노드 이름의 튜플
- `tasks`: 다음에 실행할 작업에 대한 정보를 포함하는 `PregelTask` 객체의 튜플. 이전에 단계를 시도한 경우 오류 정보가 포함됩니다. 노드 내에서 그래프가 [동적으로](../how-tos/human_in_the_loop/add-human-in-the-loop.md#pause-using-interrupt) 중단된 경우, tasks에는 중단과 관련된 추가 데이터가 포함됩니다.

Checkpoint는 지속되며 나중에 thread의 상태를 복원하는 데 사용할 수 있습니다.

간단한 그래프를 다음과 같이 호출할 때 저장되는 체크포인트를 살펴보겠습니다:

:::python

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": ""}, config)
```

:::

:::js

```typescript
import { StateGraph, START, END, MemoryServer } from "@langchain/langgraph";
import { withLangGraph } from "@langchain/langgraph/zod";
import { z } from "zod";

const State = z.object({
  foo: z.string(),
  bar: withLangGraph(z.array(z.string()), {
    reducer: {
      fn: (x, y) => x.concat(y),
    },
    default: () => [],
  }),
});

const workflow = new StateGraph(State)
  .addNode("nodeA", (state) => {
    return { foo: "a", bar: ["a"] };
  })
  .addNode("nodeB", (state) => {
    return { foo: "b", bar: ["b"] };
  })
  .addEdge(START, "nodeA")
  .addEdge("nodeA", "nodeB")
  .addEdge("nodeB", END);

const checkpointer = new MemorySaver();
const graph = workflow.compile({ checkpointer });

const config = { configurable: { thread_id: "1" } };
await graph.invoke({ foo: "" }, config);
```

:::

:::js

```typescript
import { StateGraph, START, END, MemoryServer } from "@langchain/langgraph";
import { withLangGraph } from "@langchain/langgraph/zod";
import { z } from "zod";

const State = z.object({
  foo: z.string(),
  bar: withLangGraph(z.array(z.string()), {
    reducer: {
      fn: (x, y) => x.concat(y),
    },
    default: () => [],
  }),
});

const workflow = new StateGraph(State)
  .addNode("nodeA", (state) => {
    return { foo: "a", bar: ["a"] };
  })
  .addNode("nodeB", (state) => {
    return { foo: "b", bar: ["b"] };
  })
  .addEdge(START, "nodeA")
  .addEdge("nodeA", "nodeB")
  .addEdge("nodeB", END);

const checkpointer = new MemorySaver();
const graph = workflow.compile({ checkpointer });

const config = { configurable: { thread_id: "1" } };
await graph.invoke({ foo: "" }, config);
```

:::

:::python

그래프를 실행한 후 정확히 4개의 체크포인트가 생성됩니다:

- 다음에 실행할 노드로 `START`가 있는 빈 체크포인트
- 사용자 입력 `{'foo': '', 'bar': []}`과 다음에 실행할 노드로 `node_a`가 있는 체크포인트
- `node_a`의 출력 `{'foo': 'a', 'bar': ['a']}`과 다음에 실행할 노드로 `node_b`가 있는 체크포인트
- `node_b`의 출력 `{'foo': 'b', 'bar': ['a', 'b']}`과 다음에 실행할 노드가 없는 체크포인트

`bar` 채널에 reducer가 있기 때문에 `bar` 채널 값에는 두 노드의 출력이 모두 포함됩니다.

:::

:::js

그래프를 실행한 후 정확히 4개의 체크포인트가 생성됩니다:

- 다음에 실행할 노드로 `START`가 있는 빈 체크포인트
- 사용자 입력 `{'foo': '', 'bar': []}`과 다음에 실행할 노드로 `nodeA`가 있는 체크포인트
- `nodeA`의 출력 `{'foo': 'a', 'bar': ['a']}`과 다음에 실행할 노드로 `nodeB`가 있는 체크포인트
- `nodeB`의 출력 `{'foo': 'b', 'bar': ['a', 'b']}`과 다음에 실행할 노드가 없는 체크포인트

`bar` 채널에 reducer가 있기 때문에 `bar` 채널 값에는 두 노드의 출력이 모두 포함됩니다.
:::

### 상태 조회

:::python
저장된 그래프 상태와 상호작용할 때는 [thread 식별자](#threads)를 **반드시** 지정해야 합니다. `graph.get_state(config)`를 호출하여 그래프의 _최신_ 상태를 볼 수 있습니다. 이는 config에 제공된 thread ID와 연결된 최신 체크포인트 또는 제공된 경우 thread의 checkpoint ID와 연결된 체크포인트에 해당하는 `StateSnapshot` 객체를 반환합니다.

```python
# get the latest state snapshot
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)

# get a state snapshot for a specific checkpoint_id
config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
graph.get_state(config)
```

:::

:::js
저장된 그래프 상태와 상호작용할 때는 [thread 식별자](#threads)를 **반드시** 지정해야 합니다. `graph.getState(config)`를 호출하여 그래프의 _최신_ 상태를 볼 수 있습니다. 이는 config에 제공된 thread ID와 연결된 최신 체크포인트 또는 제공된 경우 thread의 checkpoint ID와 연결된 체크포인트에 해당하는 `StateSnapshot` 객체를 반환합니다.

```typescript
// get the latest state snapshot
const config = { configurable: { thread_id: "1" } };
await graph.getState(config);

// get a state snapshot for a specific checkpoint_id
const config = {
  configurable: {
    thread_id: "1",
    checkpoint_id: "1ef663ba-28fe-6528-8002-5a559208592c",
  },
};
await graph.getState(config);
```

:::

:::python
예제에서 `get_state`의 출력은 다음과 같습니다:

```
StateSnapshot(
    values={'foo': 'b', 'bar': ['a', 'b']},
    next=(),
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
    metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
    created_at='2024-08-29T19:19:38.821749+00:00',
    parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}}, tasks=()
)
```

:::

:::js
예제에서 `getState`의 출력은 다음과 같습니다:

```
StateSnapshot {
  values: { foo: 'b', bar: ['a', 'b'] },
  next: [],
  config: {
    configurable: {
      thread_id: '1',
      checkpoint_ns: '',
      checkpoint_id: '1ef663ba-28fe-6528-8002-5a559208592c'
    }
  },
  metadata: {
    source: 'loop',
    writes: { nodeB: { foo: 'b', bar: ['b'] } },
    step: 2
  },
  createdAt: '2024-08-29T19:19:38.821749+00:00',
  parentConfig: {
    configurable: {
      thread_id: '1',
      checkpoint_ns: '',
      checkpoint_id: '1ef663ba-28f9-6ec4-8001-31981c2c39f8'
    }
  },
  tasks: []
}
```

:::

### 상태 히스토리 조회

:::python
`graph.get_state_history(config)`를 호출하여 주어진 thread에 대한 그래프 실행의 전체 히스토리를 가져올 수 있습니다. 이는 config에 제공된 thread ID와 연결된 `StateSnapshot` 객체 목록을 반환합니다. 중요한 점은 체크포인트가 시간순으로 정렬되며 가장 최근 체크포인트 / `StateSnapshot`이 목록의 첫 번째에 위치한다는 것입니다.

```python
config = {"configurable": {"thread_id": "1"}}
list(graph.get_state_history(config))
```

:::

:::js
`graph.getStateHistory(config)`를 호출하여 주어진 thread에 대한 그래프 실행의 전체 히스토리를 가져올 수 있습니다. 이는 config에 제공된 thread ID와 연결된 `StateSnapshot` 객체 목록을 반환합니다. 중요한 점은 체크포인트가 시간순으로 정렬되며 가장 최근 체크포인트 / `StateSnapshot`이 목록의 첫 번째에 위치한다는 것입니다.

```typescript
const config = { configurable: { thread_id: "1" } };
for await (const state of graph.getStateHistory(config)) {
  console.log(state);
}
```

:::

:::python
예제에서 `get_state_history`의 출력은 다음과 같습니다:

```
[
    StateSnapshot(
        values={'foo': 'b', 'bar': ['a', 'b']},
        next=(),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
        metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
        created_at='2024-08-29T19:19:38.821749+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
        tasks=(),
    ),
    StateSnapshot(
        values={'foo': 'a', 'bar': ['a']},
        next=('node_b',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
        metadata={'source': 'loop', 'writes': {'node_a': {'foo': 'a', 'bar': ['a']}}, 'step': 1},
        created_at='2024-08-29T19:19:38.819946+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
        tasks=(PregelTask(id='6fb7314f-f114-5413-a1f3-d37dfe98ff44', name='node_b', error=None, interrupts=()),),
    ),
    StateSnapshot(
        values={'foo': '', 'bar': []},
        next=('node_a',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
        metadata={'source': 'loop', 'writes': None, 'step': 0},
        created_at='2024-08-29T19:19:38.817813+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
        tasks=(PregelTask(id='f1b14528-5ee5-579c-949b-23ef9bfbed58', name='node_a', error=None, interrupts=()),),
    ),
    StateSnapshot(
        values={'bar': []},
        next=('__start__',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
        metadata={'source': 'input', 'writes': {'foo': ''}, 'step': -1},
        created_at='2024-08-29T19:19:38.816205+00:00',
        parent_config=None,
        tasks=(PregelTask(id='6d27aa2e-d72b-5504-a36f-8620e54a76dd', name='__start__', error=None, interrupts=()),),
    )
]
```

:::

:::js
예제에서 `getStateHistory`의 출력은 다음과 같습니다:

```
[
  StateSnapshot {
    values: { foo: 'b', bar: ['a', 'b'] },
    next: [],
    config: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28fe-6528-8002-5a559208592c'
      }
    },
    metadata: {
      source: 'loop',
      writes: { nodeB: { foo: 'b', bar: ['b'] } },
      step: 2
    },
    createdAt: '2024-08-29T19:19:38.821749+00:00',
    parentConfig: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28f9-6ec4-8001-31981c2c39f8'
      }
    },
    tasks: []
  },
  StateSnapshot {
    values: { foo: 'a', bar: ['a'] },
    next: ['nodeB'],
    config: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28f9-6ec4-8001-31981c2c39f8'
      }
    },
    metadata: {
      source: 'loop',
      writes: { nodeA: { foo: 'a', bar: ['a'] } },
      step: 1
    },
    createdAt: '2024-08-29T19:19:38.819946+00:00',
    parentConfig: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28f4-6b4a-8000-ca575a13d36a'
      }
    },
    tasks: [
      PregelTask {
        id: '6fb7314f-f114-5413-a1f3-d37dfe98ff44',
        name: 'nodeB',
        error: null,
        interrupts: []
      }
    ]
  },
  StateSnapshot {
    values: { foo: '', bar: [] },
    next: ['node_a'],
    config: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28f4-6b4a-8000-ca575a13d36a'
      }
    },
    metadata: {
      source: 'loop',
      writes: null,
      step: 0
    },
    createdAt: '2024-08-29T19:19:38.817813+00:00',
    parentConfig: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28f0-6c66-bfff-6723431e8481'
      }
    },
    tasks: [
      PregelTask {
        id: 'f1b14528-5ee5-579c-949b-23ef9bfbed58',
        name: 'node_a',
        error: null,
        interrupts: []
      }
    ]
  },
  StateSnapshot {
    values: { bar: [] },
    next: ['__start__'],
    config: {
      configurable: {
        thread_id: '1',
        checkpoint_ns: '',
        checkpoint_id: '1ef663ba-28f0-6c66-bfff-6723431e8481'
      }
    },
    metadata: {
      source: 'input',
      writes: { foo: '' },
      step: -1
    },
    createdAt: '2024-08-29T19:19:38.816205+00:00',
    parentConfig: null,
    tasks: [
      PregelTask {
        id: '6d27aa2e-d72b-5504-a36f-8620e54a76dd',
        name: '__start__',
        error: null,
        interrupts: []
      }
    ]
  }
]
```

:::

![State](img/persistence/get_state.jpg)

### 재생 (Replay)

이전 그래프 실행을 재생하는 것도 가능합니다. `thread_id`와 `checkpoint_id`를 사용하여 그래프를 `invoke`하면, `checkpoint_id`에 해당하는 체크포인트 _이전_에 실행된 단계를 _재생_하고 체크포인트 _이후_의 단계만 실행합니다.

- `thread_id`는 thread의 ID입니다.
- `checkpoint_id`는 thread 내의 특정 체크포인트를 참조하는 식별자입니다.

그래프를 호출할 때 config의 `configurable` 부분에 이것들을 전달해야 합니다:

:::python

```python
config = {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}
graph.invoke(None, config=config)
```

:::

:::js

```typescript
const config = {
  configurable: {
    thread_id: "1",
    checkpoint_id: "0c62ca34-ac19-445d-bbb0-5b4984975b2a",
  },
};
await graph.invoke(null, config);
```

:::

중요한 점은 LangGraph가 특정 단계가 이전에 실행되었는지 여부를 알고 있다는 것입니다. 실행되었다면 LangGraph는 해당 단계를 단순히 _재생_하고 다시 실행하지 않지만, 이는 제공된 `checkpoint_id` _이전_의 단계에만 해당됩니다. `checkpoint_id` _이후_의 모든 단계는 이전에 실행되었더라도 실행됩니다(즉, 새로운 분기). 재생에 대한 자세한 내용은 [타임 트래블 가이드](../how-tos/human_in_the_loop/time-travel.md)를 참조하세요.

![Replay](img/persistence/re_play.png)

### 상태 업데이트

:::python

특정 `checkpoint`에서 그래프를 재생하는 것 외에도 그래프 상태를 _편집_할 수 있습니다. 이를 위해 `graph.update_state()`를 사용합니다. 이 메서드는 세 가지 다른 인수를 받습니다:

:::

:::js

특정 `checkpoint`에서 그래프를 재생하는 것 외에도 그래프 상태를 _편집_할 수 있습니다. 이를 위해 `graph.updateState()`를 사용합니다. 이 메서드는 세 가지 다른 인수를 받습니다:

:::

#### `config`

Config에는 업데이트할 thread를 지정하는 `thread_id`가 포함되어야 합니다. `thread_id`만 전달하면 현재 상태를 업데이트(또는 분기)합니다. 선택적으로 `checkpoint_id` 필드를 포함하면 선택한 체크포인트를 분기합니다.

#### `values`

상태를 업데이트하는 데 사용될 값입니다. 이 업데이트는 노드의 업데이트와 정확히 동일하게 처리됩니다. 즉, 그래프 상태의 일부 채널에 대해 정의된 경우 이러한 값이 [reducer](./low_level.md#reducers) 함수에 전달됩니다. 이는 `update_state`가 모든 채널의 채널 값을 자동으로 덮어쓰지 않고 reducer가 없는 채널에 대해서만 덮어쓴다는 것을 의미합니다. 예제를 살펴보겠습니다.

다음 스키마로 그래프의 상태를 정의했다고 가정해봅시다(위의 전체 예제 참조):

:::python

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
```

:::

:::js

```typescript
import { withLangGraph } from "@langchain/langgraph/zod";
import { z } from "zod";

const State = z.object({
  foo: z.number(),
  bar: withLangGraph(z.array(z.string()), {
    reducer: {
      fn: (x, y) => x.concat(y),
    },
    default: () => [],
  }),
});
```

:::

이제 그래프의 현재 상태가 다음과 같다고 가정해봅시다

:::python

```
{"foo": 1, "bar": ["a"]}
```

:::

:::js

```typescript
{ foo: 1, bar: ["a"] }
```

:::

다음과 같이 상태를 업데이트하면:

:::python

```python
graph.update_state(config, {"foo": 2, "bar": ["b"]})
```

:::

:::js

```typescript
await graph.updateState(config, { foo: 2, bar: ["b"] });
```

:::

그러면 그래프의 새로운 상태는 다음과 같습니다:

:::python

```
{"foo": 2, "bar": ["a", "b"]}
```

`foo` 키(채널)는 완전히 변경됩니다(해당 채널에 대해 지정된 reducer가 없으므로 `update_state`가 덮어씁니다). 그러나 `bar` 키에 대해서는 reducer가 지정되어 있으므로 `bar`의 상태에 `"b"`가 추가됩니다.
:::

:::js

```typescript
{ foo: 2, bar: ["a", "b"] }
```

`foo` 키(채널)는 완전히 변경됩니다(해당 채널에 대해 지정된 reducer가 없으므로 `updateState`가 덮어씁니다). 그러나 `bar` 키에 대해서는 reducer가 지정되어 있으므로 `bar`의 상태에 `"b"`가 추가됩니다.
:::

#### `as_node`

:::python
`update_state`를 호출할 때 선택적으로 지정할 수 있는 마지막 항목은 `as_node`입니다. 이를 제공하면 업데이트가 노드 `as_node`에서 온 것처럼 적용됩니다. `as_node`가 제공되지 않으면 명확한 경우 상태를 마지막으로 업데이트한 노드로 설정됩니다. 이것이 중요한 이유는 다음에 실행할 단계가 마지막으로 업데이트를 제공한 노드에 따라 달라지기 때문이며, 이를 사용하여 다음에 실행할 노드를 제어할 수 있습니다. 상태 분기에 대한 자세한 내용은 [타임 트래블 가이드](../how-tos/human_in_the_loop/time-travel.md)를 참조하세요.
:::

:::js
`updateState`를 호출할 때 선택적으로 지정할 수 있는 마지막 항목은 `asNode`입니다. 이를 제공하면 업데이트가 노드 `asNode`에서 온 것처럼 적용됩니다. `asNode`가 제공되지 않으면 명확한 경우 상태를 마지막으로 업데이트한 노드로 설정됩니다. 이것이 중요한 이유는 다음에 실행할 단계가 마지막으로 업데이트를 제공한 노드에 따라 달라지기 때문이며, 이를 사용하여 다음에 실행할 노드를 제어할 수 있습니다. 상태 분기에 대한 자세한 내용은 [타임 트래블 가이드](../how-tos/human_in_the_loop/time-travel.md)를 참조하세요.
:::

![Update](img/persistence/checkpoints_full_story.jpg)

## Memory Store

![Model of shared state](img/persistence/shared_state.png)

[상태 스키마](low_level.md#schema)는 그래프가 실행될 때 채워지는 키 집합을 지정합니다. 위에서 논의한 것처럼 상태는 각 그래프 단계에서 체크포인터에 의해 thread에 기록되어 상태 지속성을 가능하게 합니다.

그러나 _thread 간에_ 일부 정보를 유지하려면 어떻게 해야 할까요? 사용자와의 _모든_ 채팅 대화(예: thread)에 걸쳐 사용자에 대한 특정 정보를 유지하려는 챗봇의 경우를 생각해 보세요!

체크포인터만으로는 thread 간에 정보를 공유할 수 없습니다. 이것이 [`Store`](../reference/store.md#langgraph.store.base.BaseStore) 인터페이스가 필요한 이유입니다. 예를 들어 `InMemoryStore`를 정의하여 thread 간에 사용자에 대한 정보를 저장할 수 있습니다. 이전처럼 체크포인터와 함께 그래프를 컴파일하고, 새로운 `in_memory_store` 변수를 추가하기만 하면 됩니다.

!!! info "LangGraph API는 store를 자동으로 처리합니다"

    LangGraph API를 사용할 때는 store를 수동으로 구현하거나 구성할 필요가 없습니다. API가 백그라운드에서 모든 스토리지 인프라를 처리합니다.

### 기본 사용법

먼저 LangGraph를 사용하지 않고 독립적으로 이를 보여드리겠습니다.

:::python

```python
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()
```

:::

:::js

```typescript
import { MemoryStore } from "@langchain/langgraph";

const memoryStore = new MemoryStore();
```

:::

메모리는 `tuple`로 네임스페이스화되며, 이 특정 예제에서는 `(<user_id>, "memories")`가 됩니다. 네임스페이스는 임의의 길이가 될 수 있으며 무엇이든 나타낼 수 있고, 사용자별로 구분될 필요는 없습니다.

:::python

```python
user_id = "1"
namespace_for_memory = (user_id, "memories")
```

:::

:::js

```typescript
const userId = "1";
const namespaceForMemory = [userId, "memories"];
```

:::

store의 네임스페이스에 메모리를 저장하기 위해 `store.put` 메서드를 사용합니다. 이렇게 할 때 위에서 정의한 네임스페이스와 메모리의 키-값 쌍을 지정합니다: 키는 메모리의 고유 식별자(`memory_id`)이고 값(딕셔너리)은 메모리 자체입니다.

:::python

```python
memory_id = str(uuid.uuid4())
memory = {"food_preference" : "I like pizza"}
in_memory_store.put(namespace_for_memory, memory_id, memory)
```

:::

:::js

```typescript
import { v4 as uuidv4 } from "uuid";

const memoryId = uuidv4();
const memory = { food_preference: "I like pizza" };
await memoryStore.put(namespaceForMemory, memoryId, memory);
```

:::

네임스페이스의 메모리를 읽으려면 `store.search` 메서드를 사용하며, 이는 주어진 사용자의 모든 메모리를 리스트로 반환합니다. 가장 최근 메모리가 리스트의 마지막에 있습니다.

:::python

```python
memories = in_memory_store.search(namespace_for_memory)
memories[-1].dict()
{'value': {'food_preference': 'I like pizza'},
 'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
 'namespace': ['1', 'memories'],
 'created_at': '2024-10-02T17:22:31.590602+00:00',
 'updated_at': '2024-10-02T17:22:31.590605+00:00'}
```

Each memory type is a Python class ([`Item`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.Item)) with certain attributes. We can access it as a dictionary by converting via `.dict` as above.

The attributes it has are:

- `value`: The value (itself a dictionary) of this memory
- `key`: A unique key for this memory in this namespace
- `namespace`: A list of strings, the namespace of this memory type
- `created_at`: Timestamp for when this memory was created
- `updated_at`: Timestamp for when this memory was updated

:::

:::js

```typescript
const memories = await memoryStore.search(namespaceForMemory);
memories[memories.length - 1];

// {
//   value: { food_preference: 'I like pizza' },
//   key: '07e0caf4-1631-47b7-b15f-65515d4c1843',
//   namespace: ['1', 'memories'],
//   createdAt: '2024-10-02T17:22:31.590602+00:00',
//   updatedAt: '2024-10-02T17:22:31.590605+00:00'
// }
```

다음과 같은 속성을 가지고 있습니다:

- `value`: 이 메모리의 값
- `key`: 이 네임스페이스에서 이 메모리의 고유 키
- `namespace`: 이 메모리 타입의 네임스페이스인 문자열 리스트
- `createdAt`: 이 메모리가 생성된 시간의 타임스탬프
- `updatedAt`: 이 메모리가 업데이트된 시간의 타임스탬프

:::

### 시맨틱 검색

단순 조회 외에도 store는 시맨틱 검색을 지원하여 정확한 일치가 아닌 의미를 기반으로 메모리를 찾을 수 있습니다. 이를 활성화하려면 임베딩 모델로 store를 구성합니다:

:::python

```python
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),  # Embedding provider
        "dims": 1536,                              # Embedding dimensions
        "fields": ["food_preference", "$"]              # Fields to embed
    }
)
```

:::

:::js

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";

const store = new InMemoryStore({
  index: {
    embeddings: new OpenAIEmbeddings({ model: "text-embedding-3-small" }),
    dims: 1536,
    fields: ["food_preference", "$"], // Fields to embed
  },
});
```

:::

이제 검색할 때 자연어 쿼리를 사용하여 관련 메모리를 찾을 수 있습니다:

:::python

```python
# Find memories about food preferences
# (This can be done after putting memories into the store)
memories = store.search(
    namespace_for_memory,
    query="What does the user like to eat?",
    limit=3  # Return top 3 matches
)
```

:::

:::js

```typescript
// Find memories about food preferences
// (This can be done after putting memories into the store)
const memories = await store.search(namespaceForMemory, {
  query: "What does the user like to eat?",
  limit: 3, // Return top 3 matches
});
```

:::

메모리를 저장할 때 `fields` 매개변수를 구성하거나 `index` 매개변수를 지정하여 메모리의 어느 부분을 임베딩할지 제어할 수 있습니다:

:::python

```python
# Store with specific fields to embed
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {
        "food_preference": "I love Italian cuisine",
        "context": "Discussing dinner plans"
    },
    index=["food_preference"]  # Only embed "food_preferences" field
)

# Store without embedding (still retrievable, but not searchable)
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {"system_info": "Last updated: 2024-01-01"},
    index=False
)
```

:::

:::js

```typescript
// Store with specific fields to embed
await store.put(
  namespaceForMemory,
  uuidv4(),
  {
    food_preference: "I love Italian cuisine",
    context: "Discussing dinner plans",
  },
  { index: ["food_preference"] } // Only embed "food_preferences" field
);

// Store without embedding (still retrievable, but not searchable)
await store.put(
  namespaceForMemory,
  uuidv4(),
  { system_info: "Last updated: 2024-01-01" },
  { index: false }
);
```

:::

### LangGraph에서 사용하기

:::python
이제 모든 것이 준비되었으므로 LangGraph에서 `in_memory_store`를 사용합니다. `in_memory_store`는 체크포인터와 함께 작동합니다: 위에서 논의한 것처럼 체크포인터는 thread에 상태를 저장하고, `in_memory_store`는 thread _간에_ 액세스할 임의의 정보를 저장할 수 있게 합니다. 다음과 같이 체크포인터와 `in_memory_store` 모두로 그래프를 컴파일합니다.

```python
from langgraph.checkpoint.memory import InMemorySaver

# We need this because we want to enable threads (conversations)
checkpointer = InMemorySaver()

# ... Define the graph ...

# Compile the graph with the checkpointer and store
graph = graph.compile(checkpointer=checkpointer, store=in_memory_store)
```

:::

:::js
이제 모든 것이 준비되었으므로 LangGraph에서 `memoryStore`를 사용합니다. `memoryStore`는 체크포인터와 함께 작동합니다: 위에서 논의한 것처럼 체크포인터는 thread에 상태를 저장하고, `memoryStore`는 thread _간에_ 액세스할 임의의 정보를 저장할 수 있게 합니다. 다음과 같이 체크포인터와 `memoryStore` 모두로 그래프를 컴파일합니다.

```typescript
import { MemorySaver } from "@langchain/langgraph";

// We need this because we want to enable threads (conversations)
const checkpointer = new MemorySaver();

// ... Define the graph ...

// Compile the graph with the checkpointer and store
const graph = workflow.compile({ checkpointer, store: memoryStore });
```

:::

이전과 같이 `thread_id`와 함께 그래프를 호출하고, 위에서 보여준 것처럼 이 특정 사용자에게 메모리를 네임스페이스화하는 데 사용할 `user_id`도 함께 호출합니다.

:::python

```python
# Invoke the graph
user_id = "1"
config = {"configurable": {"thread_id": "1", "user_id": user_id}}

# First let's just say hi to the AI
for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi"}]}, config, stream_mode="updates"
):
    print(update)
```

:::

:::js

```typescript
// Invoke the graph
const userId = "1";
const config = { configurable: { thread_id: "1", user_id: userId } };

// First let's just say hi to the AI
for await (const update of await graph.stream(
  { messages: [{ role: "user", content: "hi" }] },
  { ...config, streamMode: "updates" }
)) {
  console.log(update);
}
```

:::

:::python
We can access the `in_memory_store` and the `user_id` in _any node_ by passing `store: BaseStore` and `config: RunnableConfig` as node arguments. Here's how we might use semantic search in a node to find relevant memories:

```python
def update_memory(state: MessagesState, config: RunnableConfig, *, store: BaseStore):

    # Get the user id from the config
    user_id = config["configurable"]["user_id"]

    # Namespace the memory
    namespace = (user_id, "memories")

    # ... Analyze conversation and create a new memory

    # Create a new memory ID
    memory_id = str(uuid.uuid4())

    # We create a new memory
    store.put(namespace, memory_id, {"memory": memory})

```

:::

:::js
We can access the `memoryStore` and the `user_id` in _any node_ by accessing `config` and `store` as node arguments. Here's how we might use semantic search in a node to find relevant memories:

```typescript
import {
  LangGraphRunnableConfig,
  BaseStore,
  MessagesZodState,
} from "@langchain/langgraph";
import { z } from "zod";

const updateMemory = async (
  state: z.infer<typeof MessagesZodState>,
  config: LangGraphRunnableConfig,
  store: BaseStore
) => {
  // Get the user id from the config
  const userId = config.configurable?.user_id;

  // Namespace the memory
  const namespace = [userId, "memories"];

  // ... Analyze conversation and create a new memory

  // Create a new memory ID
  const memoryId = uuidv4();

  // We create a new memory
  await store.put(namespace, memoryId, { memory });
};
```

:::

As we showed above, we can also access the store in any node and use the `store.search` method to get memories. Recall the memories are returned as a list of objects that can be converted to a dictionary.

:::python

```python
memories[-1].dict()
{'value': {'food_preference': 'I like pizza'},
 'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
 'namespace': ['1', 'memories'],
 'created_at': '2024-10-02T17:22:31.590602+00:00',
 'updated_at': '2024-10-02T17:22:31.590605+00:00'}
```

:::

:::js

```typescript
memories[memories.length - 1];
// {
//   value: { food_preference: 'I like pizza' },
//   key: '07e0caf4-1631-47b7-b15f-65515d4c1843',
//   namespace: ['1', 'memories'],
//   createdAt: '2024-10-02T17:22:31.590602+00:00',
//   updatedAt: '2024-10-02T17:22:31.590605+00:00'
// }
```

:::

We can access the memories and use them in our model call.

:::python

```python
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    # Get the user id from the config
    user_id = config["configurable"]["user_id"]

    # Namespace the memory
    namespace = (user_id, "memories")

    # Search based on the most recent message
    memories = store.search(
        namespace,
        query=state["messages"][-1].content,
        limit=3
    )
    info = "\n".join([d.value["memory"] for d in memories])

    # ... Use memories in the model call
```

:::

:::js

```typescript
const callModel = async (
  state: z.infer<typeof MessagesZodState>,
  config: LangGraphRunnableConfig,
  store: BaseStore
) => {
  // Get the user id from the config
  const userId = config.configurable?.user_id;

  // Namespace the memory
  const namespace = [userId, "memories"];

  // Search based on the most recent message
  const memories = await store.search(namespace, {
    query: state.messages[state.messages.length - 1].content,
    limit: 3,
  });
  const info = memories.map((d) => d.value.memory).join("\n");

  // ... Use memories in the model call
};
```

:::

If we create a new thread, we can still access the same memories so long as the `user_id` is the same.

:::python

```python
# Invoke the graph
config = {"configurable": {"thread_id": "2", "user_id": "1"}}

# Let's say hi again
for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi, tell me about my memories"}]}, config, stream_mode="updates"
):
    print(update)
```

:::

:::js

```typescript
// Invoke the graph
const config = { configurable: { thread_id: "2", user_id: "1" } };

// Let's say hi again
for await (const update of await graph.stream(
  { messages: [{ role: "user", content: "hi, tell me about my memories" }] },
  { ...config, streamMode: "updates" }
)) {
  console.log(update);
}
```

:::

When we use the LangGraph Platform, either locally (e.g., in LangGraph Studio) or with LangGraph Platform, the base store is available to use by default and does not need to be specified during graph compilation. To enable semantic search, however, you **do** need to configure the indexing settings in your `langgraph.json` file. For example:

```json
{
    ...
    "store": {
        "index": {
            "embed": "openai:text-embeddings-3-small",
            "dims": 1536,
            "fields": ["$"]
        }
    }
}
```

See the [deployment guide](../cloud/deployment/semantic_search.md) for more details and configuration options.

## Checkpointer libraries

Under the hood, checkpointing is powered by checkpointer objects that conform to @[BaseCheckpointSaver] interface. LangGraph provides several checkpointer implementations, all implemented via standalone, installable libraries:

:::python

- `langgraph-checkpoint`: The base interface for checkpointer savers (@[BaseCheckpointSaver]) and serialization/deserialization interface (@[SerializerProtocol][SerializerProtocol]). Includes in-memory checkpointer implementation (@[InMemorySaver][InMemorySaver]) for experimentation. LangGraph comes with `langgraph-checkpoint` included.
- `langgraph-checkpoint-sqlite`: An implementation of LangGraph checkpointer that uses SQLite database (@[SqliteSaver][SqliteSaver] / @[AsyncSqliteSaver]). Ideal for experimentation and local workflows. Needs to be installed separately.
- `langgraph-checkpoint-postgres`: An advanced checkpointer that uses Postgres database (@[PostgresSaver][PostgresSaver] / @[AsyncPostgresSaver]), used in LangGraph Platform. Ideal for using in production. Needs to be installed separately.

:::

:::js

- `@langchain/langgraph-checkpoint`: The base interface for checkpointer savers (@[BaseCheckpointSaver][BaseCheckpointSaver]) and serialization/deserialization interface (@[SerializerProtocol][SerializerProtocol]). Includes in-memory checkpointer implementation (@[MemorySaver]) for experimentation. LangGraph comes with `@langchain/langgraph-checkpoint` included.
- `@langchain/langgraph-checkpoint-sqlite`: An implementation of LangGraph checkpointer that uses SQLite database (@[SqliteSaver]). Ideal for experimentation and local workflows. Needs to be installed separately.
- `@langchain/langgraph-checkpoint-postgres`: An advanced checkpointer that uses Postgres database (@[PostgresSaver]), used in LangGraph Platform. Ideal for using in production. Needs to be installed separately.

:::

### Checkpointer interface

:::python
Each checkpointer conforms to @[BaseCheckpointSaver] interface and implements the following methods:

- `.put` - Store a checkpoint with its configuration and metadata.
- `.put_writes` - Store intermediate writes linked to a checkpoint (i.e. [pending writes](#pending-writes)).
- `.get_tuple` - Fetch a checkpoint tuple using for a given configuration (`thread_id` and `checkpoint_id`). This is used to populate `StateSnapshot` in `graph.get_state()`.
- `.list` - List checkpoints that match a given configuration and filter criteria. This is used to populate state history in `graph.get_state_history()`

If the checkpointer is used with asynchronous graph execution (i.e. executing the graph via `.ainvoke`, `.astream`, `.abatch`), asynchronous versions of the above methods will be used (`.aput`, `.aput_writes`, `.aget_tuple`, `.alist`).

!!! note

    For running your graph asynchronously, you can use `InMemorySaver`, or async versions of Sqlite/Postgres checkpointers -- `AsyncSqliteSaver` / `AsyncPostgresSaver` checkpointers.

:::

:::js
Each checkpointer conforms to the @[BaseCheckpointSaver][BaseCheckpointSaver] interface and implements the following methods:

- `.put` - Store a checkpoint with its configuration and metadata.
- `.putWrites` - Store intermediate writes linked to a checkpoint (i.e. [pending writes](#pending-writes)).
- `.getTuple` - Fetch a checkpoint tuple using for a given configuration (`thread_id` and `checkpoint_id`). This is used to populate `StateSnapshot` in `graph.getState()`.
- `.list` - List checkpoints that match a given configuration and filter criteria. This is used to populate state history in `graph.getStateHistory()`
  :::

### Serializer

When checkpointers save the graph state, they need to serialize the channel values in the state. This is done using serializer objects.

:::python
`langgraph_checkpoint` defines @[protocol][SerializerProtocol] for implementing serializers provides a default implementation (@[JsonPlusSerializer][JsonPlusSerializer]) that handles a wide variety of types, including LangChain and LangGraph primitives, datetimes, enums and more.

#### Serialization with `pickle`

The default serializer, @[`JsonPlusSerializer`][JsonPlusSerializer], uses ormsgpack and JSON under the hood, which is not suitable for all types of objects.

If you want to fallback to pickle for objects not currently supported by our msgpack encoder (such as Pandas dataframes),
you can use the `pickle_fallback` argument of the `JsonPlusSerializer`:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# ... Define the graph ...
graph.compile(
    checkpointer=InMemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))
)
```

#### Encryption

Checkpointers can optionally encrypt all persisted state. To enable this, pass an instance of @[`EncryptedSerializer`][EncryptedSerializer] to the `serde` argument of any `BaseCheckpointSaver` implementation. The easiest way to create an encrypted serializer is via @[`from_pycryptodome_aes`][from_pycryptodome_aes], which reads the AES key from the `LANGGRAPH_AES_KEY` environment variable (or accepts a `key` argument):

```python
import sqlite3

from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

serde = EncryptedSerializer.from_pycryptodome_aes()  # reads LANGGRAPH_AES_KEY
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db"), serde=serde)
```

```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.postgres import PostgresSaver

serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = PostgresSaver.from_conn_string("postgresql://...", serde=serde)
checkpointer.setup()
```

When running on LangGraph Platform, encryption is automatically enabled whenever `LANGGRAPH_AES_KEY` is present, so you only need to provide the environment variable. Other encryption schemes can be used by implementing @[`CipherProtocol`][CipherProtocol] and supplying it to `EncryptedSerializer`.
:::

:::js
`@langchain/langgraph-checkpoint` defines protocol for implementing serializers and provides a default implementation that handles a wide variety of types, including LangChain and LangGraph primitives, datetimes, enums and more.
:::

## 기능

### Human-in-the-loop

첫째, 체크포인터는 사람이 그래프 단계를 검사, 중단 및 승인할 수 있도록 하여 [Human-in-the-loop 워크플로](agentic_concepts.md#human-in-the-loop)를 촉진합니다. 사람이 언제든지 그래프의 상태를 볼 수 있어야 하고 사람이 상태를 업데이트한 후 그래프가 실행을 재개할 수 있어야 하므로 이러한 워크플로에는 체크포인터가 필요합니다. 예제는 [how-to 가이드](../how-tos/human_in_the_loop/add-human-in-the-loop.md)를 참조하세요.

### 메모리

둘째, 체크포인터는 상호작용 간에 ["메모리"](../concepts/memory.md)를 허용합니다. 반복되는 사람의 상호작용(대화 등)의 경우 모든 후속 메시지를 해당 thread로 보낼 수 있으며, thread는 이전 메시지의 메모리를 유지합니다. 체크포인터를 사용하여 대화 메모리를 추가하고 관리하는 방법에 대한 정보는 [메모리 추가](../how-tos/memory/add-memory.md)를 참조하세요.

### 타임 트래블

셋째, 체크포인터는 ["타임 트래블"](time-travel.md)을 허용하여 사용자가 이전 그래프 실행을 재생하여 특정 그래프 단계를 검토 및/또는 디버그할 수 있습니다. 또한 체크포인터를 사용하면 임의의 체크포인트에서 그래프 상태를 분기하여 대안적인 경로를 탐색할 수 있습니다.

### 장애 허용

마지막으로 체크포인팅은 장애 허용 및 오류 복구를 제공합니다: 주어진 superstep에서 하나 이상의 노드가 실패하면 마지막으로 성공한 단계에서 그래프를 다시 시작할 수 있습니다. 또한 주어진 superstep에서 그래프 노드가 실행 중 실패할 때 LangGraph는 해당 superstep에서 성공적으로 완료된 다른 노드의 보류 중인 체크포인트 쓰기를 저장하므로, 해당 superstep에서 그래프 실행을 재개할 때 성공한 노드를 다시 실행하지 않습니다.

#### 보류 중인 쓰기 {#pending-writes}

또한 주어진 superstep에서 그래프 노드가 실행 중 실패할 때 LangGraph는 해당 superstep에서 성공적으로 완료된 다른 노드의 보류 중인 체크포인트 쓰기를 저장하므로, 해당 superstep에서 그래프 실행을 재개할 때 성공한 노드를 다시 실행하지 않습니다.
