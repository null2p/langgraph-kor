---
search:
  boost: 2
---

# Functional API 개념

## 개요

**Functional API**를 사용하면 기존 코드를 최소한으로 변경하면서 LangGraph의 주요 기능인 [지속성](./persistence.md), [메모리](../how-tos/memory/add-memory.md), [human-in-the-loop](./human_in_the_loop.md), [스트리밍](./streaming.md)을 애플리케이션에 추가할 수 있습니다.

`if` 문, `for` 루프, 함수 호출과 같은 표준 언어 프리미티브를 사용하는 기존 코드에 이러한 기능을 통합하도록 설계되었습니다. 코드를 명시적인 파이프라인이나 DAG로 재구성해야 하는 많은 데이터 오케스트레이션 프레임워크와 달리, Functional API를 사용하면 엄격한 실행 모델을 강제하지 않고 이러한 기능을 통합할 수 있습니다.

Functional API는 두 가지 주요 빌딩 블록을 사용합니다:

:::python

- **`@entrypoint`** – 워크플로의 시작점으로 함수를 표시하여 로직을 캡슐화하고 장기 실행 작업 및 중단 처리를 포함한 실행 흐름을 관리합니다.
- **`@task`** – API 호출 또는 데이터 처리 단계와 같은 개별 작업 단위를 나타내며, entrypoint 내에서 비동기적으로 실행할 수 있습니다. Task는 대기하거나 동기적으로 해결할 수 있는 future와 유사한 객체를 반환합니다.
  :::

:::js

- **`entrypoint`** – entrypoint는 워크플로 로직을 캡슐화하고 장기 실행 작업 및 중단 처리를 포함한 실행 흐름을 관리합니다.
- **`task`** – API 호출 또는 데이터 처리 단계와 같은 개별 작업 단위를 나타내며, entrypoint 내에서 비동기적으로 실행할 수 있습니다. Task는 대기하거나 동기적으로 해결할 수 있는 future와 유사한 객체를 반환합니다.
  :::

이는 상태 관리 및 스트리밍을 사용하여 워크플로를 구축하기 위한 최소한의 추상화를 제공합니다.

!!! tip

    Functional API 사용 방법에 대한 정보는 [Functional API 사용](../how-tos/use-functional-api.md)을 참조하세요.

## Functional API vs. Graph API

더 선언적인 접근 방식을 선호하는 사용자를 위해 LangGraph의 [Graph API](./low_level.md)를 사용하면 그래프 패러다임을 사용하여 워크플로를 정의할 수 있습니다. 두 API는 동일한 기본 런타임을 공유하므로 동일한 애플리케이션에서 함께 사용할 수 있습니다.

주요 차이점은 다음과 같습니다:

- **제어 흐름**: Functional API는 그래프 구조에 대해 생각할 필요가 없습니다. 표준 Python 구조를 사용하여 워크플로를 정의할 수 있습니다. 이렇게 하면 일반적으로 작성해야 하는 코드의 양이 줄어듭니다.
- **단기 메모리**: **GraphAPI**는 [**State**](./low_level.md#state)를 선언해야 하며 그래프 상태에 대한 업데이트를 관리하기 위해 [**reducers**](./low_level.md#reducers)를 정의해야 할 수 있습니다. `@entrypoint`와 `@tasks`는 상태가 함수로 범위가 지정되고 함수 간에 공유되지 않으므로 명시적인 상태 관리가 필요하지 않습니다.
- **체크포인팅**: 두 API 모두 체크포인트를 생성하고 사용합니다. **Graph API**에서는 모든 [superstep](./low_level.md) 후에 새 체크포인트가 생성됩니다. **Functional API**에서는 task가 실행될 때 결과가 새 체크포인트를 생성하는 대신 주어진 entrypoint와 연결된 기존 체크포인트에 저장됩니다.
- **시각화**: Graph API를 사용하면 워크플로를 그래프로 시각화하기 쉬우며, 이는 디버깅, 워크플로 이해 및 다른 사람과 공유하는 데 유용할 수 있습니다. Functional API는 그래프가 런타임 중에 동적으로 생성되므로 시각화를 지원하지 않습니다.

## 예제

아래에서는 에세이를 작성하고 사람의 검토를 요청하기 위해 [중단](human_in_the_loop.md)하는 간단한 애플리케이션을 보여줍니다.

:::python

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import interrupt

@task
def write_essay(topic: str) -> str:
    """Write an essay about the given topic."""
    time.sleep(1) # A placeholder for a long-running task.
    return f"An essay about topic: {topic}"

@entrypoint(checkpointer=InMemorySaver())
def workflow(topic: str) -> dict:
    """A simple workflow that writes an essay and asks for a review."""
    essay = write_essay("cat").result()
    is_approved = interrupt({
        # Any json-serializable payload provided to interrupt as argument.
        # It will be surfaced on the client side as an Interrupt when streaming data
        # from the workflow.
        "essay": essay, # The essay we want reviewed.
        # We can add any additional information that we need.
        # For example, introduce a key called "action" with some instructions.
        "action": "Please approve/reject the essay",
    })

    return {
        "essay": essay, # The essay that was generated
        "is_approved": is_approved, # Response from HIL
    }
```

:::

:::js

```typescript
import { MemorySaver, entrypoint, task, interrupt } from "@langchain/langgraph";

const writeEssay = task("writeEssay", async (topic: string) => {
  // A placeholder for a long-running task.
  await new Promise((resolve) => setTimeout(resolve, 1000));
  return `An essay about topic: ${topic}`;
});

const workflow = entrypoint(
  { checkpointer: new MemorySaver(), name: "workflow" },
  async (topic: string) => {
    const essay = await writeEssay(topic);
    const isApproved = interrupt({
      // Any json-serializable payload provided to interrupt as argument.
      // It will be surfaced on the client side as an Interrupt when streaming data
      // from the workflow.
      essay, // The essay we want reviewed.
      // We can add any additional information that we need.
      // For example, introduce a key called "action" with some instructions.
      action: "Please approve/reject the essay",
    });

    return {
      essay, // The essay that was generated
      isApproved, // Response from HIL
    };
  }
);
```

:::

??? example "상세 설명"

    이 워크플로는 주제 "cat"에 대한 에세이를 작성한 다음 사람의 검토를 받기 위해 일시 중지됩니다. 워크플로는 검토가 제공될 때까지 무기한으로 중단될 수 있습니다.

    워크플로가 재개되면 처음부터 실행되지만, `writeEssay` task의 결과가 이미 저장되어 있으므로 task 결과는 다시 계산되지 않고 체크포인트에서 로드됩니다.

    :::python
    ```python
    import time
    import uuid
    from langgraph.func import entrypoint, task
    from langgraph.types import interrupt
    from langgraph.checkpoint.memory import InMemorySaver


    @task
    def write_essay(topic: str) -> str:
        """Write an essay about the given topic."""
        time.sleep(1)  # This is a placeholder for a long-running task.
        return f"An essay about topic: {topic}"

    @entrypoint(checkpointer=InMemorySaver())
    def workflow(topic: str) -> dict:
        """A simple workflow that writes an essay and asks for a review."""
        essay = write_essay("cat").result()
        is_approved = interrupt(
            {
                # Any json-serializable payload provided to interrupt as argument.
                # It will be surfaced on the client side as an Interrupt when streaming data
                # from the workflow.
                "essay": essay,  # The essay we want reviewed.
                # We can add any additional information that we need.
                # For example, introduce a key called "action" with some instructions.
                "action": "Please approve/reject the essay",
            }
        )
        return {
            "essay": essay,  # The essay that was generated
            "is_approved": is_approved,  # Response from HIL
        }


    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    for item in workflow.stream("cat", config):
        print(item)
    # > {'write_essay': 'An essay about topic: cat'}
    # > {
    # >     '__interrupt__': (
    # >        Interrupt(
    # >            value={
    # >                'essay': 'An essay about topic: cat',
    # >                'action': 'Please approve/reject the essay'
    # >            },
    # >            id='b9b2b9d788f482663ced6dc755c9e981'
    # >        ),
    # >    )
    # > }
    ```

    에세이가 작성되었고 검토 준비가 되었습니다. 검토가 제공되면 워크플로를 재개할 수 있습니다:

    ```python
    from langgraph.types import Command

    # Get review from a user (e.g., via a UI)
    # In this case, we're using a bool, but this can be any json-serializable value.
    human_review = True

    for item in workflow.stream(Command(resume=human_review), config):
        print(item)
    ```

    ```pycon
    {'workflow': {'essay': 'An essay about topic: cat', 'is_approved': False}}
    ```

    워크플로가 완료되었고 검토가 에세이에 추가되었습니다.
    :::

    :::js
    ```typescript
    import { v4 as uuidv4 } from "uuid";
    import { MemorySaver, entrypoint, task, interrupt } from "@langchain/langgraph";

    const writeEssay = task("writeEssay", async (topic: string) => {
      // This is a placeholder for a long-running task.
      await new Promise(resolve => setTimeout(resolve, 1000));
      return `An essay about topic: ${topic}`;
    });

    const workflow = entrypoint(
      { checkpointer: new MemorySaver(), name: "workflow" },
      async (topic: string) => {
        const essay = await writeEssay(topic);
        const isApproved = interrupt({
          // Any json-serializable payload provided to interrupt as argument.
          // It will be surfaced on the client side as an Interrupt when streaming data
          // from the workflow.
          essay, // The essay we want reviewed.
          // We can add any additional information that we need.
          // For example, introduce a key called "action" with some instructions.
          action: "Please approve/reject the essay",
        });

        return {
          essay, // The essay that was generated
          isApproved, // Response from HIL
        };
      }
    );

    const threadId = uuidv4();

    const config = {
      configurable: {
        thread_id: threadId
      }
    };

    for await (const item of workflow.stream("cat", config)) {
      console.log(item);
    }
    ```

    ```console
    { writeEssay: 'An essay about topic: cat' }
    {
      __interrupt__: [{
        value: { essay: 'An essay about topic: cat', action: 'Please approve/reject the essay' },
        resumable: true,
        ns: ['workflow:f7b8508b-21c0-8b4c-5958-4e8de74d2684'],
        when: 'during'
      }]
    }
    ```

    에세이가 작성되었고 검토 준비가 되었습니다. 검토가 제공되면 워크플로를 재개할 수 있습니다:

    ```typescript
    import { Command } from "@langchain/langgraph";

    // Get review from a user (e.g., via a UI)
    // In this case, we're using a bool, but this can be any json-serializable value.
    const humanReview = true;

    for await (const item of workflow.stream(new Command({ resume: humanReview }), config)) {
      console.log(item);
    }
    ```

    ```console
    { workflow: { essay: 'An essay about topic: cat', isApproved: true } }
    ```

    워크플로가 완료되었고 검토가 에세이에 추가되었습니다.
    :::

## Entrypoint

:::python
@[`@entrypoint`][entrypoint] 데코레이터를 사용하여 함수에서 워크플로를 만들 수 있습니다. 워크플로 로직을 캡슐화하고 _장기 실행 작업_ 및 [중단](./human_in_the_loop.md) 처리를 포함한 실행 흐름을 관리합니다.
:::

:::js
@[`entrypoint`][entrypoint] 함수를 사용하여 함수에서 워크플로를 만들 수 있습니다. 워크플로 로직을 캡슐화하고 _장기 실행 작업_ 및 [중단](./human_in_the_loop.md) 처리를 포함한 실행 흐름을 관리합니다.
:::

### 정의

:::python
**entrypoint**는 `@entrypoint` 데코레이터로 함수를 데코레이트하여 정의됩니다.

함수는 **단일 위치 인수를 받아야 하며**, 이는 워크플로 입력 역할을 합니다. 여러 데이터를 전달해야 하는 경우 첫 번째 인수의 입력 타입으로 딕셔너리를 사용하세요.

`entrypoint`로 함수를 데코레이트하면 @[`Pregel`][Pregel.stream] 인스턴스가 생성되어 워크플로 실행을 관리하는 데 도움이 됩니다(예: 스트리밍, 재개 및 체크포인팅 처리).

일반적으로 **checkpointer**를 `@entrypoint` 데코레이터에 전달하여 지속성을 활성화하고 **human-in-the-loop**과 같은 기능을 사용하고 싶을 것입니다.

=== "Sync"

    ```python
    from langgraph.func import entrypoint

    @entrypoint(checkpointer=checkpointer)
    def my_workflow(some_input: dict) -> int:
        # some logic that may involve long-running tasks like API calls,
        # and may be interrupted for human-in-the-loop.
        ...
        return result
    ```

=== "Async"

    ```python
    from langgraph.func import entrypoint

    @entrypoint(checkpointer=checkpointer)
    async def my_workflow(some_input: dict) -> int:
        # some logic that may involve long-running tasks like API calls,
        # and may be interrupted for human-in-the-loop
        ...
        return result
    ```

:::

:::js
**entrypoint**는 구성과 함수를 사용하여 `entrypoint` 함수를 호출하여 정의됩니다.

함수는 **단일 위치 인수를 받아야 하며**, 이는 워크플로 입력 역할을 합니다. 여러 데이터를 전달해야 하는 경우 첫 번째 인수의 입력 타입으로 객체를 사용하세요.

함수로 entrypoint를 생성하면 워크플로 실행을 관리하는 데 도움이 되는 워크플로 인스턴스가 생성됩니다(예: 스트리밍, 재개 및 체크포인팅 처리).

**human-in-the-loop**과 같은 기능을 사용하고 지속성을 활성화하려면 종종 **checkpointer**를 `entrypoint` 함수에 전달하고 싶을 것입니다.

```typescript
import { entrypoint } from "@langchain/langgraph";

const myWorkflow = entrypoint(
  { checkpointer, name: "workflow" },
  async (someInput: Record<string, any>): Promise<number> => {
    // some logic that may involve long-running tasks like API calls,
    // and may be interrupted for human-in-the-loop
    return result;
  }
);
```

:::

!!! important "직렬화"

    entrypoint의 **입력**과 **출력**은 체크포인팅을 지원하기 위해 JSON 직렬화 가능해야 합니다. 자세한 내용은 [직렬화](#serialization) 섹션을 참조하세요.

:::python

### 주입 가능한 파라미터

`entrypoint`를 선언할 때 런타임에 자동으로 주입될 추가 파라미터에 대한 액세스를 요청할 수 있습니다. 이러한 파라미터에는 다음이 포함됩니다:

| 파라미터 | 설명 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **previous** | 주어진 thread에 대한 이전 `checkpoint`와 연결된 상태에 액세스합니다. [단기 메모리](#short-term-memory)를 참조하세요. |
| **store** | [BaseStore][langgraph.store.base.BaseStore]의 인스턴스입니다. [장기 메모리](../how-tos/use-functional-api.md#long-term-memory)에 유용합니다. |
| **writer** | Async Python < 3.11에서 작업할 때 StreamWriter에 액세스하는 데 사용합니다. 자세한 내용은 [Functional API를 사용한 스트리밍](../how-tos/use-functional-api.md#streaming)을 참조하세요. |
| **config** | 런타임 구성에 액세스하는 데 사용합니다. 정보는 [RunnableConfig](https://python.langchain.com/docs/concepts/runnables/#runnableconfig)를 참조하세요. |

!!! important

    적절한 이름과 타입 주석으로 파라미터를 선언하세요.

??? example "주입 가능한 파라미터 요청"

    ```python
    from langchain_core.runnables import RunnableConfig
    from langgraph.func import entrypoint
    from langgraph.store.base import BaseStore
    from langgraph.store.memory import InMemoryStore

    in_memory_store = InMemoryStore(...)  # An instance of InMemoryStore for long-term memory

    @entrypoint(
        checkpointer=checkpointer,  # Specify the checkpointer
        store=in_memory_store  # Specify the store
    )
    def my_workflow(
        some_input: dict,  # The input (e.g., passed via `invoke`)
        *,
        previous: Any = None, # For short-term memory
        store: BaseStore,  # For long-term memory
        writer: StreamWriter,  # For streaming custom data
        config: RunnableConfig  # For accessing the configuration passed to the entrypoint
    ) -> ...:
    ```

:::

### 실행

:::python
[`@entrypoint`](#entrypoint)를 사용하면 `invoke`, `ainvoke`, `stream`, `astream` 메서드를 사용하여 실행할 수 있는 @[`Pregel`][Pregel.stream] 객체가 생성됩니다.

=== "Invoke"

    ```python
    config = {
        "configurable": {
            "thread_id": "some_thread_id"
        }
    }
    my_workflow.invoke(some_input, config)  # Wait for the result synchronously
    ```

=== "Async Invoke"

    ```python
    config = {
        "configurable": {
            "thread_id": "some_thread_id"
        }
    }
    await my_workflow.ainvoke(some_input, config)  # Await result asynchronously
    ```

=== "Stream"

    ```python
    config = {
        "configurable": {
            "thread_id": "some_thread_id"
        }
    }

    for chunk in my_workflow.stream(some_input, config):
        print(chunk)
    ```

=== "Async Stream"

    ```python
    config = {
        "configurable": {
            "thread_id": "some_thread_id"
        }
    }

    async for chunk in my_workflow.astream(some_input, config):
        print(chunk)
    ```

:::

:::js
[`entrypoint`](#entrypoint) 함수를 사용하면 `invoke` 및 `stream` 메서드를 사용하여 실행할 수 있는 객체가 반환됩니다.

=== "Invoke"

    ```typescript
    const config = {
      configurable: {
        thread_id: "some_thread_id"
      }
    };
    await myWorkflow.invoke(someInput, config); // Wait for the result
    ```

=== "Stream"

    ```typescript
    const config = {
      configurable: {
        thread_id: "some_thread_id"
      }
    };

    for await (const chunk of myWorkflow.stream(someInput, config)) {
      console.log(chunk);
    }
    ```

:::

### 재개

:::python
@[interrupt][interrupt] 후 실행을 재개하는 것은 @[Command] 프리미티브에 **resume** 값을 전달하여 수행할 수 있습니다.

=== "Invoke"

    ```python
    from langgraph.types import Command

    config = {
        "configurable": {
            "thread_id": "some_thread_id"
        }
    }

    my_workflow.invoke(Command(resume=some_resume_value), config)
    ```

=== "Async Invoke"

    ```python
    from langgraph.types import Command

    config = {
        "configurable": {
            "thread_id": "some_thread_id"
        }
    }

    await my_workflow.ainvoke(Command(resume=some_resume_value), config)
    ```

=== "Stream"

    ```python
    from langgraph.types import Command

    config = {
        "configurable": {
            "thread_id": "some_thread_id"
        }
    }

    for chunk in my_workflow.stream(Command(resume=some_resume_value), config):
        print(chunk)
    ```

=== "Async Stream"

    ```python
    from langgraph.types import Command

    config = {
        "configurable": {
            "thread_id": "some_thread_id"
        }
    }

    async for chunk in my_workflow.astream(Command(resume=some_resume_value), config):
        print(chunk)
    ```

:::

:::js
@[interrupt][interrupt] 후 실행을 재개하는 것은 @[`Command`][Command] 프리미티브에 **resume** 값을 전달하여 수행할 수 있습니다.

=== "Invoke"

    ```typescript
    import { Command } from "@langchain/langgraph";

    const config = {
      configurable: {
        thread_id: "some_thread_id"
      }
    };

    await myWorkflow.invoke(new Command({ resume: someResumeValue }), config);
    ```

=== "Stream"

    ```typescript
    import { Command } from "@langchain/langgraph";

    const config = {
      configurable: {
        thread_id: "some_thread_id"
      }
    };

    const stream = await myWorkflow.stream(
      new Command({ resume: someResumableValue }),
      config,
    )

    for await (const chunk of stream) {
      console.log(chunk);
    }
    ```

:::

:::python

**오류 후 재개**

오류 후 재개하려면 `None`과 동일한 **thread id** (config)로 `entrypoint`를 실행하세요.

이는 기본 **오류**가 해결되었고 실행이 성공적으로 진행될 수 있다고 가정합니다.

=== "Invoke"

    ```python

    config = {
        "configurable": {
            "thread_id": "some_thread_id"
        }
    }

    my_workflow.invoke(None, config)
    ```

=== "Async Invoke"

    ```python

    config = {
        "configurable": {
            "thread_id": "some_thread_id"
        }
    }

    await my_workflow.ainvoke(None, config)
    ```

=== "Stream"

    ```python

    config = {
        "configurable": {
            "thread_id": "some_thread_id"
        }
    }

    for chunk in my_workflow.stream(None, config):
        print(chunk)
    ```

=== "Async Stream"

    ```python

    config = {
        "configurable": {
            "thread_id": "some_thread_id"
        }
    }

    async for chunk in my_workflow.astream(None, config):
        print(chunk)
    ```

:::

:::js

**오류 후 재개**

오류 후 재개하려면 `null`과 동일한 **thread id** (config)로 `entrypoint`를 실행하세요.

이는 기본 **오류**가 해결되었고 실행이 성공적으로 진행될 수 있다고 가정합니다.

=== "Invoke"

    ```typescript
    const config = {
      configurable: {
        thread_id: "some_thread_id"
      }
    };

    await myWorkflow.invoke(null, config);
    ```

=== "Stream"

    ```typescript
    const config = {
      configurable: {
        thread_id: "some_thread_id"
      }
    };

    for await (const chunk of myWorkflow.stream(null, config)) {
      console.log(chunk);
    }
    ```

:::

### 단기 메모리

`entrypoint`가 `checkpointer`로 정의되면 [체크포인트](persistence.md#checkpoints)에서 동일한 **thread id**에 대한 연속 호출 간에 정보를 저장합니다.

:::python
이를 통해 `previous` 파라미터를 사용하여 이전 호출의 상태에 액세스할 수 있습니다.

기본적으로 `previous` 파라미터는 이전 호출의 반환 값입니다.

```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(number: int, *, previous: Any = None) -> int:
    previous = previous or 0
    return number + previous

config = {
    "configurable": {
        "thread_id": "some_thread_id"
    }
}

my_workflow.invoke(1, config)  # 1 (previous was None)
my_workflow.invoke(2, config)  # 3 (previous was 1 from the previous invocation)
```

:::

:::js
이를 통해 `getPreviousState` 함수를 사용하여 이전 호출의 상태에 액세스할 수 있습니다.

기본적으로 `getPreviousState` 함수는 이전 호출의 반환 값을 반환합니다.

```typescript
import { entrypoint, getPreviousState } from "@langchain/langgraph";

const myWorkflow = entrypoint(
  { checkpointer, name: "workflow" },
  async (number: number) => {
    const previous = getPreviousState<number>() ?? 0;
    return number + previous;
  }
);

const config = {
  configurable: {
    thread_id: "some_thread_id",
  },
};

await myWorkflow.invoke(1, config); // 1 (previous was undefined)
await myWorkflow.invoke(2, config); // 3 (previous was 1 from the previous invocation)
```

:::

#### `entrypoint.final`

:::python
@[`entrypoint.final`][entrypoint.final]은 entrypoint에서 반환할 수 있는 특수 프리미티브로, **체크포인트에 저장되는 값**과 **entrypoint의 반환 값**을 **분리**할 수 있습니다.

첫 번째 값은 entrypoint의 반환 값이고, 두 번째 값은 체크포인트에 저장될 값입니다. 타입 주석은 `entrypoint.final[return_type, save_type]`입니다.

```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(number: int, *, previous: Any = None) -> entrypoint.final[int, int]:
    previous = previous or 0
    # This will return the previous value to the caller, saving
    # 2 * number to the checkpoint, which will be used in the next invocation
    # for the `previous` parameter.
    return entrypoint.final(value=previous, save=2 * number)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

my_workflow.invoke(3, config)  # 0 (previous was None)
my_workflow.invoke(1, config)  # 6 (previous was 3 * 2 from the previous invocation)
```

:::

:::js
@[`entrypoint.final`][entrypoint.final]은 entrypoint에서 반환할 수 있는 특수 프리미티브로, **체크포인트에 저장되는 값**과 **entrypoint의 반환 값**을 **분리**할 수 있습니다.

첫 번째 값은 entrypoint의 반환 값이고, 두 번째 값은 체크포인트에 저장될 값입니다.

```typescript
import { entrypoint, getPreviousState } from "@langchain/langgraph";

const myWorkflow = entrypoint(
  { checkpointer, name: "workflow" },
  async (number: number) => {
    const previous = getPreviousState<number>() ?? 0;
    // This will return the previous value to the caller, saving
    // 2 * number to the checkpoint, which will be used in the next invocation
    // for the `previous` parameter.
    return entrypoint.final({
      value: previous,
      save: 2 * number,
    });
  }
);

const config = {
  configurable: {
    thread_id: "1",
  },
};

await myWorkflow.invoke(3, config); // 0 (previous was undefined)
await myWorkflow.invoke(1, config); // 6 (previous was 3 * 2 from the previous invocation)
```

:::

## Task

**task**는 API 호출 또는 데이터 처리 단계와 같은 개별 작업 단위를 나타냅니다. 두 가지 주요 특성이 있습니다:

- **비동기 실행**: Task는 비동기적으로 실행되도록 설계되어 여러 작업이 차단 없이 동시에 실행될 수 있습니다.
- **체크포인팅**: Task 결과는 체크포인트에 저장되어 마지막으로 저장된 상태에서 워크플로를 재개할 수 있습니다. (자세한 내용은 [지속성](persistence.md)을 참조하세요).

### 정의

:::python
Task는 일반 Python 함수를 래핑하는 `@task` 데코레이터를 사용하여 정의됩니다.

```python
from langgraph.func import task

@task()
def slow_computation(input_value):
    # Simulate a long-running operation
    ...
    return result
```

:::

:::js
Task는 일반 함수를 래핑하는 `task` 함수를 사용하여 정의됩니다.

```typescript
import { task } from "@langchain/langgraph";

const slowComputation = task("slowComputation", async (inputValue: any) => {
  // Simulate a long-running operation
  return result;
});
```

:::

!!! important "직렬화"

    task의 **출력**은 체크포인팅을 지원하기 위해 JSON 직렬화 가능해야 합니다.

### 실행

**Task**는 **entrypoint**, 다른 **task** 또는 [state graph node](./low_level.md#nodes) 내에서만 호출할 수 있습니다.

Task는 메인 애플리케이션 코드에서 직접 호출할 수 _없습니다_.

:::python
**task**를 호출하면 future 객체와 함께 _즉시_ 반환됩니다. future는 나중에 사용 가능한 결과에 대한 자리 표시자입니다.

**task**의 결과를 얻으려면 동기적으로 대기(`result()` 사용)하거나 비동기적으로 대기(`await` 사용)할 수 있습니다.

=== "Synchronous Invocation"

    ```python
    @entrypoint(checkpointer=checkpointer)
    def my_workflow(some_input: int) -> int:
        future = slow_computation(some_input)
        return future.result()  # Wait for the result synchronously
    ```

=== "Asynchronous Invocation"

    ```python
    @entrypoint(checkpointer=checkpointer)
    async def my_workflow(some_input: int) -> int:
        return await slow_computation(some_input)  # Await result asynchronously
    ```

:::

:::js
**task**를 호출하면 대기할 수 있는 Promise가 반환됩니다.

```typescript
const myWorkflow = entrypoint(
  { checkpointer, name: "workflow" },
  async (someInput: number): Promise<number> => {
    return await slowComputation(someInput);
  }
);
```

:::

## Task를 사용해야 하는 경우

**Task**는 다음 시나리오에서 유용합니다:

- **체크포인팅**: 워크플로를 재개할 때 다시 계산할 필요가 없도록 장기 실행 작업의 결과를 체크포인트에 저장해야 할 때.
- **Human-in-the-loop**: 사람의 개입이 필요한 워크플로를 구축하는 경우, 워크플로가 올바르게 재개될 수 있도록 임의성(예: API 호출)을 캡슐화하기 위해 **task**를 사용해야 합니다. 자세한 내용은 [결정론](#determinism) 섹션을 참조하세요.
- **병렬 실행**: I/O 바운드 작업의 경우 **task**는 차단 없이 여러 작업을 동시에 실행할 수 있는 병렬 실행을 가능하게 합니다(예: 여러 API 호출).
- **관찰 가능성**: **task**에서 작업을 래핑하면 워크플로의 진행 상황을 추적하고 [LangSmith](https://docs.smith.langchain.com/)를 사용하여 개별 작업의 실행을 모니터링할 수 있습니다.
- **재시도 가능한 작업**: 실패 또는 불일치를 처리하기 위해 작업을 재시도해야 할 때 **task**는 재시도 로직을 캡슐화하고 관리하는 방법을 제공합니다.

## 직렬화

LangGraph의 직렬화에는 두 가지 주요 측면이 있습니다:

1. `entrypoint` 입력과 출력은 JSON 직렬화 가능해야 합니다.
2. `task` 출력은 JSON 직렬화 가능해야 합니다.

:::python
이러한 요구 사항은 체크포인팅 및 워크플로 재개를 활성화하는 데 필요합니다. 입력과 출력이 직렬화 가능한지 확인하려면 딕셔너리, 리스트, 문자열, 숫자, 불린과 같은 Python 프리미티브를 사용하세요.
:::

:::js
이러한 요구 사항은 체크포인팅 및 워크플로 재개를 활성화하는 데 필요합니다. 입력과 출력이 직렬화 가능한지 확인하려면 객체, 배열, 문자열, 숫자, 불린과 같은 프리미티브를 사용하세요.
:::

직렬화는 task 결과 및 중간 값과 같은 워크플로 상태를 안정적으로 저장하고 복원할 수 있도록 합니다. 이는 human-in-the-loop 상호작용, 내결함성 및 병렬 실행을 활성화하는 데 중요합니다.

직렬화할 수 없는 입력 또는 출력을 제공하면 워크플로가 checkpointer로 구성될 때 런타임 오류가 발생합니다.

## 결정론

**human-in-the-loop**과 같은 기능을 활용하려면 임의성을 **task** 내부에 캡슐화해야 합니다. 이렇게 하면 실행이 중지되고(예: human-in-the-loop을 위해) 재개될 때 **task** 결과가 비결정론적이더라도 동일한 _단계 시퀀스_를 따르는 것이 보장됩니다.

LangGraph는 실행 중에 **task** 및 [**subgraph**](./subgraphs.md) 결과를 유지하여 이 동작을 달성합니다. 잘 설계된 워크플로는 실행 재개 시 _동일한 단계 시퀀스_를 따르도록 하여 이전에 계산된 결과를 다시 실행하지 않고 올바르게 검색할 수 있습니다. 이는 장기 실행 **task** 또는 비결정론적 결과가 있는 **task**에 특히 유용하며, 이전에 수행한 작업을 반복하지 않고 본질적으로 동일한 위치에서 재개할 수 있습니다.

워크플로의 다른 실행은 다른 결과를 생성할 수 있지만, **특정** 실행을 재개하면 항상 동일한 기록된 단계 시퀀스를 따라야 합니다. 이를 통해 LangGraph는 그래프가 중단되기 전에 실행된 **task** 및 **subgraph** 결과를 효율적으로 조회하고 다시 계산하지 않을 수 있습니다.

## 멱등성

멱등성은 동일한 작업을 여러 번 실행해도 동일한 결과가 생성되도록 합니다. 이는 실패로 인해 단계가 다시 실행되는 경우 중복 API 호출 및 중복 처리를 방지하는 데 도움이 됩니다. 체크포인팅을 위해 항상 **task** 함수 내부에 API 호출을 배치하고, 재실행의 경우를 대비하여 멱등성을 갖도록 설계하세요. **task**가 시작되었지만 성공적으로 완료되지 않은 경우 재실행이 발생할 수 있습니다. 그런 다음 워크플로가 재개되면 **task**가 다시 실행됩니다. 중복을 방지하려면 멱등성 키를 사용하거나 기존 결과를 확인하세요.

## 일반적인 함정

### 부작용 처리

부작용(예: 파일에 쓰기, 이메일 보내기)을 task에 캡슐화하여 워크플로를 재개할 때 여러 번 실행되지 않도록 하세요.

=== "Incorrect"

    이 예제에서는 부작용(파일에 쓰기)이 워크플로에 직접 포함되어 있으므로 워크플로를 재개할 때 두 번째로 실행됩니다.

    :::python
    ```python
    @entrypoint(checkpointer=checkpointer)
    def my_workflow(inputs: dict) -> int:
        # This code will be executed a second time when resuming the workflow.
        # Which is likely not what you want.
        # highlight-next-line
        with open("output.txt", "w") as f:
            # highlight-next-line
            f.write("Side effect executed")
        value = interrupt("question")
        return value
    ```
    :::

    :::js
    ```typescript
    import { entrypoint, interrupt } from "@langchain/langgraph";
    import fs from "fs";

    const myWorkflow = entrypoint(
      { checkpointer, name: "workflow },
      async (inputs: Record<string, any>) => {
        // This code will be executed a second time when resuming the workflow.
        // Which is likely not what you want.
        fs.writeFileSync("output.txt", "Side effect executed");
        const value = interrupt("question");
        return value;
      }
    );
    ```
    :::

=== "Correct"

    이 예제에서는 부작용이 task에 캡슐화되어 재개 시 일관된 실행을 보장합니다.

    :::python
    ```python
    from langgraph.func import task

    # highlight-next-line
    @task
    # highlight-next-line
    def write_to_file():
        with open("output.txt", "w") as f:
            f.write("Side effect executed")

    @entrypoint(checkpointer=checkpointer)
    def my_workflow(inputs: dict) -> int:
        # The side effect is now encapsulated in a task.
        write_to_file().result()
        value = interrupt("question")
        return value
    ```
    :::

    :::js
    ```typescript
    import { entrypoint, task, interrupt } from "@langchain/langgraph";
    import * as fs from "fs";

    const writeToFile = task("writeToFile", async () => {
      fs.writeFileSync("output.txt", "Side effect executed");
    });

    const myWorkflow = entrypoint(
      { checkpointer, name: "workflow" },
      async (inputs: Record<string, any>) => {
        // The side effect is now encapsulated in a task.
        await writeToFile();
        const value = interrupt("question");
        return value;
      }
    );
    ```
    :::

### 비결정론적 제어 흐름

매번 다른 결과를 제공할 수 있는 작업(현재 시간 가져오기 또는 난수 생성과 같은)은 재개 시 동일한 결과가 반환되도록 task에 캡슐화해야 합니다.

- task에서: 난수 가져오기 (5) → 중단 → 재개 → (5를 다시 반환) → ...
- task가 아닌 경우: 난수 가져오기 (5) → 중단 → 재개 → 새 난수 가져오기 (7) → ...

:::python
이는 여러 interrupt 호출이 있는 **human-in-the-loop** 워크플로를 사용할 때 특히 중요합니다. LangGraph는 각 task/entrypoint에 대한 resume 값 목록을 유지합니다. interrupt가 발생하면 해당 resume 값과 일치합니다. 이 일치는 엄격하게 **인덱스 기반**이므로 resume 값의 순서는 interrupt의 순서와 일치해야 합니다.
:::

:::js
이는 여러 interrupt 호출이 있는 **human-in-the-loop** 워크플로를 사용할 때 특히 중요합니다. LangGraph는 각 task/entrypoint에 대한 resume 값 목록을 유지합니다. interrupt가 발생하면 해당 resume 값과 일치합니다. 이 일치는 엄격하게 **인덱스 기반**이므로 resume 값의 순서는 interrupt의 순서와 일치해야 합니다.
:::

재개 시 실행 순서가 유지되지 않으면 하나의 `interrupt` 호출이 잘못된 `resume` 값과 일치하여 잘못된 결과가 발생할 수 있습니다.

자세한 내용은 [결정론](#determinism) 섹션을 읽어보세요.

=== "Incorrect"

    이 예제에서 워크플로는 현재 시간을 사용하여 실행할 task를 결정합니다. 워크플로의 결과가 실행 시간에 따라 달라지므로 이는 비결정론적입니다.

    :::python
    ```python
    from langgraph.func import entrypoint

    @entrypoint(checkpointer=checkpointer)
    def my_workflow(inputs: dict) -> int:
        t0 = inputs["t0"]
        # highlight-next-line
        t1 = time.time()

        delta_t = t1 - t0

        if delta_t > 1:
            result = slow_task(1).result()
            value = interrupt("question")
        else:
            result = slow_task(2).result()
            value = interrupt("question")

        return {
            "result": result,
            "value": value
        }
    ```
    :::

    :::js
    ```typescript
    import { entrypoint, interrupt } from "@langchain/langgraph";

    const myWorkflow = entrypoint(
      { checkpointer, name: "workflow" },
      async (inputs: { t0: number }) => {
        const t1 = Date.now();

        const deltaT = t1 - inputs.t0;

        if (deltaT > 1000) {
          const result = await slowTask(1);
          const value = interrupt("question");
          return { result, value };
        } else {
          const result = await slowTask(2);
          const value = interrupt("question");
          return { result, value };
        }
      }
    );
    ```
    :::

=== "Correct"

    :::python
    이 예제에서 워크플로는 입력 `t0`를 사용하여 실행할 task를 결정합니다. 워크플로의 결과가 입력에만 의존하므로 이는 결정론적입니다.

    ```python
    import time

    from langgraph.func import task

    # highlight-next-line
    @task
    # highlight-next-line
    def get_time() -> float:
        return time.time()

    @entrypoint(checkpointer=checkpointer)
    def my_workflow(inputs: dict) -> int:
        t0 = inputs["t0"]
        # highlight-next-line
        t1 = get_time().result()

        delta_t = t1 - t0

        if delta_t > 1:
            result = slow_task(1).result()
            value = interrupt("question")
        else:
            result = slow_task(2).result()
            value = interrupt("question")

        return {
            "result": result,
            "value": value
        }
    ```
    :::

    :::js
    이 예제에서 워크플로는 입력 `t0`를 사용하여 실행할 task를 결정합니다. 워크플로의 결과가 입력에만 의존하므로 이는 결정론적입니다.

    ```typescript
    import { entrypoint, task, interrupt } from "@langchain/langgraph";

    const getTime = task("getTime", () => Date.now());

    const myWorkflow = entrypoint(
      { checkpointer, name: "workflow" },
      async (inputs: { t0: number }): Promise<any> => {
        const t1 = await getTime();

        const deltaT = t1 - inputs.t0;

        if (deltaT > 1000) {
          const result = await slowTask(1);
          const value = interrupt("question");
          return { result, value };
        } else {
          const result = await slowTask(2);
          const value = interrupt("question");
          return { result, value };
        }
      }
    );
    ```
    :::
