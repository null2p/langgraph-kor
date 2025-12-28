---
search:
  boost: 2
---

# 메모리

[메모리](../how-tos/memory/add-memory.md)는 이전 상호작용에 대한 정보를 기억하는 시스템입니다. AI 에이전트에게 메모리는 이전 상호작용을 기억하고, 피드백으로부터 학습하며, 사용자 선호도에 적응할 수 있게 해주기 때문에 매우 중요합니다. 에이전트가 수많은 사용자 상호작용이 있는 더 복잡한 작업을 처리할수록, 이 기능은 효율성과 사용자 만족도 모두에 필수적이 됩니다.

이 개념 가이드는 회상 범위를 기반으로 두 가지 유형의 메모리를 다룹니다:

- [단기 메모리](#short-term-memory) 또는 [스레드](persistence.md#threads) 범위 메모리는 세션 내에서 메시지 히스토리를 유지하여 진행 중인 대화를 추적합니다. LangGraph는 단기 메모리를 에이전트의 [상태](low_level.md#state)의 일부로 관리합니다. 상태는 [체크포인터](persistence.md#checkpoints)를 사용하여 데이터베이스에 영속화되므로 스레드는 언제든지 재개될 수 있습니다. 단기 메모리는 그래프가 호출되거나 단계가 완료될 때 업데이트되며, 상태는 각 단계의 시작 시 읽힙니다.

- [장기 메모리](#long-term-memory)는 세션 간에 사용자별 또는 애플리케이션 레벨 데이터를 저장하며 대화 스레드 _전체에서_ 공유됩니다. _언제든지_ 그리고 _모든 스레드에서_ 회상될 수 있습니다. 메모리는 단일 스레드 ID 내에서만이 아니라 모든 커스텀 네임스페이스로 범위가 지정됩니다. LangGraph는 장기 메모리를 저장하고 회상할 수 있도록 [스토어](persistence.md#memory-store) ([레퍼런스 문서](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore))를 제공합니다.

![](img/memory/short-vs-long.png)


## 단기 메모리 {#short-term-memory}

[단기 메모리](../how-tos/memory/add-memory.md#add-short-term-memory)는 애플리케이션이 단일 [스레드](persistence.md#threads) 또는 대화 내에서 이전 상호작용을 기억할 수 있게 합니다. [스레드](persistence.md#threads)는 이메일이 단일 대화에서 메시지를 그룹화하는 방식과 유사하게 세션 내에서 여러 상호작용을 구성합니다.

LangGraph는 단기 메모리를 에이전트 상태의 일부로 관리하며, 스레드 범위 체크포인트를 통해 영속화합니다. 이 상태는 일반적으로 대화 히스토리와 함께 업로드된 파일, 검색된 문서 또는 생성된 아티팩트와 같은 기타 상태 데이터를 포함할 수 있습니다. 이를 그래프의 상태에 저장함으로써, 봇은 서로 다른 스레드 간의 분리를 유지하면서 주어진 대화에 대한 전체 컨텍스트에 액세스할 수 있습니다.

### 단기 메모리 관리 {#manage-short-term-memory}

대화 히스토리는 단기 메모리의 가장 일반적인 형태이며, 긴 대화는 오늘날의 LLM에게 도전 과제가 됩니다. 전체 히스토리가 LLM의 컨텍스트 윈도우에 맞지 않을 수 있어 복구 불가능한 오류가 발생할 수 있습니다. LLM이 전체 컨텍스트 길이를 지원하더라도, 대부분의 LLM은 여전히 긴 컨텍스트에서 성능이 저하됩니다. 오래되거나 주제에서 벗어난 콘텐츠에 "산만해지며", 동시에 응답 시간이 느려지고 비용이 높아집니다.

채팅 모델은 개발자가 제공한 지침(시스템 메시지)과 사용자 입력(인간 메시지)을 포함하는 메시지를 사용하여 컨텍스트를 받아들입니다. 채팅 애플리케이션에서 메시지는 인간 입력과 모델 응답 사이를 교대하여, 시간이 지남에 따라 길어지는 메시지 목록이 생성됩니다. 컨텍스트 윈도우는 제한적이고 토큰이 많은 메시지 목록은 비용이 많이 들 수 있기 때문에, 많은 애플리케이션은 오래된 정보를 수동으로 제거하거나 잊는 기술을 사용함으로써 이점을 얻을 수 있습니다.

![](img/memory/filter.png)

메시지 관리를 위한 일반적인 기술에 대한 자세한 내용은 [메모리 추가 및 관리](../how-tos/memory/add-memory.md#manage-short-term-memory) 가이드를 참조하세요.

## 장기 메모리 {#long-term-memory}

LangGraph의 [장기 메모리](../how-tos/memory/add-memory.md#add-long-term-memory)는 시스템이 서로 다른 대화 또는 세션 간에 정보를 유지할 수 있게 합니다. **스레드 범위**인 단기 메모리와 달리, 장기 메모리는 커스텀 "네임스페이스" 내에 저장됩니다.

장기 메모리는 만능 솔루션이 없는 복잡한 과제입니다. 그러나 다음 질문들은 다양한 기술을 탐색하는 데 도움이 되는 프레임워크를 제공합니다:

- [메모리의 유형은 무엇인가?](#memory-types) 인간은 사실([의미 메모리](#semantic-memory)), 경험([일화 메모리](#episodic-memory)), 규칙([절차 메모리](#procedural-memory))을 기억하기 위해 메모리를 사용합니다. AI 에이전트도 같은 방식으로 메모리를 사용할 수 있습니다. 예를 들어, AI 에이전트는 작업을 완수하기 위해 사용자에 대한 특정 사실을 기억하기 위해 메모리를 사용할 수 있습니다.

- [메모리를 언제 업데이트하고 싶은가?](#writing-memories) 메모리는 에이전트의 애플리케이션 로직의 일부로("핫 패스에서") 업데이트될 수 있습니다. 이 경우 에이전트는 일반적으로 사용자에게 응답하기 전에 사실을 기억하기로 결정합니다. 또는 메모리는 백그라운드 작업(백그라운드에서/비동기적으로 실행되고 메모리를 생성하는 로직)으로 업데이트될 수 있습니다. [아래 섹션](#writing-memories)에서 이러한 접근 방식 간의 트레이드오프를 설명합니다.

### 메모리 유형 {#memory-types}

서로 다른 애플리케이션은 다양한 유형의 메모리를 필요로 합니다. 비유가 완벽하지는 않지만, [인간 메모리 유형](https://www.psychologytoday.com/us/basics/memory/types-of-memory?ref=blog.langchain.dev)을 살펴보는 것은 통찰력이 있을 수 있습니다. 일부 연구(예: [CoALA 논문](https://arxiv.org/pdf/2309.02427))에서는 이러한 인간 메모리 유형을 AI 에이전트에서 사용되는 유형으로 매핑하기도 했습니다.

| 메모리 유형 | 저장되는 것 | 인간 예시 | 에이전트 예시 |
|-------------|----------------|---------------|---------------|
| [의미](#semantic-memory) | 사실 | 학교에서 배운 것들 | 사용자에 대한 사실 |
| [일화](#episodic-memory) | 경험 | 내가 한 일들 | 과거 에이전트 작업 |
| [절차](#procedural-memory) | 지침 | 본능 또는 운동 기술 | 에이전트 시스템 프롬프트 |

#### 의미 메모리 {#semantic-memory}

인간과 AI 에이전트 모두에서 [의미 메모리](https://en.wikipedia.org/wiki/Semantic_memory)는 특정 사실과 개념의 보유를 포함합니다. 인간의 경우 학교에서 배운 정보와 개념 및 그 관계에 대한 이해를 포함할 수 있습니다. AI 에이전트의 경우, 의미 메모리는 종종 과거 상호작용에서 사실이나 개념을 기억하여 애플리케이션을 개인화하는 데 사용됩니다. 

!!! note

    Semantic memory is different from "semantic search," which is a technique for finding similar content using "meaning" (usually as embeddings). Semantic memory is a term from psychology, referring to storing facts and knowledge, while semantic search is a method for retrieving information based on meaning rather than exact matches.


##### 프로필

의미 메모리는 다양한 방식으로 관리될 수 있습니다. 예를 들어, 메모리는 사용자, 조직 또는 기타 엔티티(에이전트 자체 포함)에 대한 범위가 명확하고 구체적인 정보의 단일하고 지속적으로 업데이트되는 "프로필"일 수 있습니다. 프로필은 일반적으로 도메인을 나타내기 위해 선택한 다양한 키-값 쌍이 있는 JSON 문서일 뿐입니다.

프로필을 기억할 때는 매번 프로필을 **업데이트**하고 있는지 확인해야 합니다. 결과적으로 이전 프로필을 전달하고 [모델에게 새 프로필을 생성하도록 요청](https://github.com/langchain-ai/memory-template)하거나(또는 이전 프로필에 적용할 일부 [JSON 패치](https://github.com/hinthornw/trustcall)) 해야 합니다. 프로필이 커질수록 오류가 발생하기 쉬워질 수 있으며, 프로필을 여러 문서로 분할하거나 문서를 생성할 때 **엄격한** 디코딩을 사용하여 메모리 스키마가 유효한 상태로 유지되도록 하는 것이 도움이 될 수 있습니다.

![](img/memory/update-profile.png)

##### 컬렉션

또는 메모리는 시간이 지남에 따라 지속적으로 업데이트되고 확장되는 문서 컬렉션일 수 있습니다. 각 개별 메모리는 더 좁은 범위로 지정되고 생성하기가 더 쉬울 수 있으며, 이는 시간이 지남에 따라 정보를 **손실**할 가능성이 낮다는 것을 의미합니다. LLM이 새로운 정보에 대해 _새로운_ 객체를 생성하는 것이 기존 프로필과 새로운 정보를 조정하는 것보다 쉽습니다. 결과적으로 문서 컬렉션은 [더 높은 다운스트림 재현율](https://en.wikipedia.org/wiki/Precision_and_recall)로 이어지는 경향이 있습니다.

그러나 이는 메모리 업데이트에 일부 복잡성을 이동시킵니다. 이제 모델은 목록의 기존 항목을 _삭제_하거나 _업데이트_해야 하는데, 이는 까다로울 수 있습니다. 또한 일부 모델은 기본적으로 과도하게 삽입하고 다른 모델은 기본적으로 과도하게 업데이트할 수 있습니다. 이를 관리하는 한 가지 방법으로 [Trustcall](https://github.com/hinthornw/trustcall) 패키지를 참조하고, 동작을 조정하는 데 도움이 되도록 평가(예: [LangSmith](https://docs.smith.langchain.com/tutorials/Developers/evaluation)와 같은 도구 사용)를 고려하세요.

문서 컬렉션으로 작업하면 목록에 대한 메모리 **검색**으로 복잡성도 이동합니다. `Store`는 현재 [의미 검색](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.SearchOp.query)과 [콘텐츠별 필터링](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.SearchOp.filter)을 모두 지원합니다.

마지막으로, 메모리 컬렉션을 사용하면 모델에 포괄적인 컨텍스트를 제공하기가 어려울 수 있습니다. 개별 메모리가 특정 스키마를 따를 수 있지만, 이 구조는 메모리 간의 전체 컨텍스트나 관계를 포착하지 못할 수 있습니다. 결과적으로 이러한 메모리를 사용하여 응답을 생성할 때, 모델은 통합 프로필 접근 방식에서 더 쉽게 사용할 수 있는 중요한 컨텍스트 정보가 부족할 수 있습니다.

![](img/memory/update-list.png)

메모리 관리 접근 방식에 관계없이, 핵심은 에이전트가 의미 메모리를 사용하여 [응답을 근거화](https://python.langchain.com/docs/concepts/rag/)한다는 것이며, 이는 종종 더 개인화되고 관련성 있는 상호작용으로 이어집니다.

#### 일화 메모리 {#episodic-memory}

인간과 AI 에이전트 모두에서 [일화 메모리](https://en.wikipedia.org/wiki/Episodic_memory)는 과거 이벤트나 작업을 회상하는 것을 포함합니다. [CoALA 논문](https://arxiv.org/pdf/2309.02427)은 이를 잘 설명합니다: 사실은 의미 메모리에 작성될 수 있지만, *경험*은 일화 메모리에 작성될 수 있습니다. AI 에이전트의 경우, 일화 메모리는 종종 에이전트가 작업을 수행하는 방법을 기억하는 데 도움을 주기 위해 사용됩니다. 

:::python
In practice, episodic memories are often implemented through [few-shot example prompting](https://python.langchain.com/docs/concepts/few_shot_prompting/), where agents learn from past sequences to perform tasks correctly. Sometimes it's easier to "show" than "tell" and LLMs learn well from examples. Few-shot learning lets you ["program"](https://x.com/karpathy/status/1627366413840322562) your LLM by updating the prompt with input-output examples to illustrate the intended behavior. While various [best-practices](https://python.langchain.com/docs/concepts/#1-generating-examples) can be used to generate few-shot examples, often the challenge lies in selecting the most relevant examples based on user input.
:::

:::js
In practice, episodic memories are often implemented through few-shot example prompting, where agents learn from past sequences to perform tasks correctly. Sometimes it's easier to "show" than "tell" and LLMs learn well from examples. Few-shot learning lets you ["program"](https://x.com/karpathy/status/1627366413840322562) your LLM by updating the prompt with input-output examples to illustrate the intended behavior. While various best-practices can be used to generate few-shot examples, often the challenge lies in selecting the most relevant examples based on user input.
:::

:::python
Note that the memory [store](persistence.md#memory-store) is just one way to store data as few-shot examples. If you want to have more developer involvement, or tie few-shots more closely to your evaluation harness, you can also use a [LangSmith Dataset](https://docs.smith.langchain.com/evaluation/how_to_guides/datasets/index_datasets_for_dynamic_few_shot_example_selection) to store your data. Then dynamic few-shot example selectors can be used out-of-the box to achieve this same goal. LangSmith will index the dataset for you and enable retrieval of few shot examples that are most relevant to the user input based upon keyword similarity ([using a BM25-like algorithm](https://docs.smith.langchain.com/how_to_guides/datasets/index_datasets_for_dynamic_few_shot_example_selection) for keyword based similarity). 

See this how-to [video](https://www.youtube.com/watch?v=37VaU7e7t5o) for example usage of dynamic few-shot example selection in LangSmith. Also, see this [blog post](https://blog.langchain.dev/few-shot-prompting-to-improve-tool-calling-performance/) showcasing few-shot prompting to improve tool calling performance and this [blog post](https://blog.langchain.dev/aligning-llm-as-a-judge-with-human-preferences/) using few-shot example to align an LLMs to human preferences.
:::

:::js
Note that the memory [store](persistence.md#memory-store) is just one way to store data as few-shot examples. If you want to have more developer involvement, or tie few-shots more closely to your evaluation harness, you can also use a LangSmith Dataset to store your data. Then dynamic few-shot example selectors can be used out-of-the box to achieve this same goal. LangSmith will index the dataset for you and enable retrieval of few shot examples that are most relevant to the user input based upon keyword similarity.

See this how-to [video](https://www.youtube.com/watch?v=37VaU7e7t5o) for example usage of dynamic few-shot example selection in LangSmith. Also, see this [blog post](https://blog.langchain.dev/few-shot-prompting-to-improve-tool-calling-performance/) showcasing few-shot prompting to improve tool calling performance and this [blog post](https://blog.langchain.dev/aligning-llm-as-a-judge-with-human-preferences/) using few-shot example to align an LLMs to human preferences.
:::

#### 절차 메모리 {#procedural-memory}

인간과 AI 에이전트 모두에서 [절차 메모리](https://en.wikipedia.org/wiki/Procedural_memory)는 작업을 수행하는 데 사용되는 규칙을 기억하는 것을 포함합니다. 인간의 경우, 절차 메모리는 기본 운동 기술과 균형을 통해 자전거를 타는 것과 같은 작업을 수행하는 방법에 대한 내재화된 지식과 같습니다. 반면에 일화 메모리는 보조 바퀴 없이 자전거를 성공적으로 탄 첫 경험이나 경치 좋은 길을 통한 기억에 남는 자전거 타기와 같은 특정 경험을 회상하는 것을 포함합니다. AI 에이전트의 경우, 절차 메모리는 모델 가중치, 에이전트 코드 및 에이전트 프롬프트의 조합으로, 이들이 집합적으로 에이전트의 기능을 결정합니다.

실제로 에이전트가 모델 가중치를 수정하거나 코드를 다시 작성하는 것은 상당히 드뭅니다. 그러나 에이전트가 자신의 프롬프트를 수정하는 것은 더 일반적입니다. 

에이전트의 지침을 개선하는 효과적인 접근 방식 중 하나는 ["반성"](https://blog.langchain.dev/reflection-agents/) 또는 메타 프롬프팅입니다. 이는 현재 지침(예: 시스템 프롬프트)과 함께 최근 대화 또는 명시적 사용자 피드백으로 에이전트에게 프롬프트하는 것을 포함합니다. 그런 다음 에이전트는 이 입력을 기반으로 자신의 지침을 개선합니다. 이 방법은 지침을 사전에 지정하기 어려운 작업에 특히 유용하며, 에이전트가 상호작용에서 학습하고 적응할 수 있게 합니다.

예를 들어, 우리는 외부 피드백과 프롬프트 재작성을 사용하여 Twitter를 위한 고품질 논문 요약을 생성하는 [트윗 생성기](https://www.youtube.com/watch?v=Vn8A3BxfplE)를 구축했습니다. 이 경우 특정 요약 프롬프트를 *사전에* 지정하기는 어려웠지만, 사용자가 생성된 트윗을 비평하고 요약 프로세스를 개선하는 방법에 대한 피드백을 제공하는 것은 상당히 쉬웠습니다. 

The below pseudo-code shows how you might implement this with the LangGraph memory [store](persistence.md#memory-store), using the store to save a prompt, the `update_instructions` node to get the current prompt (as well as feedback from the conversation with the user captured in `state["messages"]`), update the prompt, and save the new prompt back to the store. Then, the `call_model` get the updated prompt from the store and uses it to generate a response.

:::python
```python
# Node that *uses* the instructions
def call_model(state: State, store: BaseStore):
    namespace = ("agent_instructions", )
    instructions = store.get(namespace, key="agent_a")[0]
    # Application logic
    prompt = prompt_template.format(instructions=instructions.value["instructions"])
    ...

# Node that updates instructions
def update_instructions(state: State, store: BaseStore):
    namespace = ("instructions",)
    current_instructions = store.search(namespace)[0]
    # Memory logic
    prompt = prompt_template.format(instructions=current_instructions.value["instructions"], conversation=state["messages"])
    output = llm.invoke(prompt)
    new_instructions = output['new_instructions']
    store.put(("agent_instructions",), "agent_a", {"instructions": new_instructions})
    ...
```
:::

:::js
```typescript
// Node that *uses* the instructions
const callModel = async (state: State, store: BaseStore) => {
    const namespace = ["agent_instructions"];
    const instructions = await store.get(namespace, "agent_a");
    // Application logic
    const prompt = promptTemplate.format({ 
        instructions: instructions[0].value.instructions 
    });
    // ...
};

// Node that updates instructions
const updateInstructions = async (state: State, store: BaseStore) => {
    const namespace = ["instructions"];
    const currentInstructions = await store.search(namespace);
    // Memory logic
    const prompt = promptTemplate.format({ 
        instructions: currentInstructions[0].value.instructions, 
        conversation: state.messages 
    });
    const output = await llm.invoke(prompt);
    const newInstructions = output.new_instructions;
    await store.put(["agent_instructions"], "agent_a", { 
        instructions: newInstructions 
    });
    // ...
};
```
:::

![](img/memory/update-instructions.png)

### 메모리 작성 {#writing-memories}

에이전트가 메모리를 작성하는 두 가지 주요 방법이 있습니다: ["핫 패스에서"](#in-the-hot-path)와 ["백그라운드에서"](#in-the-background)입니다.

![](img/memory/hot_path_vs_background.png)

#### 핫 패스에서 {#in-the-hot-path}

런타임 중에 메모리를 생성하는 것은 장점과 과제를 모두 제공합니다. 긍정적인 측면에서, 이 접근 방식은 실시간 업데이트를 허용하여 새로운 메모리를 후속 상호작용에서 즉시 사용할 수 있게 합니다. 또한 메모리가 생성되고 저장될 때 사용자에게 알릴 수 있어 투명성을 가능하게 합니다.

그러나 이 방법은 과제도 제시합니다. 에이전트가 메모리에 커밋할 내용을 결정하기 위해 새로운 도구가 필요한 경우 복잡성이 증가할 수 있습니다. 또한 메모리에 저장할 내용을 추론하는 프로세스가 에이전트 지연 시간에 영향을 미칠 수 있습니다. 마지막으로 에이전트는 메모리 생성과 다른 책임 사이에서 멀티태스킹을 해야 하므로 생성되는 메모리의 양과 품질에 잠재적으로 영향을 미칠 수 있습니다.

예를 들어, ChatGPT는 [save_memories](https://openai.com/index/memory-and-new-controls-for-chatgpt/) 도구를 사용하여 메모리를 콘텐츠 문자열로 업서트하고, 각 사용자 메시지에서 이 도구를 사용할지 여부와 방법을 결정합니다. 참조 구현으로 [memory-agent](https://github.com/langchain-ai/memory-agent) 템플릿을 참조하세요.

#### 백그라운드에서 {#in-the-background}

별도의 백그라운드 작업으로 메모리를 생성하는 것은 여러 장점을 제공합니다. 주요 애플리케이션의 지연 시간을 제거하고, 애플리케이션 로직을 메모리 관리에서 분리하며, 에이전트가 더 집중된 작업 완료를 할 수 있게 합니다. 이 접근 방식은 또한 중복 작업을 피하기 위해 메모리 생성 시기를 유연하게 조정할 수 있습니다.

그러나 이 방법에는 고유한 과제가 있습니다. 메모리 작성 빈도를 결정하는 것이 중요해지는데, 업데이트가 드물면 다른 스레드에 새로운 컨텍스트가 없을 수 있습니다. 메모리 형성을 트리거할 시기를 결정하는 것도 중요합니다. 일반적인 전략에는 설정된 시간 후에 스케줄링(새 이벤트가 발생하면 재스케줄링), cron 스케줄 사용 또는 사용자나 애플리케이션 로직에 의한 수동 트리거 허용이 포함됩니다.

참조 구현으로 [memory-service](https://github.com/langchain-ai/memory-template) 템플릿을 참조하세요.

### 메모리 저장

LangGraph는 장기 메모리를 [스토어](persistence.md#memory-store)의 JSON 문서로 저장합니다. 각 메모리는 커스텀 `namespace`(폴더와 유사) 및 고유한 `key`(파일 이름과 유사) 아래에 구성됩니다. 네임스페이스에는 종종 사용자 또는 조직 ID 또는 정보를 더 쉽게 구성할 수 있게 하는 기타 레이블이 포함됩니다. 이 구조는 메모리의 계층적 구성을 가능하게 합니다. 그런 다음 콘텐츠 필터를 통해 네임스페이스 간 검색이 지원됩니다.

:::python
```python
from langgraph.store.memory import InMemoryStore


def embed(texts: list[str]) -> list[list[float]]:
    # Replace with an actual embedding function or LangChain embeddings object
    return [[1.0, 2.0] * len(texts)]


# InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production use.
store = InMemoryStore(index={"embed": embed, "dims": 2})
user_id = "my-user"
application_context = "chitchat"
namespace = (user_id, application_context)
store.put(
    namespace,
    "a-memory",
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
        "my-key": "my-value",
    },
)
# get the "memory" by ID
item = store.get(namespace, "a-memory")
# search for "memories" within this namespace, filtering on content equivalence, sorted by vector similarity
items = store.search(
    namespace, filter={"my-key": "my-value"}, query="language preferences"
)
```
:::

:::js
```typescript
import { InMemoryStore } from "@langchain/langgraph";

const embed = (texts: string[]): number[][] => {
    // Replace with an actual embedding function or LangChain embeddings object
    return texts.map(() => [1.0, 2.0]);
};

// InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production use.
const store = new InMemoryStore({ index: { embed, dims: 2 } });
const userId = "my-user";
const applicationContext = "chitchat";
const namespace = [userId, applicationContext];

await store.put(
    namespace,
    "a-memory",
    {
        rules: [
            "User likes short, direct language",
            "User only speaks English & TypeScript",
        ],
        "my-key": "my-value",
    }
);

// get the "memory" by ID
const item = await store.get(namespace, "a-memory");

// search for "memories" within this namespace, filtering on content equivalence, sorted by vector similarity
const items = await store.search(
    namespace, 
    { 
        filter: { "my-key": "my-value" }, 
        query: "language preferences" 
    }
);
```
:::

메모리 스토어에 대한 자세한 내용은 [영속성](persistence.md#memory-store) 가이드를 참조하세요.
