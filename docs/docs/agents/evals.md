---
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# Evals

에이전트의 성능을 평가하려면 `LangSmith` [evaluations](https://docs.smith.langchain.com/evaluation)를 사용할 수 있습니다. 먼저 최종 출력이나 궤적(trajectory)과 같은 에이전트 결과를 판단하는 평가 함수를 정의해야 합니다. 평가 기법에 따라 참조 출력이 필요할 수도 있고 필요하지 않을 수도 있습니다:

:::python

```python
def evaluator(*, outputs: dict, reference_outputs: dict):
    # compare agent outputs against reference outputs
    output_messages = outputs["messages"]
    reference_messages = reference_outputs["messages"]
    score = compare_messages(output_messages, reference_messages)
    return {"key": "evaluator_score", "score": score}
```

:::

:::js

```typescript
type EvaluatorParams = {
  outputs: Record<string, any>;
  referenceOutputs: Record<string, any>;
};

function evaluator({ outputs, referenceOutputs }: EvaluatorParams) {
  // compare agent outputs against reference outputs
  const outputMessages = outputs.messages;
  const referenceMessages = referenceOutputs.messages;
  const score = compareMessages(outputMessages, referenceMessages);
  return { key: "evaluator_score", score: score };
}
```

:::

시작하려면 `AgentEvals` 패키지에서 미리 제공되는 평가자를 사용할 수 있습니다:

:::python

```bash
pip install -U agentevals
```

:::

:::js

```bash
npm install agentevals
```

:::

## Create evaluator

에이전트 성능을 평가하는 일반적인 방법은 에이전트의 궤적(도구를 호출하는 순서)을 참조 궤적과 비교하는 것입니다:

:::python

```python
import json
# highlight-next-line
from agentevals.trajectory.match import create_trajectory_match_evaluator

outputs = [
    {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "san francisco"}),
                }
            },
            {
                "function": {
                    "name": "get_directions",
                    "arguments": json.dumps({"destination": "presidio"}),
                }
            }
        ],
    }
]
reference_outputs = [
    {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "san francisco"}),
                }
            },
        ],
    }
]

# Create the evaluator
evaluator = create_trajectory_match_evaluator(
    # highlight-next-line
    trajectory_match_mode="superset",  # (1)!
)

# Run the evaluator
result = evaluator(
    outputs=outputs, reference_outputs=reference_outputs
)
```

:::

:::js

```typescript
import { createTrajectoryMatchEvaluator } from "agentevals/trajectory/match";

const outputs = [
  {
    role: "assistant",
    tool_calls: [
      {
        function: {
          name: "get_weather",
          arguments: JSON.stringify({ city: "san francisco" }),
        },
      },
      {
        function: {
          name: "get_directions",
          arguments: JSON.stringify({ destination: "presidio" }),
        },
      },
    ],
  },
];

const referenceOutputs = [
  {
    role: "assistant",
    tool_calls: [
      {
        function: {
          name: "get_weather",
          arguments: JSON.stringify({ city: "san francisco" }),
        },
      },
    ],
  },
];

// Create the evaluator
const evaluator = createTrajectoryMatchEvaluator({
  // Specify how the trajectories will be compared. `superset` will accept output trajectory as valid if it's a superset of the reference one. Other options include: strict, unordered and subset
  trajectoryMatchMode: "superset", // (1)!
});

// Run the evaluator
const result = evaluator({
  outputs: outputs,
  referenceOutputs: referenceOutputs,
});
```

:::

1. 궤적을 비교하는 방법을 지정합니다. `superset`은 출력 궤적이 참조 궤적의 상위 집합인 경우 유효한 것으로 간주합니다. 다른 옵션으로는 [strict](https://github.com/langchain-ai/agentevals?tab=readme-ov-file#strict-match), [unordered](https://github.com/langchain-ai/agentevals?tab=readme-ov-file#unordered-match), [subset](https://github.com/langchain-ai/agentevals?tab=readme-ov-file#subset-and-superset-match)이 있습니다.

다음 단계로, [trajectory match evaluator를 커스터마이징하는 방법](https://github.com/langchain-ai/agentevals?tab=readme-ov-file#agent-trajectory-match)에 대해 자세히 알아보세요.

### LLM-as-a-judge

LLM을 사용하여 궤적을 참조 출력과 비교하고 점수를 출력하는 LLM-as-a-judge 평가자를 사용할 수 있습니다:

:::python

```python
import json
from agentevals.trajectory.llm import (
    # highlight-next-line
    create_trajectory_llm_as_judge,
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE
)

evaluator = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    model="openai:o3-mini"
)
```

:::

:::js

```typescript
import {
  createTrajectoryLlmAsJudge,
  TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
} from "agentevals/trajectory/llm";

const evaluator = createTrajectoryLlmAsJudge({
  prompt: TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
  model: "openai:o3-mini",
});
```

:::

## Run evaluator

평가자를 실행하려면 먼저 [LangSmith dataset](https://docs.smith.langchain.com/evaluation/concepts#datasets)을 생성해야 합니다. 미리 제공되는 AgentEvals 평가자를 사용하려면 다음 스키마를 가진 데이터셋이 필요합니다:

- **input**: `{"messages": [...]}` 에이전트를 호출할 때 사용할 입력 메시지.
- **output**: `{"messages": [...]}` 에이전트 출력에서 예상되는 메시지 히스토리. 궤적 평가의 경우, assistant 메시지만 유지하도록 선택할 수 있습니다.

:::python

```python
from langsmith import Client
from langgraph.prebuilt import create_react_agent
from agentevals.trajectory.match import create_trajectory_match_evaluator

client = Client()
agent = create_react_agent(...)
evaluator = create_trajectory_match_evaluator(...)

experiment_results = client.evaluate(
    lambda inputs: agent.invoke(inputs),
    # replace with your dataset name
    data="<Name of your dataset>",
    evaluators=[evaluator]
)
```

:::

:::js

```typescript
import { Client } from "langsmith";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { createTrajectoryMatchEvaluator } from "agentevals/trajectory/match";

const client = new Client();
const agent = createReactAgent({...});
const evaluator = createTrajectoryMatchEvaluator({...});

const experimentResults = await client.evaluate(
    (inputs) => agent.invoke(inputs),
    // replace with your dataset name
    { data: "<Name of your dataset>" },
    { evaluators: [evaluator] }
);
```

:::
