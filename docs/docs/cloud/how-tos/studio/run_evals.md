# 데이터셋에 대한 실험 실행

LangGraph Studio는 미리 정의된 LangSmith 데이터셋에 대해 assistant를 실행할 수 있도록 하여 평가를 지원합니다. 이를 통해 다양한 입력에 대한 애플리케이션 성능을 이해하고, 결과를 참조 출력과 비교하고, [evaluator](../../../agents/evals.md)를 사용하여 결과를 점수화할 수 있습니다.

이 가이드는 Studio에서 실험을 처음부터 끝까지 실행하는 방법을 보여줍니다.

---

## 사전 요구사항

실험을 실행하기 전에 다음이 필요합니다:

1.  **LangSmith 데이터셋**: 데이터셋에는 테스트하려는 입력과 선택적으로 비교를 위한 참조 출력이 포함되어야 합니다.

    - 입력의 스키마는 assistant에 필요한 입력 스키마와 일치해야 합니다. 스키마에 대한 자세한 내용은 [여기](../../../concepts/low_level.md#schema)를 참조하세요.
    - 데이터셋 생성에 대한 자세한 내용은 [데이터셋 관리 방법](https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_in_application#set-up-your-dataset)을 참조하세요.

2.  **(선택 사항) Evaluator**: LangSmith의 데이터셋에 evaluator(예: LLM-as-a-Judge, 휴리스틱 또는 커스텀 함수)를 연결할 수 있습니다. 이들은 그래프가 모든 입력을 처리한 후 자동으로 실행됩니다.

    - 자세한 내용은 [Evaluation 개념](https://docs.smith.langchain.com/evaluation/concepts#evaluators)을 참조하세요.

3.  **실행 중인 애플리케이션**: 실험은 다음에 대해 실행될 수 있습니다:
    - [LangGraph Platform](../../quick_start.md)에 배포된 애플리케이션.
    - [langgraph-cli](../../../tutorials/langgraph-platform/local-server.md)를 통해 시작된 로컬 실행 애플리케이션.

---

## 단계별 가이드

### 1. 실험 시작

Studio 페이지 오른쪽 상단의 **Run experiment** 버튼을 클릭합니다.

### 2. 데이터셋 선택

나타나는 모달에서 실험에 사용할 데이터셋(또는 특정 데이터셋 분할)을 선택하고 **Start**를 클릭합니다.

### 3. 진행 상황 모니터링

이제 데이터셋의 모든 입력이 활성 assistant에 대해 실행됩니다. 오른쪽 상단의 배지를 통해 실험 진행 상황을 모니터링합니다.

실험이 백그라운드에서 실행되는 동안 Studio에서 계속 작업할 수 있습니다. 언제든지 화살표 아이콘 버튼을 클릭하여 LangSmith로 이동하고 자세한 실험 결과를 확인할 수 있습니다.

---

## 트러블슈팅

### "Run experiment" 버튼이 비활성화됨

"Run experiment" 버튼이 비활성화되어 있는 경우 다음을 확인하세요:

- **배포된 애플리케이션**: 애플리케이션이 LangGraph Platform에 배포된 경우 이 기능을 활성화하려면 새 리비전을 생성해야 할 수 있습니다.
- **로컬 개발 서버**: 애플리케이션을 로컬에서 실행하는 경우 최신 버전의 `langgraph-cli`로 업그레이드했는지 확인하세요(`pip install -U langgraph-cli`). 또한 프로젝트의 `.env` 파일에서 `LANGSMITH_API_KEY`를 설정하여 추적을 활성화했는지 확인하세요.

### Evaluator 결과가 누락됨

실험을 실행하면 연결된 evaluator가 실행 대기열에 예약됩니다. 결과가 즉시 표시되지 않으면 아직 대기 중일 가능성이 높습니다.
