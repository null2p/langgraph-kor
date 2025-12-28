!!! info "Prerequisites"

    - [LangGraph Studio 개요](../../../concepts/langgraph_studio.md)

LangGraph Studio는 두 가지 유형의 그래프에 연결할 수 있습니다:

- [LangGraph Platform](../../../cloud/quick_start.md)에 배포된 그래프
- [LangGraph Server](../../../tutorials/langgraph-platform/local-server.md)를 통해 로컬에서 실행되는 그래프

LangGraph Studio는 LangSmith UI의 LangGraph Platform Deployments 탭 내에서 액세스할 수 있습니다.

## 배포된 애플리케이션

LangGraph Platform에 [배포된](../../quick_start.md) 애플리케이션의 경우, 해당 배포의 일부로 Studio에 액세스할 수 있습니다. 이를 위해 LangSmith UI 내의 LangGraph Platform에서 배포로 이동하여 "LangGraph Studio" 버튼을 클릭하세요.

이렇게 하면 실시간 배포에 연결된 Studio UI가 로드되어, 해당 배포의 [threads](../../../concepts/persistence.md#threads), [assistants](../../../concepts/assistants.md), [memory](../../../concepts//memory.md)를 생성, 읽기, 업데이트할 수 있습니다.

## 로컬 개발 서버 {#local-development-server}

LangGraph Studio를 사용하여 로컬에서 실행 중인 애플리케이션을 테스트하려면 [이 가이드](https://langchain-ai.github.io/langgraph/cloud/deployment/setup/)에 따라 애플리케이션을 설정하세요.

!!! info "LangSmith 추적"
    로컬 개발의 경우, 데이터가 LangSmith로 추적되지 않기를 원한다면 애플리케이션의 `.env` 파일에 `LANGSMITH_TRACING=false`를 설정하세요. 추적이 비활성화되면 데이터가 로컬 서버를 떠나지 않습니다.

다음으로, [LangGraph CLI](../../../concepts/langgraph_cli.md)를 설치하세요:

```
pip install -U "langgraph-cli[inmem]"
```

그리고 실행하세요:

```
langgraph dev
```

!!! warning "브라우저 호환성"
    Safari는 Studio에 대한 `localhost` 연결을 차단합니다. 이를 해결하려면 `--tunnel`과 함께 위 명령을 실행하여 보안 터널을 통해 Studio에 액세스하세요.

이렇게 하면 LangGraph Server가 로컬에서 인메모리로 실행됩니다. 서버는 watch 모드로 실행되어 코드 변경 사항을 수신 대기하고 자동으로 재시작합니다. API 서버를 시작하는 모든 옵션을 알아보려면 이 [참조](https://langchain-ai.github.io/langgraph/cloud/reference/cli/#dev)를 읽어보세요.

성공하면 다음 로그가 표시됩니다:

> Ready!
>
> - API: [http://localhost:2024](http://localhost:2024/)
>
> - Docs: http://localhost:2024/docs
>
> - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

실행되면 자동으로 LangGraph Studio로 이동됩니다.

이미 실행 중인 서버의 경우 다음 중 하나로 Studio에 액세스하세요:

1.  다음 URL로 직접 이동: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.
2.  LangSmith 내에서 LangGraph Platform Deployments 탭으로 이동하여 "LangGraph Studio" 버튼을 클릭하고 `http://127.0.0.1:2024`를 입력한 다음 "Connect"를 클릭합니다.

다른 호스트나 포트에서 서버를 실행하는 경우, `baseUrl`을 일치하도록 업데이트하세요.

### (선택 사항) 디버거 연결

중단점 및 변수 검사를 사용한 단계별 디버깅의 경우:

```bash
# debugpy 패키지 설치
pip install debugpy

# 디버깅이 활성화된 상태로 서버 시작
langgraph dev --debug-port 5678
```

그런 다음 선호하는 디버거를 연결하세요:

=== "VS Code"

    `launch.json`에 이 구성을 추가하세요:

    ```json
    {
        "name": "Attach to LangGraph",
        "type": "debugpy",
        "request": "attach",
        "connect": {
          "host": "0.0.0.0",
          "port": 5678
        }
    }
    ```

=== "PyCharm"

    1. Run → Edit Configurations로 이동
    2. +를 클릭하고 "Python Debug Server" 선택
    3. IDE host name 설정: `localhost`
    4. port 설정: `5678` (또는 이전 단계에서 선택한 포트 번호)
    5. "OK"를 클릭하고 디버깅 시작

## 문제 해결

시작하는 데 문제가 있는 경우, 이 [문제 해결 가이드](../../../troubleshooting/studio.md)를 참조하세요.

## 다음 단계

Studio 사용 방법에 대한 자세한 내용은 다음 가이드를 참조하세요:

- [애플리케이션 실행](../invoke_studio.md)
- [어시스턴트 관리](./manage_assistants.md)
- [Thread 관리](../threads_studio.md)
- [프롬프트 반복 개선](../iterate_graph_studio.md)
- [LangSmith 추적 디버그](../clone_traces_studio.md)
- [데이터셋에 노드 추가](../datasets_studio.md)
