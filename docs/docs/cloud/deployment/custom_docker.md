# Dockerfile 사용자 정의 방법

사용자는 부모 LangGraph 이미지에서 import한 후 Dockerfile에 추가할 추가 라인의 배열을 추가할 수 있습니다. 이를 위해서는 `langgraph.json` 파일을 수정하고 실행하려는 명령을 `dockerfile_lines` 키에 전달하기만 하면 됩니다. 예를 들어, 그래프에서 `Pillow`를 사용하려면 다음 의존성을 추가해야 합니다:

```
{
    "dependencies": ["."],
    "graphs": {
        "openai_agent": "./openai_agent.py:agent",
    },
    "env": "./.env",
    "dockerfile_lines": [
        "RUN apt-get update && apt-get install -y libjpeg-dev zlib1g-dev libpng-dev",
        "RUN pip install Pillow"
    ]
}
```

이는 `jpeg` 또는 `png` 이미지 형식으로 작업하는 경우 Pillow를 사용하는 데 필요한 시스템 패키지를 설치합니다.
