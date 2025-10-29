# INVALID_LICENSE

이 에러는 self-hosted LangGraph Platform 서버를 시작하려고 할 때 라이선스 검증이 실패하면 발생합니다. 이 에러는 LangGraph Platform에만 해당되며 오픈소스 라이브러리와는 관련이 없습니다.

## 발생 시점

이 에러는 유효한 엔터프라이즈 라이선스나 API 키 없이 LangGraph Platform의 self-hosted 배포를 실행할 때 발생합니다.

## 트러블슈팅

### 배포 유형 확인

먼저 원하는 배포 모드를 확인하세요.

#### 로컬 개발용

로컬에서 개발만 하는 경우, `langgraph dev`를 실행하여 경량 인메모리 서버를 사용할 수 있습니다.
자세한 내용은 [로컬 서버](../../tutorials/langgraph-platform/local-server.md) 문서를 참조하세요.

#### 관리형 LangGraph Platform용

빠른 관리형 환경을 원하시면 [Cloud SaaS](../../concepts/langgraph_cloud.md) 배포 옵션을 고려해보세요. 추가 라이선스 키가 필요하지 않습니다.

#### Standalone Container용

self-hosting의 경우 `LANGGRAPH_CLOUD_LICENSE_KEY` 환경 변수를 설정하세요. 엔터프라이즈 라이선스 키에 관심이 있으시면 LangChain 지원팀에 문의하세요.

배포 옵션과 기능에 대한 자세한 내용은 [배포 옵션](../../concepts/deployment_options.md) 문서를 참조하세요.


### 자격 증명 확인

LangGraph Platform을 self-host하려는 것이 확실하다면 자격 증명을 확인하세요.

#### Standalone Container용

1. 배포 환경이나 `.env` 파일에 작동하는 `LANGGRAPH_CLOUD_LICENSE_KEY` 환경 변수를 제공했는지 확인하세요
2. 키가 아직 유효하고 만료일을 넘지 않았는지 확인하세요
