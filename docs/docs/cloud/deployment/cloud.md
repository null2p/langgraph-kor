# Cloud SaaS 배포 방법

배포하기 전에 [Cloud SaaS](../../concepts/langgraph_cloud.md) 배포 옵션에 대한 개념 가이드를 검토하세요.

## Prerequisites

1. LangGraph Platform 애플리케이션은 GitHub 리포지토리에서 배포됩니다. LangGraph Platform에 배포하기 위해 LangGraph Platform 애플리케이션을 GitHub 리포지토리에 구성하고 업로드합니다.
1. [LangGraph API가 로컬에서 실행되는지 확인](../../tutorials/langgraph-platform/local-server.md)합니다. API가 성공적으로 실행되지 않으면(예: `langgraph dev`), LangGraph Platform에 배포하는 것도 실패합니다.

## Create New Deployment

<a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>에서 시작하여...

1. 왼쪽 탐색 패널에서 `LangGraph Platform`을 선택합니다. `LangGraph Platform` 뷰에는 기존 LangGraph Platform 배포 목록이 포함되어 있습니다.
1. 오른쪽 상단에서 `+ New Deployment`를 선택하여 새 배포를 생성합니다.
1. `Create New Deployment` 패널에서 필수 필드를 작성합니다.
    1. `Deployment details`
        1. `Import from GitHub`을 선택하고 GitHub OAuth 워크플로우를 따라 LangChain의 `hosted-langserve` GitHub 앱을 설치하고 선택한 리포지토리에 액세스할 수 있도록 권한을 부여합니다. 설치가 완료되면 `Create New Deployment` 패널로 돌아가서 드롭다운 메뉴에서 배포할 GitHub 리포지토리를 선택합니다. **참고**: LangChain의 `hosted-langserve` GitHub 앱을 설치하는 GitHub 사용자는 조직 또는 계정의 [소유자](https://docs.github.com/en/organizations/managing-peoples-access-to-your-organization-with-roles/roles-in-an-organization#organization-owners)여야 합니다.
        1. 배포 이름을 지정합니다.
        1. 원하는 `Git Branch`를 지정합니다. 배포는 브랜치에 연결됩니다. 새 리비전이 생성되면 연결된 브랜치의 코드가 배포됩니다. 브랜치는 나중에 [Deployment Settings](#deployment-settings)에서 업데이트할 수 있습니다.
        1. 파일 이름을 포함한 [LangGraph API config 파일](../reference/cli.md#configuration-file)의 전체 경로를 지정합니다. 예를 들어, `langgraph.json` 파일이 리포지토리의 루트에 있는 경우 `langgraph.json`만 지정하면 됩니다.
        1. `Automatically update deployment on push to branch` 체크박스를 선택/선택 해제합니다. 선택하면 지정된 `Git Branch`에 변경 사항이 푸시될 때 배포가 자동으로 업데이트됩니다. 이 설정은 나중에 [Deployment Settings](#deployment-settings)에서 활성화/비활성화할 수 있습니다.
    1. 원하는 `Deployment Type`을 선택합니다.
        1. `Development` 배포는 비프로덕션 사용 사례를 위한 것이며 최소한의 리소스로 프로비저닝됩니다.
        1. `Production` 배포는 초당 최대 500개의 요청을 처리할 수 있으며 자동 백업이 포함된 고가용성 스토리지로 프로비저닝됩니다.
    1. 배포가 `Shareable through LangGraph Studio`인지 결정합니다.
        1. 선택하지 않으면 배포는 작업 공간의 유효한 LangSmith API 키로만 액세스할 수 있습니다.
        1. 선택하면 배포는 모든 LangSmith 사용자에게 LangGraph Studio를 통해 액세스할 수 있습니다. 다른 LangSmith 사용자와 공유할 수 있도록 배포에 대한 LangGraph Studio의 직접 URL이 제공됩니다.
    1. `Environment Variables` 및 비밀을 지정합니다. 배포에 대한 추가 변수를 구성하려면 [Environment Variables reference](../reference/env_var.md)를 참조하세요.
        1. API 키(예: `OPENAI_API_KEY`)와 같은 민감한 값은 비밀로 지정해야 합니다.
        1. 추가 비밀이 아닌 환경 변수도 지정할 수 있습니다.
    1. 새 LangSmith `Tracing Project`가 배포와 동일한 이름으로 자동 생성됩니다.
1. 오른쪽 상단에서 `Submit`을 선택합니다. 몇 초 후 `Deployment` 뷰가 나타나고 새 배포가 프로비저닝 대기열에 추가됩니다.

## Create New Revision

[새 배포를 생성](#create-new-deployment)할 때 기본적으로 새 리비전이 생성됩니다. 새 코드 변경 사항을 배포하기 위해 후속 리비전을 생성할 수 있습니다.

<a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>에서 시작하여...

1. 왼쪽 탐색 패널에서 `LangGraph Platform`을 선택합니다. `LangGraph Platform` 뷰에는 기존 LangGraph Platform 배포 목록이 포함되어 있습니다.
1. 새 리비전을 생성할 기존 배포를 선택합니다.
1. `Deployment` 뷰의 오른쪽 상단에서 `+ New Revision`을 선택합니다.
1. `New Revision` 모달에서 필수 필드를 작성합니다.
    1. 파일 이름을 포함한 [LangGraph API config 파일](../reference/cli.md#configuration-file)의 전체 경로를 지정합니다. 예를 들어, `langgraph.json` 파일이 리포지토리의 루트에 있는 경우 `langgraph.json`만 지정하면 됩니다.
    1. 배포가 `Shareable through LangGraph Studio`인지 결정합니다.
        1. 선택하지 않으면 배포는 작업 공간의 유효한 LangSmith API 키로만 액세스할 수 있습니다.
        1. 선택하면 배포는 모든 LangSmith 사용자에게 LangGraph Studio를 통해 액세스할 수 있습니다. 다른 LangSmith 사용자와 공유할 수 있도록 배포에 대한 LangGraph Studio의 직접 URL이 제공됩니다.
    1. `Environment Variables` 및 비밀을 지정합니다. 기존 비밀 및 환경 변수가 미리 채워집니다. 리비전에 대한 추가 변수를 구성하려면 [Environment Variables reference](../reference/env_var.md)를 참조하세요.
        1. 새 비밀 또는 환경 변수를 추가합니다.
        1. 기존 비밀 또는 환경 변수를 제거합니다.
        1. 기존 비밀 또는 환경 변수의 값을 업데이트합니다.
1. `Submit`을 선택합니다. 몇 초 후 `New Revision` 모달이 닫히고 새 리비전이 배포 대기열에 추가됩니다.

## View Build and Server Logs

각 리비전에 대한 빌드 및 서버 로그를 사용할 수 있습니다.

`LangGraph Platform` 뷰에서 시작하여...

1. `Revisions` 테이블에서 원하는 리비전을 선택합니다. 오른쪽에서 패널이 슬라이드되어 열리고 `Build` 탭이 기본적으로 선택되며, 리비전에 대한 빌드 로그가 표시됩니다.
1. 패널에서 `Server` 탭을 선택하여 리비전에 대한 서버 로그를 봅니다. 서버 로그는 리비전이 배포된 후에만 사용할 수 있습니다.
1. `Server` 탭 내에서 필요에 따라 날짜/시간 범위 선택기를 조정합니다. 기본적으로 날짜/시간 범위 선택기는 `Last 7 days`로 설정됩니다.

## View Deployment Metrics

<a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>에서 시작하여...

1. 왼쪽 탐색 패널에서 `LangGraph Platform`을 선택합니다. `LangGraph Platform` 뷰에는 기존 LangGraph Platform 배포 목록이 포함되어 있습니다.
1. 모니터링할 기존 배포를 선택합니다.
1. `Monitoring` 탭을 선택하여 배포 메트릭을 봅니다. [사용 가능한 모든 메트릭](../../concepts/langgraph_control_plane.md#monitoring) 목록을 참조하세요.
1. `Monitoring` 탭 내에서 필요에 따라 날짜/시간 범위 선택기를 사용합니다. 기본적으로 날짜/시간 범위 선택기는 `Last 15 minutes`로 설정됩니다.

## Interrupt Revision

리비전을 중단하면 리비전 배포가 중지됩니다.

!!! warning "Undefined Behavior"
    중단된 리비전은 정의되지 않은 동작을 합니다. 이는 새 리비전을 배포해야 하고 이미 진행 중인 리비전이 "멈춰 있는" 경우에만 유용합니다. 향후 이 기능은 제거될 수 있습니다.

`LangGraph Platform` 뷰에서 시작하여...

1. `Revisions` 테이블에서 원하는 리비전의 행 오른쪽에 있는 메뉴 아이콘(점 세 개)을 선택합니다.
1. 메뉴에서 `Interrupt`를 선택합니다.
1. 모달이 나타납니다. 확인 메시지를 검토합니다. `Interrupt revision`을 선택합니다.

## Delete Deployment

<a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>에서 시작하여...

1. 왼쪽 탐색 패널에서 `LangGraph Platform`을 선택합니다. `LangGraph Platform` 뷰에는 기존 LangGraph Platform 배포 목록이 포함되어 있습니다.
1. 원하는 배포의 행 오른쪽에 있는 메뉴 아이콘(점 세 개)을 선택하고 `Delete`를 선택합니다.
1. `Confirmation` 모달이 나타납니다. `Delete`를 선택합니다.

## Deployment Settings

`LangGraph Platform` 뷰에서 시작하여...

1. 오른쪽 상단에서 기어 아이콘(`Deployment Settings`)을 선택합니다.
1. `Git Branch`를 원하는 브랜치로 업데이트합니다.
1. `Automatically update deployment on push to branch` 체크박스를 선택/선택 해제합니다.
1. 브랜치 생성/삭제 및 태그 생성/삭제 이벤트는 업데이트를 트리거하지 않습니다. 기존 브랜치에 대한 푸시만 업데이트를 트리거합니다.
1. 브랜치에 빠르게 연속해서 푸시하면 후속 업데이트가 대기열에 추가됩니다. 빌드가 완료되면 가장 최근 커밋이 빌드를 시작하고 대기열에 있는 다른 빌드는 건너뜁니다.

## Add or Remove GitHub Repositories

LangChain의 `hosted-langserve` GitHub 앱을 설치하고 권한을 부여한 후, 앱의 리포지토리 액세스를 수정하여 새 리포지토리를 추가하거나 기존 리포지토리를 제거할 수 있습니다. 새 리포지토리가 생성된 경우 명시적으로 추가해야 할 수 있습니다.

1. GitHub 프로필에서 `Settings` > `Applications` > `hosted-langserve` > `Configure` 클릭으로 이동합니다.
1. `Repository access` 아래에서 `All repositories` 또는 `Only select repositories`를 선택합니다. `Only select repositories`를 선택한 경우 새 리포지토리를 명시적으로 추가해야 합니다.
1. `Save`를 클릭합니다.
1. 새 배포를 생성할 때 드롭다운 메뉴의 GitHub 리포지토리 목록이 리포지토리 액세스 변경 사항을 반영하여 업데이트됩니다.

## Whitelisting IP Addresses

2025년 1월 6일 이후에 생성된 `LangGraph Platform` 배포의 모든 트래픽은 NAT 게이트웨이를 통해 전송됩니다.
이 NAT 게이트웨이는 배포하는 지역에 따라 여러 정적 IP 주소를 갖습니다. 화이트리스트에 추가할 IP 주소 목록은 아래 표를 참조하세요:

| US             | EU              |
|----------------|-----------------|
| 35.197.29.146  | 34.90.213.236   |
| 34.145.102.123 | 34.13.244.114   |
| 34.169.45.153  | 34.32.180.189   |
| 34.82.222.17   | 34.34.69.108    |
| 35.227.171.135 | 34.32.145.240   |
| 34.169.88.30   | 34.90.157.44    |
| 34.19.93.202   | 34.141.242.180  |
| 34.19.34.50    | 34.32.141.108   |
