# Self-Hosted Data Plane 배포 방법

배포하기 전에 [Self-Hosted Data Plane](../../concepts/langgraph_self_hosted_data_plane.md) 배포 옵션에 대한 개념 가이드를 검토하세요.

!!! info "중요"
    Self-Hosted Data Plane 배포 옵션은 [Enterprise](../../concepts/plans.md) 플랜이 필요합니다.

## 사전 요구사항

1. [LangGraph CLI](../../concepts/langgraph_cli.md)를 사용하여 [애플리케이션을 로컬에서 테스트](../../tutorials/langgraph-platform/local-server.md)하세요.
1. [LangGraph CLI](../../concepts/langgraph_cli.md)를 사용하여 Docker 이미지를 빌드하고(예: `langgraph build`) Kubernetes 클러스터 또는 Amazon ECS 클러스터가 액세스할 수 있는 레지스트리에 푸시하세요.

## Kubernetes

### 사전 요구사항
1. `KEDA`가 클러스터에 설치되어 있어야 합니다.

        helm repo add kedacore https://kedacore.github.io/charts
        helm install keda kedacore/keda --namespace keda --create-namespace

1. 유효한 `Ingress` 컨트롤러가 클러스터에 설치되어 있어야 합니다.
1. 클러스터에 여러 배포를 위한 여유 공간이 있어야 합니다. 새 노드를 자동으로 프로비저닝하려면 `Cluster-Autoscaler`를 권장합니다.
1. 두 개의 control plane URL에 대한 egress를 활성화해야 합니다. listener는 배포를 위해 이러한 엔드포인트를 폴링합니다:

        https://api.host.langchain.com
        https://api.smith.langchain.com

### 설정

1. LangSmith 조직 ID를 제공합니다. 조직에 대해 Self-Hosted Data Plane을 활성화합니다.
1. Kubernetes 클러스터를 설정하기 위해 실행하는 [Helm 차트](https://github.com/langchain-ai/helm/tree/main/charts/langgraph-dataplane)를 제공합니다. 이 차트에는 몇 가지 중요한 구성 요소가 포함되어 있습니다.
    1. `langgraph-listener`: 배포의 변경 사항을 LangChain의 [control plane](../../concepts/langgraph_control_plane.md)에서 수신하고 다운스트림 CRD를 생성/업데이트하는 서비스입니다.
    1. `LangGraphPlatform CRD`: LangGraph Platform 배포를 위한 CRD입니다. LangGraph Platform 배포 인스턴스를 관리하기 위한 spec을 포함합니다.
    1. `langgraph-platform-operator`: LangGraph Platform CRD의 변경 사항을 처리하는 operator입니다.
1. `langgraph-dataplane-values.yaml` 파일을 구성합니다.

        config:
          langsmithApiKey: "" # Workspace의 API Key
          langsmithWorkspaceId: "" # Workspace ID
          hostBackendUrl: "https://api.host.langchain.com" # EU인 경우에만 재정의
          smithBackendUrl: "https://api.smith.langchain.com" # EU인 경우에만 재정의

1. `langgraph-dataplane` Helm 차트를 배포합니다.

        helm repo add langchain https://langchain-ai.github.io/helm/
        helm repo update
        helm upgrade -i langgraph-dataplane langchain/langgraph-dataplane --values langgraph-dataplane-values.yaml

1. 성공하면 네임스페이스에서 두 개의 서비스가 시작되는 것을 볼 수 있습니다.

        NAME                                          READY   STATUS              RESTARTS   AGE
        langgraph-dataplane-listener-7fccd788-wn2dx   0/1     Running             0          9s
        langgraph-dataplane-redis-0                   0/1     ContainerCreating   0          9s

1. [control plane UI](../../concepts/langgraph_control_plane.md#control-plane-ui)에서 배포를 생성합니다.

## Amazon ECS

곧 제공 예정입니다!
