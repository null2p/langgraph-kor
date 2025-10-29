# Self-Hosted Control Plane 배포 방법

배포하기 전에 [Self-Hosted Control Plane](../../concepts/langgraph_self_hosted_control_plane.md) 배포 옵션에 대한 개념 가이드를 검토하세요.

!!! info "중요"
    Self-Hosted Control Plane 배포 옵션은 [Enterprise](../../concepts/plans.md) 플랜이 필요합니다.

## 사전 요구사항

1. Kubernetes를 사용하고 있어야 합니다.
1. self-hosted LangSmith가 배포되어 있어야 합니다.
1. [LangGraph CLI](../../concepts/langgraph_cli.md)를 사용하여 [애플리케이션을 로컬에서 테스트](../../tutorials/langgraph-platform/local-server.md)하세요.
1. [LangGraph CLI](../../concepts/langgraph_cli.md)를 사용하여 Docker 이미지를 빌드하고(예: `langgraph build`) Kubernetes 클러스터가 액세스할 수 있는 레지스트리에 푸시하세요.
1. `KEDA`가 클러스터에 설치되어 있어야 합니다.

         helm repo add kedacore https://kedacore.github.io/charts
         helm install keda kedacore/keda --namespace keda --create-namespace
1. Ingress 구성
    1. LangSmith 인스턴스에 대한 ingress를 설정해야 합니다. 모든 에이전트는 이 ingress 뒤의 Kubernetes 서비스로 배포됩니다.
    1. 인스턴스에 대한 [ingress 설정](https://docs.smith.langchain.com/self_hosting/configuration/ingress) 가이드를 사용할 수 있습니다.
1. 클러스터에 여러 배포를 위한 여유 공간이 있어야 합니다. 새 노드를 자동으로 프로비저닝하려면 `Cluster-Autoscaler`를 권장합니다.
1. 클러스터에서 사용 가능한 유효한 Dynamic PV provisioner 또는 PV가 있어야 합니다. 다음을 실행하여 확인할 수 있습니다:

        kubectl get storageclass

1. 네트워크에서 `https://beacon.langchain.com`으로의 egress가 필요합니다. air-gapped 모드에서 실행하지 않는 경우 라이선스 검증 및 사용량 보고에 필요합니다. 자세한 내용은 [Egress 문서](../../cloud/deployment/egress.md)를 참조하세요.

## 설정

1. Self-Hosted LangSmith 인스턴스를 구성하는 과정에서 `langgraphPlatform` 옵션을 활성화합니다. 이렇게 하면 몇 가지 핵심 리소스가 프로비저닝됩니다.
    1. `listener`: 배포의 변경 사항을 [control plane](../../concepts/langgraph_control_plane.md)에서 수신하고 다운스트림 CRD를 생성/업데이트하는 서비스입니다.
    1. `LangGraphPlatform CRD`: LangGraph Platform 배포를 위한 CRD입니다. LangGraph Platform 배포 인스턴스를 관리하기 위한 spec을 포함합니다.
    1. `operator`: LangGraph Platform CRD의 변경 사항을 처리하는 operator입니다.
    1. `host-backend`: [control plane](../../concepts/langgraph_control_plane.md)입니다.
1. 차트에서 두 개의 추가 이미지가 사용됩니다. 최신 릴리스에 지정된 이미지를 사용하세요.

        hostBackendImage:
          repository: "docker.io/langchain/hosted-langserve-backend"
          pullPolicy: IfNotPresent
        operatorImage:
          repository: "docker.io/langchain/langgraph-operator"
          pullPolicy: IfNotPresent

1. langsmith의 구성 파일(일반적으로 `langsmith_config.yaml`)에서 `langgraphPlatform` 옵션을 활성화합니다. 유효한 ingress 설정도 필요합니다:

        config:
          langgraphPlatform:
            enabled: true
            langgraphPlatformLicenseKey: "YOUR_LANGGRAPH_PLATFORM_LICENSE_KEY"
1. `values.yaml` 파일에서 `hostBackendImage` 및 `operatorImage` 옵션을 구성합니다(이미지를 미러링해야 하는 경우)

1. [여기](https://github.com/langchain-ai/helm/blob/main/charts/langsmith/values.yaml#L898)의 기본 템플릿을 재정의하여 에이전트에 대한 기본 템플릿을 구성할 수도 있습니다.
1. [control plane UI](../../concepts/langgraph_control_plane.md#control-plane-ui)에서 배포를 생성합니다.
