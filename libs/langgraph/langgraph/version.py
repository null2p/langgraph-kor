"""패키지 버전을 내보냅니다."""

from importlib import metadata

__all__ = ("__version__",)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # 패키지 메타데이터를 사용할 수 없는 경우입니다.
    __version__ = ""
del metadata  # 선택사항, dir(__package__)의 결과가 오염되는 것을 방지합니다
