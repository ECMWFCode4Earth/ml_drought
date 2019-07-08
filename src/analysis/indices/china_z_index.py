from .utils import (
    rolling_cumsum,
    apply_over_period,
    create_shape_aligned_climatology
)


class ChinaZIndex(BaseIndices):
    """
    """
    def __init__(self,data_path: Path,
                 resample_str: Optional[str] = None,
                 modified: bool = False) -> None:
      # modified ChinaZIndex (use median instead of mean)
    pass
