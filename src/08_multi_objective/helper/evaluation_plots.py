"""Small compatibility helpers for report artifact names and metric text."""
from pathlib import Path
import numpy as np

def _artifact_path(
    output_dir: Path,
    base_name: str,
    suffix: str,
    artifact_prefix: str = "",
) -> Path:
    prefix = f"{artifact_prefix}_" if str(artifact_prefix).strip() else ""
    return output_dir / f"{prefix}{base_name}{suffix}"

def _format_metric(value: float | None, fmt: str = "{:.2f}") -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return fmt.format(float(value))
