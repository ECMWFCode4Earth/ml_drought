from pathlib import Path
from unittest.mock import patch
import pytest

from src.exporters.vhi import (
    CHIRPSExporter,
)

project_dir = Path(__file__).resolve().parents[2]

class TestCHIRPSExporter:

    def test_(tmp_path):
        CHIRPSExporter(tmp_path)
    pass
