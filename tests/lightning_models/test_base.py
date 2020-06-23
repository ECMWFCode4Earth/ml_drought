import numpy as np
import pandas as pd
import pytest

from src.lightning_models.model_base import LightningBase


class TestLightningBase:
    def test_(self, tmp_path):
        l = LightningBase()
