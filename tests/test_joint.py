import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HESTON = ROOT / "artifacts" / "heston.json"
NEURAL = ROOT / "artifacts" / "neural_mark.json"


@pytest.mark.skipif(not HESTON.exists() or not NEURAL.exists(), reason="checkpoints not trained yet")
def test_heston_joint_residual_exceeds_neural():
    heston = json.loads(HESTON.read_text())
    neural = json.loads(NEURAL.read_text())
    h = heston["rmse"]
    n = neural["rmse"]
    joint_h = h["spx_iv"] ** 2 + h["vix_fut"] ** 2 + h["vix_iv"] ** 2
    joint_n = n["spx_iv"] ** 2 + n["vix_fut"] ** 2 + n["vix_iv"] ** 2
    assert joint_h > joint_n
    assert h["vix_fut"] + h["vix_iv"] > n["vix_fut"] + n["vix_iv"]
