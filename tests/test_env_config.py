"""Tests for the environment configuration (no gymnasium or scipy needed)."""

from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark.env import (  # noqa: E402
    REPORTED_DYNAMIC_CONFIG,
    DynamicRiskConfig,
    EnvConfig,
    EpisodeConfig,
    EvaluationRegime,
    LidarConfig,
    MissionConfig,
    make_config,
)
from ruscuav_benchmark.risk import RISK_PROFILES  # noqa: E402


def test_presets_resolve_stage_regime_and_reward():
    c = make_config("RISK+GREEN-RL", regime="primary_no_terminal")
    assert (c.stage, c.mode, c.observation_dim) == ("B", "strict_eval", 80)
    assert (c.use_urban_risk_costs, c.resolved_urban_risk_lambda, c.resolved_green_bonus_lambda) == (True, 0.10, 0.03)
    assert c.resolved_hard_sensitive_zones and not c.resolved_terminate_on_urban_constraint

    c = make_config("OBS-RL", stage="A", mode="fast_train")
    assert c.observation_dim == 100 and not c.use_urban_risk_costs
    assert not c.resolved_hard_sensitive_zones
    assert (c.resolved_lidar_step_m, c.resolved_segment_check_step_m) == (5.0, 2.0)

    c = make_config("GREEN-RL", regime="secondary_operational_stopping")
    assert (c.resolved_urban_risk_lambda, c.resolved_green_bonus_lambda) == (0.0, 0.03)
    assert c.resolved_terminate_on_urban_constraint

    assert make_config("RISK-RL").resolved_green_bonus_lambda == 0.0
    assert make_config("DYN-OBS-RL").observation_dim == 86
    assert make_config("DYN-OBS-RL").dynamic_risk == REPORTED_DYNAMIC_CONFIG
    assert make_config("RISK+OBS-RL", urban_risk_lambda=0.5).resolved_urban_risk_lambda == 0.5
    assert make_config("GEO-RL", lidar=LidarConfig(n_rays=32)).observation_dim == 48


def test_regime_objects_set_operational_stopping():
    custom = EvaluationRegime("custom", stage="B", terminate_on_soft_budgets=True)
    assert make_config("RISK+GREEN-RL", regime=custom).resolved_terminate_on_urban_constraint
    assert not make_config("RISK+GREEN-RL", regime=custom,
                           terminate_on_urban_constraint=False).resolved_terminate_on_urban_constraint


def test_profiles_and_dynamic_configurations_are_accepted_in_every_form():
    assert make_config("GEO-RL", risk_profile=RISK_PROFILES["safety_first"]).risk_profile.name == "safety_first"
    c = make_config("DYN-RISK-RL", dynamic_risk={"n_hotspots": 4, "temporal_mode": "box"})
    assert isinstance(c.dynamic_risk, DynamicRiskConfig) and c.dynamic_risk.n_hotspots == 4
    assert c.to_dict()["dynamic_risk"]["n_hotspots"] == 4


@pytest.mark.parametrize("make, match", [
    (lambda: EnvConfig(observation="dynamic"), "dynamic_risk"),
    (lambda: EnvConfig(stage="D"), "stage"),
    (lambda: EnvConfig(risk_profile="cautious"), "risk_profile"),
    (lambda: EnvConfig(dynamic_risk=42), "dynamic_risk"),
    (lambda: make_config("NOT-A-METHOD"), "Unknown method"),
    (lambda: make_config("GEO-RL", regime="primary_no_terminal", stage="A"), "either regime"),
    (lambda: make_config("GEO-RL", urban_risk_lambda=0.5), "no urban reward terms"),
    (lambda: MissionConfig(max_reset_attempts=0), "max_reset_attempts"),
    (lambda: EpisodeConfig(success_radius_m=50.0), "success_radius_m"),
    (lambda: DynamicRiskConfig(temporal_mode="Hann"), "temporal_mode"),
    (lambda: DynamicRiskConfig(hotspot_mode="parks"), "hotspot_mode"),
    (lambda: LidarConfig(n_rays=0), "n_rays"),
])
def test_invalid_configurations_are_rejected_when_built(make, match):
    with pytest.raises((ValueError, TypeError), match=match):
        make()
