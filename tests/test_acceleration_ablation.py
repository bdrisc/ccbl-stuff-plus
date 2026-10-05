"""Test the self-contained notebook without needing proprietary TrackMan data."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "CCBL_StuffPlus_Model.ipynb"


@pytest.fixture
def model_namespace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "notebook_test"}
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "code" and not source.lstrip().startswith("if "):
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


def frame(times):
    n = len(times)
    return pd.DataFrame({"Season": [2025] * n, "PitchType": ["Slider"] * n,
                         "PitcherThrows": ["Right"] * n, "ZoneTime": times,
                         "HorzBreak": [12.0] * n, "InducedVertBreak": [-6.0] * n})


def test_signed_physics_and_equal_time(model_namespace):
    make = model_namespace["prepare_acceleration_features"]
    data = frame([0.4, 0.5])
    # Same acceleration: displacement grows with t squared.
    data.loc[1, ["HorzBreak", "InducedVertBreak"]] *= (0.5 / 0.4) ** 2
    out, audit, reason = make(data)
    assert reason == ""
    np.testing.assert_allclose(out["EquivalentHorzAccel"], [12.5, 12.5])
    np.testing.assert_allclose(out["EquivalentVertAccel"], [-6.25, -6.25])
    np.testing.assert_allclose(out["HB_300ms"], [6.75, 6.75])
    assert out["AccelerationEligible"].all()
    assert audit["ValidRows"].max() == 2


def test_invalid_times_excluded_not_clipped(model_namespace):
    out, _, _ = model_namespace["prepare_acceleration_features"](frame([0, -1, np.inf, np.nan, 0.01, 2, 0.4]))
    assert out["AccelerationEligible"].tolist() == [False] * 6 + [True]
    assert out.loc[:5, "EquivalentHorzAccel"].isna().all()


def test_missing_axis_and_nonfinite_break(model_namespace):
    data = frame([0.4, 0.4, 0.4])
    data.loc[0, "HorzBreak"] = np.inf
    data.loc[1, "InducedVertBreak"] = np.nan
    out, _, _ = model_namespace["prepare_acceleration_features"](data)
    assert out["AccelerationEligible"].tolist() == [False, False, True]


def test_units_and_missing_source(model_namespace):
    make = model_namespace["prepare_acceleration_features"]
    imperial, _, _ = make(frame([0.4]))
    metric = frame([400.0])
    metric[["HorzBreak", "InducedVertBreak"]] *= 2.54
    converted, _, _ = make(metric, flight_time_units="milliseconds", break_units="centimeters")
    np.testing.assert_allclose(converted["EquivalentHorzAccel"], imperial["EquivalentHorzAccel"])
    source = frame([0.4]).drop(columns="ZoneTime")
    source["Time"] = "12:34:56"
    output, audit, reason = make(source)
    assert "Missing" in reason and not audit.empty
    assert output["EquivalentHorzAccel"].isna().all()
    _, _, reason = make(frame([0.4]), compatible_interval_confirmed=False)
    assert "not been confirmed" in reason


def test_feature_ablation_is_exact(model_namespace):
    sets = model_namespace["acceleration_feature_sets"]()
    current = model_namespace["FULL_NUMERIC_FEATURES"]
    assert sets["A_Current"] == current
    assert set(model_namespace["MOVEMENT_FEATURES"]).isdisjoint(sets["B_Acceleration"])
    assert set(model_namespace["ACCELERATION_FEATURES"]).issubset(sets["B_Acceleration"])
    for variant in sets.values():
        assert "RelSpeed" in variant and "Extension" in variant
        assert len(variant) == len(set(variant))
    assert sets["D_CurrentPlusTime"] == current + ["AccelerationFlightTime"]


def test_paired_cluster_gain_sign(model_namespace):
    y = np.array([0, 1, 0, 1] * 3)
    pitchers = np.repeat(["a", "b", "c"], 4)
    good = np.where(y, 0.8, 0.2)
    gain, low, high = model_namespace["paired_pitcher_gain"](y, np.full(len(y), 0.5), good, pitchers, model_namespace["SETTINGS"])
    assert gain > 0 and low > 0 and high > 0
    zero = model_namespace["paired_pitcher_gain"](y, good, good, pitchers, model_namespace["SETTINGS"])
    assert zero == (0.0, 0.0, 0.0)


def synthetic_trackman():
    rng = np.random.default_rng(18)
    rows = []
    for season in [2025, 2026]:
        for pitcher in range(6):
            for pitch in range(24):
                rows.append({"Date": f"{season}-07-01", "Pitcher": f"Pitcher {pitcher}",
                    "PitchUID": f"{season}-{pitcher}-{pitch}", "PitcherThrows": "Right",
                    "BatterSide": "Right" if pitch % 2 else "Left", "MyPitchType": "Slider",
                    "PitchCall": "StrikeSwinging" if pitch % 3 == 0 else "FoulBallFieldable",
                    "RelSpeed": 82 + rng.normal(), "SpinRate": 2400 + rng.normal() * 50,
                    "SpinAxis": 90, "InducedVertBreak": -5 + rng.normal(),
                    "HorzBreak": 12 + rng.normal(), "RelHeight": 6, "RelSide": -2,
                    "Extension": 6, "ZoneSpeed": 75, "EffectiveVelo": 83,
                    "ZoneTime": 0.44 + rng.normal() * 0.01, "PlateLocHeight": rng.uniform(1, 4),
                    "PlateLocSide": rng.uniform(-1, 1), "VertRelAngle": -2 + rng.normal() * .1,
                    "HorzRelAngle": 2 + rng.normal() * .1, "VertApprAngle": -6 + rng.normal() * .1, "HorzApprAngle": 3 + rng.normal() * .1})
    return pd.DataFrame(rows)


def test_full_experiment_common_sample_folds_and_outputs(model_namespace, tmp_path):
    data = synthetic_trackman()
    # Missing flight time excludes the SAME rows from every variant.
    data.loc[[0, 150], "ZoneTime"] = np.nan
    source = tmp_path / "synthetic.xlsx"
    data.to_excel(source, index=False)
    settings = replace(model_namespace["SETTINGS"], max_cv_folds=2, bootstrap_iterations=25,
                       neutral_plate_heights=(2.5,), neutral_plate_sides=(0.0,), min_angle_adjustment_rows=10)
    original_factory = model_namespace["make_model_pipeline"]
    def fast_factory(*args, **kwargs):
        pipeline = original_factory(*args, **kwargs)
        pipeline.set_params(model__n_estimators=8, model__n_jobs=1)
        return pipeline
    model_namespace["make_model_pipeline"] = fast_factory
    output_dir = tmp_path / "comparison"
    result = model_namespace["run_acceleration_ablation"]([source], output_dir, settings=settings)
    assert result["status"] == "Completed"
    output_dir = result["output_directory"]
    prediction = result["predictions"]
    assert len(prediction) == len(data) - 2
    assert prediction.filter(regex="_Raw$").notna().all().all()
    oof = prediction[prediction["Split"] == "2025_OOF"]
    assert oof.groupby("Pitcher")["Fold"].nunique().eq(1).all()
    assert set(oof["Fold"]) == {1, 2}
    assert result["provenance"]["lowest_development_logloss_variant"] in model_namespace["acceleration_feature_sets"]()
    assert result["provenance"]["status"] == "Completed"
    assert "2026_Exploratory" in set(prediction["Split"])
    assert len(result["metrics"].query("Scope == 'Overall'")) == 16
    assert (output_dir / "Acceleration_Ablation.xlsx").exists()
    assert (output_dir / "acceleration_research_models.joblib").exists()
    assert (output_dir / "acceleration_logloss.png").exists()
    # Holdout outcomes cannot influence development predictions or selection.
    altered = data.copy()
    held_mask = altered["Date"].str.startswith("2026")
    altered.loc[held_mask, "PitchCall"] = np.where(
        altered.loc[held_mask, "PitchCall"].eq("StrikeSwinging"),
        "FoulBallFieldable", "StrikeSwinging")
    altered_source = tmp_path / "altered_holdout.xlsx"
    altered.to_excel(altered_source, index=False)
    second = model_namespace["run_acceleration_ablation"]([altered_source], tmp_path / "comparison", settings=settings)
    second_oof = second["predictions"].query("Split == '2025_OOF'")
    np.testing.assert_allclose(oof.filter(regex="_Raw$").to_numpy(), second_oof.filter(regex="_Raw$").to_numpy())
    assert result["provenance"]["lowest_development_logloss_variant"] == second["provenance"]["lowest_development_logloss_variant"]
    assert second["output_directory"] != output_dir
    assert oof["PitchUID"].nunique() == len(oof)
    assert not (tmp_path / "outputs" / "StuffPlus_models.joblib").exists()


def test_missing_time_skips_with_audit(model_namespace, tmp_path):
    source = tmp_path / "missing_time.xlsx"
    synthetic_trackman().drop(columns="ZoneTime").to_excel(source, index=False)
    output_dir = tmp_path / "audit"
    result = model_namespace["run_acceleration_ablation"]([source], output_dir)
    assert result["status"] == "Skipped"
    output_dir = result["output_directory"]
    assert (output_dir / "acceleration_source_audit.csv").exists()
    assert not (output_dir / "Acceleration_Ablation.xlsx").exists()


def test_constant_rank_is_undefined_without_warning(model_namespace):
    assert np.isnan(model_namespace["acceleration_safe_correlation"](pd.Series([0.5, 0.5]), pd.Series([0.1, 0.3])))
