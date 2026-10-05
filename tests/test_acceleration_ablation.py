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
                    "ax0": 10 + rng.normal(), "ay0": 25, "az0": -35 + rng.normal(),
                    "vx0": 2, "vy0": -120, "vz0": -4, "y0": 50,
                    "ZoneTime": 0.44 + rng.normal() * 0.01, "PlateLocHeight": rng.uniform(1, 4),
                    "PlateLocSide": rng.uniform(-1, 1), "VertRelAngle": -2 + rng.normal() * .1,
                    "HorzRelAngle": 2 + rng.normal() * .1, "VertApprAngle": -6 + rng.normal() * .1, "HorzApprAngle": 3 + rng.normal() * .1})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("include_fitted", [False, True])
def test_full_experiment_common_sample_folds_and_outputs(model_namespace, tmp_path, include_fitted):
    data = synthetic_trackman()
    # Missing flight time excludes the SAME rows from every variant.
    data.loc[[0, 150], "ZoneTime"] = np.nan
    if include_fitted:
        data.loc[2, "az0"] = np.nan
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
    result = model_namespace["run_acceleration_ablation"]([source], output_dir, settings=settings, include_fitted=include_fitted)
    assert result["status"] == "Completed"
    output_dir = result["output_directory"]
    prediction = result["predictions"]
    assert len(prediction) == len(data) - 2 - int(include_fitted)
    assert prediction.filter(regex="_Raw$").notna().all().all()
    oof = prediction[prediction["Split"] == "2025_OOF"]
    assert oof.groupby("Pitcher")["Fold"].nunique().eq(1).all()
    assert set(oof["Fold"]) == {1, 2}
    assert result["provenance"]["lowest_development_logloss_variant"] in model_namespace["acceleration_feature_sets"](include_fitted)
    assert result["provenance"]["status"] == "Completed"
    assert "2026_Exploratory" in set(prediction["Split"])
    assert len(result["metrics"].query("Scope == 'Overall'")) == (28 if include_fitted else 16)
    assert (output_dir / "Acceleration_Ablation.xlsx").exists()
    assert (output_dir / "acceleration_research_models.joblib").exists()
    assert (output_dir / "acceleration_logloss.png").exists()
    if include_fitted:
        assert (output_dir / "fitted_pitch_features.csv").exists()
        pairs = result["comparisons"]
        assert ((pairs.Baseline == "G_TimeDragControl") & (pairs.Candidate == "F_FittedCombined")).any()
    # Holdout outcomes cannot influence development predictions or selection.
    altered = data.copy()
    held_mask = altered["Date"].str.startswith("2026")
    altered.loc[held_mask, "PitchCall"] = np.where(
        altered.loc[held_mask, "PitchCall"].eq("StrikeSwinging"),
        "FoulBallFieldable", "StrikeSwinging")
    altered_source = tmp_path / "altered_holdout.xlsx"
    altered.to_excel(altered_source, index=False)
    second = model_namespace["run_acceleration_ablation"]([altered_source], tmp_path / "comparison", settings=settings, include_fitted=include_fitted)
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


@pytest.mark.parametrize("column", ["TaggedPitchType", "AutoPitchType"])
def test_missing_manual_pitch_type_uses_available_source(model_namespace, column):
    data = synthetic_trackman().rename(columns={"MyPitchType": column})
    with pytest.warns(UserWarning, match="MyPitchType is absent or blank"):
        prepared, audit, _ = model_namespace["prepare_data"](data, model_namespace["SETTINGS"], pd.DataFrame())
    assert len(prepared) == len(data)
    assert prepared["PitchType"].eq("Slider").all()
    assert prepared["PitchTypeSource"].eq(column).all()
    assert column in set(audit["Detail"])


def test_manual_priority_blank_fallback_and_unsupported_labels(model_namespace):
    data = synthetic_trackman().iloc[:4].copy()
    data["TaggedPitchType"] = "Curveball"
    data["AutoPitchType"] = "Sinker"
    data["MyPitchType"] = ["Slider", "", None, "Other"]
    with pytest.warns(UserWarning):
        prepared, _, _ = model_namespace["prepare_data"](data, model_namespace["SETTINGS"], pd.DataFrame())
    assert prepared["PitchType"].tolist() == ["Slider", "Curveball", "Curveball"]
    assert prepared["PitchTypeSource"].tolist() == ["MyPitchType", "TaggedPitchType", "TaggedPitchType"]


def test_no_pitch_type_source_is_actionable(model_namespace):
    data = synthetic_trackman().drop(columns="MyPitchType")
    with pytest.raises(ValueError, match="No pitch-type source was found"):
        model_namespace["prepare_data"](data, model_namespace["SETTINGS"], pd.DataFrame())


def fitted_frame():
    data = frame([0.4] * 4)
    data = data.assign(ax0=10.0, ay0=25.0, az0=-27.174, vx0=2.0, vy0=-120.0, vz0=-4.0, y0=50.0)
    return data


def test_fitted_gravity_projection_and_coordinate_sign(model_namespace):
    data = fitted_frame()
    data.loc[0, ["ax0", "ay0", "az0"]] = [0, 0, -32.174]
    data.loc[1, "ax0"] = -10
    out, audit, reason = model_namespace["prepare_fitted_acceleration_features"](data)
    assert reason == "" and out["FittedEligible"].all()
    np.testing.assert_allclose(out.loc[0, model_namespace["FITTED_ACCELERATION_FEATURES"]].astype(float), 0, atol=1e-12)
    assert out.loc[1, "FittedTransverseX"] < 0
    a = data[["ax0", "ay0", "az0"]].to_numpy()
    v = data[["vx0", "vy0", "vz0"]].to_numpy() + a * out["FittedIntervalTime"].to_numpy()[:, None] / 2
    unit = v / np.linalg.norm(v, axis=1)[:, None]
    aero = a + [0, 0, 32.174]
    parallel = (aero * unit).sum(axis=1)
    expected = aero - parallel[:, None] * unit
    np.testing.assert_allclose(out["FittedTransverseX"], expected[:, 0])
    np.testing.assert_allclose(out["FittedTransverseZ"], expected[:, 2])
    np.testing.assert_allclose(out["FittedDragDeceleration"], -parallel)
    np.testing.assert_allclose(out["FittedOrthogonalityResidual"], 0, atol=1e-12)
    t = out["FittedIntervalTime"]
    np.testing.assert_allclose(data.y0 + data.vy0 * t + .5 * data.ay0 * t**2, 17/12)
    assert audit["ValidRows"].sum() == len(data)


def test_fitted_invalid_and_missing_sources(model_namespace):
    data = fitted_frame()
    data.loc[0, "vx0"] = np.inf
    data.loc[1, "vy0"] = 120
    data.loc[2, "y0"] = 0
    out, _, reason = model_namespace["prepare_fitted_acceleration_features"](data)
    assert not reason
    assert out["FittedEligible"].tolist() == [False, False, False, True]
    _, _, reason = model_namespace["prepare_fitted_acceleration_features"](data.drop(columns="az0"))
    assert "az0" in reason


def test_fitted_missing_columns_skips_explicitly(model_namespace, tmp_path):
    source = tmp_path / "without_fitted.xlsx"
    synthetic_trackman().drop(columns="ax0").to_excel(source, index=False)
    result = model_namespace["run_acceleration_ablation"]([source], tmp_path / "comparison", include_fitted=True)
    assert result["status"] == "Skipped"
    assert "ax0" in result["reason"]
    assert (result["output_directory"] / "fitted_acceleration_audit.csv").exists()
