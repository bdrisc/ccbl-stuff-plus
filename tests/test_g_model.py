from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest


def frame(times):
    n = len(times)
    return pd.DataFrame({"Season": [2025] * n, "PitchType": ["Slider"] * n,
                         "PitcherThrows": ["Right"] * n, "ZoneTime": times,
                         "HorzBreak": [12.0] * n, "InducedVertBreak": [-6.0] * n})


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


def fitted_frame():
    data = frame([0.4] * 4)
    data = data.assign(ax0=10.0, ay0=25.0, az0=-27.174, vx0=2.0, vy0=-120.0, vz0=-4.0, y0=50.0)
    return data


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


def test_constant_rank_is_undefined_without_warning(model_namespace):
    assert np.isnan(model_namespace["acceleration_safe_correlation"](pd.Series([0.5, 0.5]), pd.Series([0.1, 0.3])))


def test_manual_labels_only_and_report_tags(model_namespace):
    data = synthetic_trackman().iloc[:4].copy()
    data["MyPitchType"] = ["Slider", "", None, "Other"]
    data["TaggedPitchType"] = "Cutter"
    data["AutoPitchType"] = "Curveball"
    out, _, _ = model_namespace["prepare_data"](data, model_namespace["SETTINGS"], pd.DataFrame())
    assert out["PitchType"].tolist() == ["Slider"]
    assert out["ReportPitchType"].tolist() == ["Cutter"]
    assert out["PitchTypeSource"].eq("MyPitchType").all()
    with pytest.raises(ValueError, match="MyPitchType"):
        model_namespace["prepare_data"](data.drop(columns="MyPitchType"), model_namespace["SETTINGS"], pd.DataFrame())


def test_validation_labels_use_weakest_group(model_namespace):
    code = model_namespace["report_validation_code"]
    assert code(pd.Series(["Published"])) == "V"
    assert code(pd.Series(["Published", "Pooled-Supported"])) == "P"
    assert code(pd.Series(["Published", "Rejected"])) == "D"
    assert code(pd.Series([], dtype=object)) == "D"


def cache_frame(model_namespace):
    return pd.DataFrame({"Season": [2026], "Date": ["2026-07-01"], "Pitcher": ["Test"],
                         "PitchType": ["Slider"], "ReportPitchType": ["Slider"],
                         "PitcherThrows": ["Right"], "PublicationTier": ["Rejected"],
                         "ModelVersion": [model_namespace["MODEL_VERSION"]], "StuffPlusRaw": [99.]})


def test_identical_duplicate_cache_repair(model_namespace, tmp_path):
    one = cache_frame(model_namespace)
    cache = tmp_path / "duplicate.joblib"
    joblib.dump({"model_version": model_namespace["MODEL_VERSION"],
                 "scores": pd.concat([one, one[["PitchType"]]], axis=1)}, cache)
    loaded = model_namespace["load_saved_report_scores"](cache, tmp_path / "absent.xlsx")
    assert loaded.columns.is_unique
    assert joblib.load(cache)["scores"].columns.is_unique
    assert loaded.groupby(["PitchType", "ReportPitchType", "PitcherThrows", "PublicationTier"]).size().sum() == 1
    assert loaded.StuffPlusRaw.iloc[0] == 99


def test_conflicting_duplicate_columns_rejected(model_namespace):
    data = pd.DataFrame([["Slider", "Curveball"]], columns=["PitchType", "PitchType"])
    with pytest.raises(ValueError, match="conflicting values"):
        model_namespace["normalize_report_columns"](data)


@pytest.mark.parametrize("wrong_payload", [True, False])
def test_wrong_or_mixed_cache_versions_rejected(model_namespace, tmp_path, wrong_payload):
    data = cache_frame(model_namespace)
    if not wrong_payload:
        data.loc[0, "ModelVersion"] = "old"
    cache = tmp_path / "wrong.joblib"
    joblib.dump({"model_version": "old" if wrong_payload else model_namespace["MODEL_VERSION"], "scores": data}, cache)
    with pytest.raises(ValueError, match="not from this G version|mixed-version"):
        model_namespace["load_saved_report_scores"](cache, tmp_path / "absent.xlsx")


@pytest.mark.parametrize("column, value", [("ax0", np.nan), ("ZoneTime", 0.), ("HorzBreak", np.inf)])
def test_invalid_g_sources_stop_before_fitting(model_namespace, tmp_path, column, value):
    data = synthetic_trackman()
    data.loc[0, column] = value
    source = tmp_path / "invalid.xlsx"
    data.to_excel(source, index=False)
    out = tmp_path / "scores.xlsx"
    with pytest.raises(ValueError, match="G requires valid"):
        model_namespace["run_pipeline"]([source], out, settings=model_namespace["SETTINGS"])
    assert not out.exists()


def test_missing_g_source_stops(model_namespace, tmp_path):
    source = tmp_path / "missing.xlsx"
    synthetic_trackman().drop(columns="ax0").to_excel(source, index=False)
    with pytest.raises(ValueError, match="ax0"):
        model_namespace["run_pipeline"]([source], tmp_path / "scores.xlsx")


def test_feature_override_rejected(model_namespace, tmp_path):
    with pytest.raises(ValueError, match="frozen feature set"):
        model_namespace["run_pipeline"]([], tmp_path / "scores.xlsx", numeric_features_override=["RelSpeed"])


def test_report_charts_match_table_shrinkage_and_keep_outings(model_namespace, tmp_path, monkeypatch):
    ns = model_namespace
    data = pd.concat([cache_frame(ns)] * 3, ignore_index=True)
    data["StuffPlusRaw"] = [68., 69., 70.]
    data["GameID"] = ["A", "A", "B"]
    data["PitchNo"] = [1, 2, 1]
    plt = ns["plt"]
    captured = []
    close = plt.close
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(plt, "close", lambda fig: captured.append(fig))
    ns["generate_stuffplus_onepage_report"]("Test", data, pd.Timestamp("2026-06-01"),
        pd.Timestamp("2026-08-31"), tmp_path / "report.png", settings=ns["SETTINGS"])
    fig = captured[-1]
    chart = fig.axes[1]
    expected = (data.StuffPlusRaw.to_numpy() * 3 + 3000) / 33
    np.testing.assert_allclose(chart.lines[0].get_ydata(), expected[:2])
    np.testing.assert_allclose(chart.lines[1].get_ydata(), expected[2:])
    np.testing.assert_allclose(chart.lines[2].get_ydata(), [(expected[0] + expected[2]) / 2, expected[1]])
    table = list(fig.axes[0].tables)[0]
    assert table[1, 2].get_text().get_text() == str(round(expected.mean()))
    close(fig)


def test_full_g_pipeline_cache_reports_and_holdout_independence(model_namespace, tmp_path, monkeypatch):
    ns = model_namespace
    settings = replace(ns["SETTINGS"], max_cv_folds=2, bootstrap_iterations=25,
        neutral_plate_heights=(2.5,), neutral_plate_sides=(0.,), min_angle_adjustment_rows=10,
        min_reference_pitches=10)
    factory = ns["make_model_pipeline"]
    def fast_factory(*args, **kwargs):
        model = factory(*args, **kwargs)
        model.set_params(model__n_estimators=12, model__n_jobs=1, model__min_child_weight=1)
        return model
    ns["make_model_pipeline"] = fast_factory
    monkeypatch.setattr(ns["plt"], "show", lambda: None)
    data = synthetic_trackman()
    data["PitcherTeam"] = "FAL_COM"
    data["PitchCall"] = np.where(data.RelSpeed.gt(data.RelSpeed.median()), "StrikeSwinging", "FoulBallFieldable")
    source = tmp_path / "input.xlsx"
    data.to_excel(source, index=False)
    out, bundle, cache = [tmp_path / name for name in ["G.xlsx", "G.joblib", "cache.joblib"]]
    diagnostics, scores, summary, tables = ns["run_pipeline"]([source], out, bundle, cache, settings=settings)
    assert all(path.exists() for path in [out, bundle, cache])
    assert len(scores) == len(data)
    assert scores.ModelVersion.eq(ns["MODEL_VERSION"]).all()
    assert set(ns["G_ADDED_FEATURES"]).issubset(scores)
    assert scores.StuffPlusRaw.notna().all()
    assert scores.PitchUID.is_unique
    loaded = ns["load_saved_report_scores"](cache, out)
    assert loaded.columns.is_unique
    assert len(loaded) == len(data)
    np.testing.assert_allclose(loaded.StuffPlusRaw, scores.StuffPlusRaw)
    workbook_loaded = ns["load_saved_report_scores"](tmp_path / "absent.joblib", out)
    np.testing.assert_allclose(workbook_loaded.StuffPlusRaw, scores.StuffPlusRaw)
    folds = pd.read_excel(out, sheet_name="Pitcher Folds")
    assert folds.groupby("Pitcher").Fold.nunique().eq(1).all()
    assert set(folds.Fold) == {1, 2}
    report = tmp_path / "Pitcher_Report_G.png"
    ns["generate_stuffplus_onepage_report"]("Pitcher 0", loaded, pd.Timestamp("2026-06-10"),
        pd.Timestamp("2026-08-12"), report, settings=settings)
    team = tmp_path / "Team_Rankings_G.png"
    ranking = ns["generate_team_pitcher_stuffplus_graphic"](scores=loaded, team_code="FAL_COM",
        team_display_name="Falmouth Commodores", season=2026, save_path=team, min_pitches=20,
        start_date=pd.Timestamp("2026-06-10"), end_date=pd.Timestamp("2026-08-12"), settings=settings)
    assert report.exists() and team.exists() and len(ranking) == 6
    altered = data.copy()
    mask = altered.Date.str.startswith("2026")
    altered.loc[mask, "PitchCall"] = np.where(altered.loc[mask, "PitchCall"].eq("StrikeSwinging"),
                                            "FoulBallFieldable", "StrikeSwinging")
    changed = tmp_path / "changed.xlsx"
    altered.to_excel(changed, index=False)
    _, second, _, _ = ns["run_pipeline"]([changed], tmp_path / "changed_scores.xlsx", settings=settings)
    a = scores.query("Season == 2025").sort_values("PreparedRow")
    b = second.query("Season == 2025").sort_values("PreparedRow")
    np.testing.assert_allclose(a[["StuffPlusRaw", "NeutralWhiffProb", "FullWhiffProb"]],
                               b[["StuffPlusRaw", "NeutralWhiffProb", "FullWhiffProb"]])
