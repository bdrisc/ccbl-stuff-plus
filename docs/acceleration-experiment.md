# Acceleration versus movement experiment

The notebook now tests whether time-normalized movement improves whiff-on-swing prediction. Existing Stuff+ scores, model bundles, publication tiers, and pitcher reports retain the current model. Experimental outputs live in a separate, timestamped folder.

## Run locally

1. Pull the repository updates and open `notebooks/CCBL_StuffPlus_Model.ipynb`.
2. Keep your combined 2025/2026 private workbook in the existing location, or set `DATA_FILE_OVERRIDE` as before.
3. Run all cells. `RUN_ACCELERATION_EXPERIMENT = True` enables the experiment. The original `RUN_MODEL` option still controls the existing Stuff+ pipeline. To rerun only the comparison after generating the original reports once, set `RUN_MODEL = False`.
4. Inspect the printed metrics and `outputs/acceleration_ablation/run_<UTC timestamp>/Acceleration_Ablation.xlsx`.

No new modeling dependency is required. Tests use `python -m pip install -r requirements-dev.txt`, then `python -m pytest tests -q`.

## Physics and input data

The default source is TrackMan `ZoneTime` in seconds, with `HorzBreak` and `InducedVertBreak` in inches. TrackMan defines these over release to the front of home plate. Field definitions: https://support.trackmanbaseball.com/hc/en-us/articles/5089413493787-Radar-Glossary-Of-V3-Terms

For each signed movement component, equivalent acceleration in ft/s² is:

`2 * (break_inches / 12) / ZoneTime**2`

This is an equivalent constant acceleration derived from movement and flight time. It is not an independently measured acceleration, pure Magnus acceleration, or proof of late break. It does not correct drag. IVB has gravity removed by definition; do not add gravity again to this derived vertical feature.

Signed horizontal and vertical components preserve direction. The magnitude is supplemental, not a rule that bigger values mean better pitches. `HB_300ms` and `IVB_300ms` give equivalent fixed-time deflections in inches; they are not the actual movement accumulated during the last 300 ms of flight.

Custom time columns must use the same interval as the break measurements. Configure `ACCELERATION_FLIGHT_TIME_COLUMN`, `ACCELERATION_FLIGHT_TIME_UNITS` (`seconds` or `milliseconds`), and `ACCELERATION_BREAK_UNITS` (`inches` or `centimeters`). Set `ACCELERATION_COMPATIBLE_INTERVAL = False` until a custom export's definitions are verified. The rest of the existing model expects imperial TrackMan data; the break-unit setting applies only to acceleration derivation, not to a wholesale conversion of the original pipeline.

The timestamp `Time` is never treated as flight time. No velocity-based estimate is substituted. Missing columns, unconfirmed intervals, or completely invalid measurements produce a source audit and a skipped-run manifest. Invalid flight times outside 0.15–1.0 seconds, nonfinite break, and missing axes are excluded rather than clipped. The source audit lists candidate input columns to help diagnose an export.

## Comparisons

| Variant | Numeric feature change |
| --- | --- |
| A_Current | Original model features |
| B_Acceleration | Replace IVB, HB, and movement magnitude with signed equivalent accelerations and their magnitude |
| C_Combined | Original features plus the three acceleration features |
| D_CurrentPlusTime | Original features plus valid flight time |

D is a timing control: C versus D tests whether the acceleration representation helps beyond simply adding the time used to calculate it. Acceleration is a deterministic transformation of break and time, so a gain does not establish a new independent physical signal.

Velocity, spin, release traits, extension, and location-adjusted angles stay in all variants. All four use the same complete-case swings, the same pitcher-held-out 2025 folds, the same XGBoost settings, and fold-fitted angle adjustments and preprocessing. This common sample can differ from the full published model's sample: read coverage before generalizing.

## Interpret the results

- **Metrics:** Lower raw log loss is the primary criterion; Brier score, AUC, PR AUC, and calibration error provide context. Overall and pitch-type/hand results are reported separately.
- **Paired Gains:** Positive `LogLossGain` favors the candidate over its stated baseline. The pitcher-clustered interval accounts for multiple pitches from the same pitcher. Intervals are unadjusted across exploratory comparisons; a positive interval is a candidate for investigation, not an automatic publication gate.
- **Rank Stability:** Spearman correlation compares pitcher rankings by average location-neutral whiff probability on the comparison swings. These are diagnostic research rankings, not newly validated published Stuff+ scores.
- **Sample Coverage / Source Audit:** Show the original sample, eligible sample, and missing/invalid source measurements by season and pitch group.

The lowest development raw log loss is recorded before any 2026 model evaluation. Neither model selection nor calibration fitting uses 2026 outcomes. Because 2026 results have already been examined for the prior model, this evaluation is labeled exploratory; fresh data is needed for independent confirmation.

Calibrated OOF metrics use the existing cross-fitted Platt procedure as a secondary diagnostic. This meta-level cross-fitting is not a fully nested base-model/calibrator evaluation, so it is not used for candidate selection. Holdout calibration is fitted only on 2025 OOF predictions.

## Saved outputs

Each run writes its own directory, preserving previous comparisons and avoiding confusion after a skipped run:

- `Acceleration_Ablation.xlsx`: metrics, paired gains, ranking stability, coverage, source audit, folds, features, and shape summaries.
- `paired_predictions.csv`: common-sample raw/calibrated/neutral predictions with prepared row indices and original pitch identifiers where available.
- `development_raw_metrics.csv` and `pitcher_folds.csv`: reproducible development evidence.
- `acceleration_experiment.json`: feature definitions, data hashes, settings, source definitions, development choice, limitations, and status.
- `acceleration_research_models.joblib`: all four research models, calibrators, and final training angle adjusters.
- `acceleration_logloss.png`: overview of raw comparison log loss.

All player-level data and generated outputs remain excluded from Git. The experiment does not overwrite the original score/report artifacts or automatically promote a winning variant.

## Workbooks without MyPitchType

The loader prefers `MyPitchType`, then fills absent/blank labels from `TaggedPitchType`, then `AutoPitchType`. Nonblank reviewed labels are never overridden; unsupported reviewed labels still follow the existing eligibility exclusions. If no pitch-type source exists, the loader lists the available columns and stops with an actionable error.

Fallback classifications produce an explicit warning. The original validation workbook's data audit and the experiment's `Data Audit` sheet and provenance JSON record source counts. The four variants still use the same classifications and samples within a run, but results from an automatically tagged workbook should not be treated as directly interchangeable with earlier manually classified results. Review labels before publication.
