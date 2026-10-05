# G model definitions and validation

`CCBL-StuffPlus-G-v1.0` adds two features to the original full model: `AccelerationFlightTime` and `FittedDragDeceleration`. Velocity, spin, movement, release, extension, effective velocity, speed loss, and adjusted release/approach angles remain in the feature set. The production notebook freezes this set and rejects a different feature override.

## Flight time

`AccelerationFlightTime` is TrackMan `ZoneTime` in seconds, describing release to the front of home plate. It is not the clock timestamp `Time`, a speed-derived estimate, or the fitted trajectory interval described below.

The helper also calculates equivalent signed acceleration as `2 * (break_inches / 12) / ZoneTime**2`. Those equivalent acceleration components, fixed-time deflections, and fitted transverse acceleration features are retained for source diagnostics but **are not G model inputs**.

## Fitted deceleration

The legacy trajectory fields use x lateral, y toward the mound, and z upward. Pitch velocity toward the plate has negative `vy0`. Acceleration is in ft/s², velocity in ft/s, and position in feet. The `PitchTrajectoryX/Y/Z` polynomial fields have different coordinate conventions and are not substitutes.

For acceleration vector `a = [ax0, ay0, az0]` and initial velocity `v0 = [vx0, vy0, vz0]`, the helper solves the y trajectory from `y0` to the front of home plate at y = 17/12 ft. With distance `d = y0 - 17/12`, travel time is:

```text
t = 2*d / (-vy0 + sqrt(vy0**2 - 2*ay0*d))
v_mid = v0 + a*t/2
u = v_mid / norm(v_mid)
a_air = a + [0, 0, 32.174]
FittedDragDeceleration = -dot(a_air, u)
```

Positive deceleration means slowing along the midpoint velocity direction. Transverse acceleration is `a_air - dot(a_air, u)*u`; its orthogonality residual and correlations with movement are audit diagnostics. The native x sign is retained.

This is a gravity-adjusted projection of average fitted acceleration, not an independently measured instantaneous force or pure Magnus measurement. `ZoneTime` and this fitted interval describe different intervals and are not interchangeable.

Input validation requires finite legacy fields, plate-directed velocity, distance between 0 and 65 ft, positive fitted travel time below 1 second, and midpoint speed between 50 and 200 ft/s. The time/movement helper requires `ZoneTime` between 0.15 and 1.0 seconds and finite movement on both axes. G stops if any prepared pitch fails either eligibility check. Negative deceleration is audited, not clipped.

Source references:

- [TrackMan V3 glossary](https://support.trackmanbaseball.com/hc/en-us/articles/5089413493787-Radar-Glossary-Of-V3-Terms)
- [Alan Nathan, pitch movement and acceleration](https://baseball.physics.illinois.edu/Movement.pdf)
- [Tango, math behind vertical movement](https://tangotiger.com/index.php/site/comments/math-behind-vertical-movement)

## Development evidence

The fixed-sample comparison used manually reviewed `MyPitchType` only, 53,927 development pitches and 56,211 exploratory 2026 pitches, with 23,811 and 24,569 swings respectively. This restored the original sample after an earlier experiment tried automatic label fallback.

| Model | 2025 raw actual log loss | 2026 raw actual log loss |
| --- | ---: | ---: |
| Original full model | 0.491337786 | 0.504649843 |
| G: original + time + fitted deceleration | 0.490178536 | 0.502321935 |
| Original + fitted deceleration only | 0.490998999 | 0.502465245 |
| G + fitted transverse acceleration | 0.490406556 | 0.502780579 |

For G versus the original model, positive log-loss gains were 0.001159251 in 2025 and 0.002327908 in 2026. The fixed-sample experiment's Bonferroni-adjusted individual 98.75% intervals for four primary contrasts per season were [0.000267183, 0.002073101] and [0.001411747, 0.003322453]. Adding transverse acceleration to G did not produce a positive overall point gain in either season.

The packaged G run's 2025 gain against the **context-only** model was 0.004901, with a pitcher-clustered 95% interval [0.002571, 0.007429]. Actual-location raw AUC was 0.745402 in 2025 and 0.747197 in 2026. Neutral raw AUC was 0.611786 and 0.610763. Actual and neutral diagnostics answer different questions and should not be conflated.

Three further pitcher-fold assignments produced G-versus-original point gains of 0.001159, 0.000346, and 0.000817. Two had positive 95% intervals. Because these reuse the same data, they test sensitivity to partition choice, not independence.

G showed small increases in raw calibration error relative to the original model, so the log-loss improvement should not be described as improvement in every diagnostic. Group tier counts are 3 Published, 5 Pooled-Supported, and 8 Rejected. Overall added-feature drift was stable; some small groups had drift or calibration review flags.

The repository tests establish software behavior on synthetic data. The numerical results above come from the private executed comparison and packaged workbooks; they cannot be independently reproduced without authorized source data.

## Reports and reproducibility

Each run saves the workbook, models, versioned report cache, and manifest in a new timestamped folder. The manifest records source hashes, model settings, features, publication tier counts, and runtime versions. `RUN_MODEL=False` reuses a selected run. Mixed model versions are blocked; identical duplicate cache columns are repaired, while conflicting copies are rejected.

Reports can group by a display tag different from the manual model label. V/P/D uses the weakest validation tier in that displayed group. All tiers remain scored; D is diagnostic only.

For a report category with n pitches, each chart point is adjusted by `(raw*n + 100*30)/(n+30)`, matching the category's table mean before rounding. Individual outings and the combined curve remain in the chart. The combined curve averages available pitches at the same within-outing pitch number, so its plotted-point average need not equal the pitch-weighted table average.

The headline and team ranking pool the raw scores and then shrink once. The workbook summary instead weights already-shrunk arsenal categories. Keep those definitions explicit when comparing a report headline to `OverallStuffPlus` in the workbook.

2026 is exploratory because it was examined during candidate comparisons. Freezing G and evaluating genuinely new data is the next independent test.
