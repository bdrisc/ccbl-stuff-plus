# CCBL Stuff+ Model

A location neutral pitch quality model built with Cape Cod Baseball League TrackMan data from the 2025 and 2026 seasons.

This project started with a practical question: **how much does the physical quality of a pitch contribute to its ability to miss bats after removing where it was thrown?** The model estimates that contribution, converts it to a familiar Stuff+ scale, and produces reports that can be used for pitcher evaluation and player development.

The model was developed on 2025 data and evaluated on the untouched 2026 season. A score of **100 represents league average**, and every 10 points represents approximately one standard deviation above or below the relevant CCBL reference group.

[Read the full model breakdown](reports/CCBL_StuffPlus_Model_Report.pdf)

## Key result

During pitcher held out validation on the 2025 development sample, the full pitch characteristics model improved log loss over a context only baseline by **0.00374**. A pitcher clustered bootstrap produced a 95% interval of **0.00149 to 0.00634**, allowing the global publication gate to pass.

That improvement is modest, which is expected when comparing against a baseline that already knows pitch type, handedness, batter side, and plate location. More importantly, the interval remained above zero when entire pitchers were held out. This suggests that the measured pitch characteristics added information beyond context alone rather than simply memorizing the pitchers used for training.

## What the model does

The pipeline predicts the probability of a whiff on a swing using an XGBoost classifier. It compares two models:

* **Context model:** pitch type, pitcher handedness, batter side, and plate location
* **Full model:** the context variables plus velocity, movement, spin, release traits, extension, flight characteristics, and location-adjusted release and approach angles

The full model is then evaluated across a fixed set of plate locations and both batter sides. Averaging those counterfactual predictions removes the effect of the pitch's actual location and creates a location neutral estimate of its bat missing quality.

Those estimates are standardized within pitch type and pitcher hand reference groups on the log odds scale:

* `100` = league average
* `110` = approximately one standard deviation above average
* `90` = approximately one standard deviation below average

Pitcher and arsenal summaries are partially shrunk toward 100 when the sample is small. This prevents a handful of pitches from producing an overly confident ranking.

## Validation design

The validation process was designed to resemble how the model would perform on pitchers and data it had not previously seen.

1. The pipeline trains on swings from the 2025 CCBL season.
2. Five fold `GroupKFold` validation holds out entire pitchers rather than random pitches.
3. Feature preparation and angle adjustments are fitted separately inside each training fold.
4. Cross fitted Platt scaling is used to calibrate predicted whiff probabilities.
5. The full model is compared directly with the context only baseline using log loss, ROC AUC, PR AUC, Brier score, and calibration error.
6. Pitcher clustered bootstrap intervals measure uncertainty in incremental performance.
7. The completed model is evaluated on 2026 as a forward holdout that was not used during development.

## Publication tiers

Pitch groups are not treated as equally reliable. Each pitch type and pitcher hand combination receives a tier based on sample size, out of fold performance, uncertainty, and holdout diagnostics.

|Tier|Pitch groups from this run|
|-|-|
|**Published**|Four-Seam — Left; Four-Seam — Right|
|**Pooled Supported**|Curveball — Left; Sinker — Left/Right; Slider — Left/Right; Splitter — Left; Two-Seam — Right|
|**Rejected**|Changeup — Left/Right; Curveball — Right; Cutter — Left/Right; Splitter — Right; Two-Seam — Left|

`Published` groups passed the strict group level confidence gates. `Pooled Supported` groups had positive estimates under the globally validated model, but their individual confidence intervals were still inconclusive. `Rejected` groups remain available in the scoring output, but their group level results should not be presented as independently validated findings.

Sweepers are included with sliders because the manually reviewed `MyPitchType` field used in this project classifies them as sliders.

## Example outputs

### Individual pitcher report

The pitcher report summarizes overall and pitch specific Stuff+ while showing how each pitch performed throughout the selected outings.

!\[Example individual pitcher Stuff+ report](examples/Pitcher\_Report.png)

### Team ranking

The team graphic compares the top and bottom qualified Falmouth pitchers using shrinkage adjusted overall Stuff+.

!\[Example Falmouth pitcher Stuff+ rankings](examples/Team\_Rankings.png)

## Features

The full model uses measured TrackMan variables and a small number of transparent derived features:

* Release speed and zone speed
* Spin rate and circularly encoded spin axis
* Induced vertical break and horizontal break
* Movement magnitude
* Release height and release side
* Extension and effective velocity
* Velocity loss from release to the plate
* Location adjusted vertical and horizontal release angles
* Location adjusted vertical and horizontal approach angles
* Pitch type, pitcher handedness, batter side, and plate location

The project does not label movement based proxies as measured spin efficiency, active spin, or gyro spin. Those labels were intentionally excluded because the required measurements were not available in the dataset.

## Repository structure

```text
ccbl-stuff-plus/
├── assets/                 # Logos used in the report graphics
├── data/
│   └── raw/                # Private TrackMan workbook; not committed
├── examples/               # Portfolio-ready report graphics
├── notebooks/
│   └── CCBL\_StuffPlus\_Model.ipynb
├── outputs/                # Generated scores and saved models; not committed
├── reports/                # Written model breakdown
├── .gitignore
├── README.md
└── requirements.txt
```

## Running the project

### 1\. Install the dependencies

From the repository root, run:

```bash
pip install -r requirements.txt
```

### 2\. Add the private data locally

Place the combined workbook here:

```text
data/raw/2025 CCBL Trackman Data.xlsx
```

Despite its filename, this workbook contains both 2025 and 2026 pitches. The notebook assigns season using the `Date` column.

The raw data is intentionally excluded from this repository. A user must have an authorized TrackMan export with the required fields to reproduce the complete analysis.

### 3\. Add the report assets

Place the logo files here:

```text
assets/Falmouth\_Commodores\_Logo.png
assets/Dores\_Analytics\_Logo.png
```

### 4\. Run the notebook

Open:

```text
notebooks/CCBL\_StuffPlus\_Model.ipynb
```

Leave `DATA\_FILE\_OVERRIDE = None` when using the repository structure, confirm the run and report settings near the top of the notebook, and select **Run All**.

The pipeline writes model diagnostics, pitch scores, pitcher summaries, calibration tables, feature drift checks, audit tables, and saved model objects to `outputs/`. It writes the two portfolio graphics to `examples/`.

## Limitations

* The model is trained on CCBL data, so its scale and relationships should not be assumed to transfer directly to other leagues.
* Whiff probability captures one important part of pitch quality, but it does not measure called strike value, contact quality, command, sequencing, deception, durability, or pitcher intent.
* Location neutral scoring depends on a defined grid of counterfactual locations rather than every possible game situation.
* Several pitch type and handedness groups did not pass strict group level confidence gates.
* TrackMan classifications and measurements can contain tagging errors or missing values.
* The 2026 holdout revealed calibration or incremental value concerns for some groups, which are retained in the diagnostics instead of being hidden.

Stuff+ should therefore be treated as one piece of a broader evaluation that also includes command, results, video, health, role, and scouting observations.

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Joblib, and OpenPyXL.

## Author

**Brendan Driscoll**  
Baseball Operations \& Analytics  
[GitHub](https://github.com/bdrisc) · [Portfolio](https://evanescent-iris-6be.notion.site/Brendan-Driscoll-s-Portfolio-2b56c3f280ac808ebda4c1b6e66d46ba)

The data used in this project is proprietary and is not included in the public repository.

