# i. Title and Abstract

## Title
Machine Learning for Real-World Data Analytics: Linear Regression for Satellite-Derived River Discharge Prediction

## Abstract
This project develops a supervised learning pipeline to predict river discharge (m^3/s) from ESA RD_cci station time series data using linear regression methods. The workflow covers data ingestion, preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and comparative analysis across multiple river stations. Two models are compared: Linear Regression (baseline) and Ridge Regression (regularized variant). Results are evaluated with RMSE, MAE, and R2 on chronological holdout sets to avoid temporal leakage. The implementation is reproducible through script-based execution and a notebook walkthrough. Author: P24EGCP9020.

[Image Placeholder: Project overview diagram]

# ii. Introduction and Problem Definition

Reliable river discharge estimation supports flood preparedness, water resources management, and climate impact analysis. This project addresses the problem of predicting daily discharge values from historical station-level RD_cci records. The core objective is to build an interpretable supervised regression pipeline and evaluate whether regularization improves generalization over a standard linear baseline.

Problem statement:
- Input: historical station discharge series and engineered temporal features.
- Output: predicted daily discharge at time t.
- Task type: supervised regression.

# iii. Dataset Description and Justification

Dataset used:
- ESA River Discharge Climate Change Initiative (RD_cci), v1.2
- Local data path: `neodc/esacci/river_discharge/data/RD/RD-multi/v1.2/CSV/`
- Access record: http://catalogue.ceda.ac.uk/uuid/a8422dd3766c447d8b5fa80920649f31/
- DOI: https://doi.org/10.5285/a8422dd3766c447d8b5fa80920649f31
- Policy: free and open access (as stated in metadata)

Why this dataset is appropriate:
- Real-world environmental data from a credible public scientific source.
- Continuous target variable suited for regression.
- Multi-station structure enables comparative experiments.
- Rich temporal span supports lag-based and seasonality feature design.

[Image Placeholder: Dataset coverage map or station distribution figure]

# iv. Data Preprocessing and Feature Engineering

Preprocessing steps implemented in scripts:
- Parse semicolon-delimited CSV files and station metadata headers.
- Convert date/time fields into a datetime index.
- Sort records chronologically and remove duplicate timestamps.
- Preserve rows with valid target discharge values.

Feature engineering:
- Lag features: 1, 2, 7, and 30 previous observations.
- Rolling statistics: mean and standard deviation (7 and 30 windows).
- Calendar features: month, day_of_year, quarter.

Data splitting strategy:
- Chronological train/test split (default test fraction = 0.2).
- No random shuffle to prevent future information leakage.

[Image Placeholder: Feature engineering flowchart]

# v. Methodology and Model Selection

Models:
- Linear Regression as the baseline interpretable model.
- Ridge Regression as a regularized alternative.

Model selection rationale:
- Linear Regression provides transparent coefficients and clear interpretation.
- Ridge helps reduce overfitting by penalizing large coefficients.

Tuning:
- Ridge alpha tuned using TimeSeriesSplit cross-validation.
- Candidate alphas: 0.01, 0.1, 1.0, 10.0, 100.0.

Implementation modules:
- `scripts/data_loader.py`
- `scripts/preprocessing.py`
- `scripts/modeling.py`
- `scripts/evaluation.py`
- `scripts/run_pipeline.py`
- `scripts/run_multi_station.py`

# vi. Experimental Design

Experimental setup:
1. Single-station baseline run for detailed diagnostics.
2. Multi-station batch run (3 to 5 stations) for comparative analysis.
3. Evaluate both models with identical train/test partitions per station.

Stations used in this report run:
- Number of stations used: 5 (from a total of 51 available station CSV files).
- Selected stations: AM-TIMAN, LAI, SAO-FELIPE, MANACAPURU, OBIDOS.
- Selection method: deterministic script-based subset with `--max-stations 5` for reproducible course-scope comparison.

Why 5 stations were used:
- Meets the project requirement for multi-experiment comparison while keeping runtime practical for iterative notebook and report updates.
- Provides geographic and hydrological diversity (Chad and Amazon basins) without over-expanding the report scope.
- Supports clear station-level analysis tables and figures suitable for a 12-page report and 10-15 slide presentation.

Reproducible commands:
1. `pip install -r requirements.txt`
2. `python -m scripts.run_pipeline --station-file neodc/esacci/river_discharge/data/RD/RD-multi/v1.2/CSV/PO_BORGOFORTE_Q_Day.Cmd.csv`
3. `python -m scripts.run_multi_station --max-stations 5`
4. `jupyter notebook linear_regression_workflow.ipynb`

Primary artifacts:
- `results/metrics.json`
- `results/model_comparison.csv`
- `results/station_comparison.csv`
- `results/station_failures.csv`
- `results/predictions_linear.csv`
- `results/predictions_ridge.csv`
- `results/coefficients_linear.csv`
- `results/plots/*.png`

# vii. Results and Evaluation

Evaluation metrics:
- RMSE (error magnitude)
- MAE (average absolute error)
- R2 (variance explained)

Observed outcome summary:
- The generated outputs indicate strong predictive performance on tested stations.
- Example single-station run produced high R2 and low error relative to flow scale.
- Multi-station results are consolidated in `results/station_comparison.csv`.
- Station comparison table contains 5 successful stations and 0 failures for this run.

[Image Placeholder: Actual vs predicted plot]
[Image Placeholder: Residual plot]

# viii. Comparative Analysis and Discussion

Comparison dimensions:
- Linear Regression vs Ridge Regression per station.
- Station-level variation in RMSE/MAE/R2.
- Impact of regularization strength (best alpha from CV).

Discussion points:
- Performance differences vary by station dynamics and variance structure.
- Ridge may slightly improve generalization in some stations, while baseline linear may remain competitive in others.
- Cross-station comparison provides stronger evidence than single-station reporting.

[Image Placeholder: Station-level RMSE comparison chart]

# ix. Limitations, Biases, and Lessons Learned

Limitations:
- Satellite-derived products include method-dependent uncertainty.
- Temporal density and quality can vary by station.
- Linear relationships may underfit nonlinear hydrological behavior.

Potential biases:
- Basin/station representation bias in selected subset.
- Model bias due to limited feature set (no external meteorological variables).

Lessons learned:
- Time-aware splitting is essential for valid evaluation.
- Reusable script modules improve reproducibility and maintainability.
- Multi-station comparisons strengthen reporting quality for academic deliverables.

# x. Conclusion and Future Work

This project demonstrates a complete, reproducible supervised learning workflow for real-world river discharge prediction using linear models. The pipeline satisfies course requirements for data preparation, model comparison, tuning, and evaluation.

Future work:
1. Expand experiments to all available stations.
2. Add external predictors (rainfall, temperature, basin indicators).
3. Compare with nonlinear models and ensembles.
4. Add uncertainty quantification and calibration diagnostics.

# xi. References

1. Tarpanelli, A.; Filippucci, P.; Sahoo, D.P. (2024). ESA River Discharge Climate Change Initiative (RD_cci): Multispectral indices-based River Discharge Product, v1.2. NERC EDS Centre for Environmental Data Analysis. DOI: https://doi.org/10.5285/a8422dd3766c447d8b5fa80920649f31
2. CEDA Catalogue Record: http://catalogue.ceda.ac.uk/uuid/a8422dd3766c447d8b5fa80920649f31/
3. scikit-learn Documentation: https://scikit-learn.org/stable/
4. pandas Documentation: https://pandas.pydata.org/docs/
5. NumPy Documentation: https://numpy.org/doc/
6. Matplotlib Documentation: https://matplotlib.org/stable/

# xii. Appendix (optional)

## A. Repository Structure
- `scripts/`: reusable implementation modules.
- `linear_regression_workflow.ipynb`: walkthrough notebook using script functions.
- `results/`: generated metrics, tables, and plots.
- `neodc/.../CSV/`: source station data files.

## B. Notes for Report and PPT
- Use `results/station_comparison.csv` for table-style comparison in the technical report.
- Use plots in `results/plots/` for slides.
- Keep all screenshots/figures replaced from placeholders before final submission.
