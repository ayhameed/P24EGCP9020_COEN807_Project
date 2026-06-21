# COEN807 Project: Supervised Learning with Linear Regression

This project implements an end-to-end supervised machine learning workflow for river discharge prediction using ESA River Discharge Climate Change Initiative (RD_cci) data already available in this repository.

Author: P24EGCP9020

## Project Objective
Predict daily river discharge (m^3/s) using a linear regression approach with engineered lag and seasonal features. The project demonstrates data loading, preprocessing, model comparison, hyperparameter tuning, evaluation, and reproducible reporting.

## Dataset
- Source path in this repository: `neodc/esacci/river_discharge/data/RD/RD-multi/v1.2/CSV/`
- Dataset name: ESA River Discharge Climate Change Initiative (RD_cci): Multispectral indices-based River Discharge Product, v1.2
- Access record: http://catalogue.ceda.ac.uk/uuid/a8422dd3766c447d8b5fa80920649f31/
- DOI: https://doi.org/10.5285/a8422dd3766c447d8b5fa80920649f31
- License/policy: free and open access (as indicated in dataset metadata)

## Repository Structure
- `scripts/data_loader.py`: station discovery, metadata parsing, CSV loading
- `scripts/preprocessing.py`: feature engineering and temporal split
- `scripts/modeling.py`: baseline linear regression and tuned Ridge comparison
- `scripts/evaluation.py`: metrics and visualization helpers
- `scripts/run_pipeline.py`: reproducible end-to-end pipeline runner
- `scripts/run_multi_station.py`: multi-station runner for comparison tables
- `linear_regression_workflow.ipynb`: markdown-guided workflow notebook
- `results/`: generated metrics, predictions, coefficients, and plots

## Environment Setup
1. Create and activate a Python environment.
2. Install dependencies:
   - `pip install -r requirements.txt`

## Reproducible Execution Workflow
1. Run the script pipeline:
   - `python -m scripts.run_pipeline --station-file neodc/esacci/river_discharge/data/RD/RD-multi/v1.2/CSV/PO_BORGOFORTE_Q_Day.Cmd.csv`
2. Run automatic comparison across 3-5 stations:
   - `python -m scripts.run_multi_station --max-stations 5`
   - Main comparison table: `results/station_comparison.csv`
   - Failure log (if any): `results/station_failures.csv`
3. Open and run notebook from top to bottom:
   - `jupyter notebook linear_regression_workflow.ipynb`
4. Review generated artifacts in `results/`.

## Model Approaches and Evaluation
- Baseline model: Linear Regression
- Comparison model: Ridge Regression with time-series cross-validation tuning
- Metrics: RMSE, MAE, R2
- Outputs:
  - `results/metrics.json`
  - `results/model_comparison.csv`
  - `results/station_comparison.csv`
  - `results/predictions_linear.csv`
  - `results/predictions_ridge.csv`
  - `results/coefficients_linear.csv`
  - `results/plots/*.png`

## References
1. Tarpanelli, A.; Filippucci, P.; Sahoo, D.P. (2024). ESA River Discharge Climate Change Initiative (RD_cci): Multispectral indices-based River Discharge Product, v1.2. NERC EDS Centre for Environmental Data Analysis. DOI: https://doi.org/10.5285/a8422dd3766c447d8b5fa80920649f31
2. CEDA Catalogue Record: http://catalogue.ceda.ac.uk/uuid/a8422dd3766c447d8b5fa80920649f31/
3. scikit-learn Documentation: https://scikit-learn.org/stable/
4. pandas Documentation: https://pandas.pydata.org/docs/
5. NumPy Documentation: https://numpy.org/doc/
6. Matplotlib Documentation: https://matplotlib.org/stable/
