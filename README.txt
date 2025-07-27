# T2DM Hybrid Model Repository
This repository contains code and synthetic data for the *PLOS ONE* manuscript: "Fuzzy Logic and RFE for T2DM Prediction in Adults 35–45". License: MIT (https://opensource.org/licenses/MIT).

## Contents
- `t2dm_hybrid_model.py`: Implements the hybrid fuzzy logic, RFE, and logistic regression model.
- `generate_synthetic_data.py`: Generates synthetic data using CTGAN, with validation.
- `synthetic_t2dm.csv`: Synthetic dataset (5,000 samples, NHANES-based).
- `validation_results.csv`: Validation metrics (KS p>0.05, KL divergence <0.05).

## Datasets
- **PIDD**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **NHANES**: https://www.cdc.gov/nchs/nhanes/index.htm
- **IPDD**: https://data.mendeley.com/datasets/6mm33v7kzt/1
- **UK Biobank**: https://www.ukbiobank.ac.uk (application required)

## Requirements
- Python 3.8+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `skfuzzy`, `imblearn`, `ctgan`, `scipy`, `matplotlib`
- Install: `pip install -r requirements.txt`

## Usage
1. **Preprocessing and Model Training**:
   ```bash
   python t2dm_hybrid_model.py