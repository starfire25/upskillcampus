# Agriculture Crop Production Prediction in India

**UCT / UpSkill Campus Free Summer Internship — Data Science & Machine Learning**
**Student:** Diya | **Project:** Project 4

---

## Problem Statement
Predict crop production (in tonnes) across Indian states and districts based on historical data from 2001–2014, using features such as crop type, season, state, and cultivated area.

## Dataset
- **Source:** data.gov.in — District-wise Season-wise Crop Production Statistics
- **Period:** 2001–2014
- **Key Columns:** State_Name, District_Name, Crop_Year, Season, Crop, Area, Production

## Project Structure
```
upskillcampus/
├── AgricultureCropProduction_Diya_USC_UCT.py   ← Main code file
├── AgricultureCropProduction_Diya_USC_UCT.pdf  ← Project report (upload after generating)
└── README.md
```

## ML Models Used
| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2 regularisation |
| Lasso Regression | L1 regularisation + feature selection |
| Random Forest | Primary model (best performance) |
| Gradient Boosting | Comparison |

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python AgricultureCropProduction_Ankit_USC_UCT.py
```

## Output Files Generated
- `eda_production_distribution.png`
- `eda_top_states.png`
- `eda_season_production.png`
- `eda_correlation_heatmap.png`
- `model_comparison.png`
- `actual_vs_predicted.png`
- `feature_importance.png`
- `model_results_summary.csv`

## GitHub Links
- **Code:** `[https://github.com/starfire25/upskillcampus/blob/main/AgricultureCropProduction_Diya_USC_UCT.pdf]`
- **Report:** `[https://github.com/Ankit/upskillcampus/blob/main/AgricultureCropProduction_Ankit_USC_UCT.pdf]`
