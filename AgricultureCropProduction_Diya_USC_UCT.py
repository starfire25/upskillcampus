# =============================================================================
# Project 4: Prediction of Agriculture Crop Production in India
# Internship: Free Summer Internship in Data Science & Machine Learning
# Organization: UpSkill Campus (USC) / UniConverge Technologies Pvt Ltd (UCT)
# Student: Ankit
# File: AgricultureCropProduction_Diya_USC_UCT.py
# =============================================================================

# --------------------------------------------------------------------------
# SECTION 1: IMPORTS
# --------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

print("=" * 65)
print("  Prediction of Agriculture Crop Production in India")
print("  UCT / UpSkill Campus Internship | Student: Ankit")
print("=" * 65)

# --------------------------------------------------------------------------
# SECTION 2: LOAD DATA
# --------------------------------------------------------------------------
# Dataset source: data.gov.in — Agriculture Crop Production (2001–2014)
# Download from: https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics
# Expected columns: State_Name, District_Name, Crop_Year, Season, Crop,
#                   Area, Production

print("\n[1] Loading Dataset...")
# --- REPLACE THIS PATH with your actual CSV file path ---
DATA_PATH = "crop_production.csv"

try:
    df = pd.read_csv(DATA_PATH)
    print(f"    Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
except FileNotFoundError:
    # Generate a realistic synthetic sample for demonstration
    print("    [INFO] Dataset file not found. Generating synthetic sample for demonstration.")
    np.random.seed(42)
    n = 2000
    states = ['Andhra Pradesh', 'Karnataka', 'Maharashtra', 'Punjab', 'Uttar Pradesh',
              'Tamil Nadu', 'Madhya Pradesh', 'Rajasthan', 'Gujarat', 'West Bengal']
    seasons = ['Kharif', 'Rabi', 'Whole Year', 'Summer', 'Winter']
    crops = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton', 'Pulses',
             'Groundnut', 'Soyabean', 'Sunflower', 'Jowar']
    df = pd.DataFrame({
        'State_Name':    np.random.choice(states, n),
        'District_Name': [f"District_{i%50}" for i in range(n)],
        'Crop_Year':     np.random.randint(2001, 2015, n),
        'Season':        np.random.choice(seasons, n),
        'Crop':          np.random.choice(crops, n),
        'Area':          np.abs(np.random.lognormal(mean=8, sigma=1.5, size=n)),
        'Production':    np.abs(np.random.lognormal(mean=9, sigma=1.8, size=n))
    })
    print(f"    Synthetic dataset created: {df.shape[0]:,} rows x {df.shape[1]} columns")

# --------------------------------------------------------------------------
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------------------------------
print("\n[2] Exploratory Data Analysis...")
print(f"    Columns   : {list(df.columns)}")
print(f"    Data types:\n{df.dtypes}")
print(f"\n    Missing values:\n{df.isnull().sum()}")
print(f"\n    Basic statistics:\n{df[['Area','Production']].describe().round(2)}")

# --- Plot 1: Production Distribution (raw vs log-transformed) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['Production'].dropna(), bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Production — Raw Distribution')
axes[0].set_xlabel('Production (Tonnes)')
axes[0].set_ylabel('Frequency')

log_prod = np.log1p(df['Production'].dropna())
axes[1].hist(log_prod, bins=50, color='seagreen', edgecolor='white')
axes[1].set_title('Production — Log-Transformed Distribution')
axes[1].set_xlabel('log(1 + Production)')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('eda_production_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: eda_production_distribution.png")

# --- Plot 2: Top 10 States by Average Production ---
top_states = df.groupby('State_Name')['Production'].mean().nlargest(10)
plt.figure(figsize=(10, 5))
top_states.sort_values().plot(kind='barh', color='steelblue')
plt.title('Top 10 States by Average Crop Production')
plt.xlabel('Average Production (Tonnes)')
plt.tight_layout()
plt.savefig('eda_top_states.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: eda_top_states.png")

# --- Plot 3: Production by Season ---
season_avg = df.groupby('Season')['Production'].mean().sort_values(ascending=False)
plt.figure(figsize=(8, 4))
season_avg.plot(kind='bar', color='coral', edgecolor='white')
plt.title('Average Crop Production by Season')
plt.ylabel('Average Production (Tonnes)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('eda_season_production.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: eda_season_production.png")

# --- Plot 4: Correlation Heatmap ---
numeric_df = df[['Crop_Year', 'Area', 'Production']].dropna()
plt.figure(figsize=(5, 4))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='Blues', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: eda_correlation_heatmap.png")

# --------------------------------------------------------------------------
# SECTION 4: DATA PRE-PROCESSING
# --------------------------------------------------------------------------
print("\n[3] Data Pre-processing...")

# 4.1 Drop rows with missing Production (target variable)
df.dropna(subset=['Production'], inplace=True)
df['Area'].fillna(df['Area'].median(), inplace=True)
print(f"    After dropping missing target: {df.shape[0]:,} rows")

# 4.2 Log-transform target (addresses right skew identified in EDA)
df['Production_log'] = np.log1p(df['Production'])
df['Area_log'] = np.log1p(df['Area'])
print("    Applied log1p transform to Production and Area")

# 4.3 Encode categorical features
le = LabelEncoder()
for col in ['State_Name', 'Season', 'Crop']:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    print(f"    Encoded '{col}' -> {col}_enc ({df[col].nunique()} unique values)")

# 4.4 Assemble feature matrix
FEATURE_COLS = ['Area_log', 'Crop_Year', 'State_Name_enc', 'Season_enc', 'Crop_enc']
TARGET_COL   = 'Production_log'

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values
print(f"\n    Feature matrix X: {X.shape}")
print(f"    Target vector  y: {y.shape}")

# 4.5 Train / Test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
print(f"    Train: {X_train.shape[0]:,} samples  |  Test: {X_test.shape[0]:,} samples")

# 4.6 Standard scaling (for Ridge / Lasso)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# --------------------------------------------------------------------------
# SECTION 5: MODEL TRAINING & EVALUATION
# --------------------------------------------------------------------------
print("\n[4] Model Training & Evaluation...")

def evaluate_model(name, model, Xtr, Xte, ytr, yte):
    """Train model, run 5-fold CV, return metrics dict."""
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)

    rmse = np.sqrt(mean_squared_error(yte, y_pred))
    mae  = mean_absolute_error(yte, y_pred)
    r2   = r2_score(yte, y_pred)

    cv   = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, Xtr, ytr, cv=cv, scoring='r2').mean()

    print(f"\n    ── {name} ──")
    print(f"       RMSE (log scale) : {rmse:.4f}")
    print(f"       MAE  (log scale) : {mae:.4f}")
    print(f"       R²  (test set)   : {r2:.4f}")
    print(f"       R²  (5-fold CV)  : {cv_r2:.4f}")
    return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2_Test': r2, 'R2_CV': cv_r2,
            'y_pred': y_pred}

results = []
models  = {}

# 5.1 Linear Regression (baseline — no scaling needed for OLS)
res = evaluate_model("Linear Regression (Baseline)",
                     LinearRegression(), X_train, X_test, y_train, y_test)
results.append(res); models['Linear Regression'] = res['y_pred']

# 5.2 Ridge Regression (L2 regularisation)
res = evaluate_model("Ridge Regression (α=1.0)",
                     Ridge(alpha=1.0), X_train_sc, X_test_sc, y_train, y_test)
results.append(res); models['Ridge Regression'] = res['y_pred']

# 5.3 Lasso Regression (L1 regularisation — feature selection)
res = evaluate_model("Lasso Regression (α=0.01)",
                     Lasso(alpha=0.01, max_iter=5000), X_train_sc, X_test_sc, y_train, y_test)
results.append(res); models['Lasso Regression'] = res['y_pred']

# 5.4 Random Forest Regressor (primary model)
res = evaluate_model("Random Forest (n=200)",
                     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                     X_train, X_test, y_train, y_test)
results.append(res); models['Random Forest'] = res['y_pred']

# 5.5 Gradient Boosting (for comparison)
res = evaluate_model("Gradient Boosting",
                     GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                               max_depth=4, random_state=42),
                     X_train, X_test, y_train, y_test)
results.append(res); models['Gradient Boosting'] = res['y_pred']

# --------------------------------------------------------------------------
# SECTION 6: RESULTS COMPARISON
# --------------------------------------------------------------------------
print("\n[5] Model Comparison Summary")
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'y_pred'} for r in results])
results_df = results_df.sort_values('R2_Test', ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

# Identify best model
best = results_df.iloc[0]
print(f"\n    Best Model : {best['Model']}")
print(f"    R² (test)  : {best['R2_Test']:.4f}")
print(f"    RMSE       : {best['RMSE']:.4f}")

# --- Plot 5: Model Comparison Bar Chart ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['steelblue', 'seagreen', 'coral', 'mediumpurple', 'orange']

axes[0].bar(results_df['Model'], results_df['R2_Test'], color=colors)
axes[0].set_title('R² Score Comparison (Test Set)')
axes[0].set_ylabel('R² Score')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=30)
for i, v in enumerate(results_df['R2_Test']):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

axes[1].bar(results_df['Model'], results_df['RMSE'], color=colors)
axes[1].set_title('RMSE Comparison (Log Scale)')
axes[1].set_ylabel('RMSE')
axes[1].tick_params(axis='x', rotation=30)
for i, v in enumerate(results_df['RMSE']):
    axes[1].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n    Saved: model_comparison.png")

# --- Plot 6: Actual vs Predicted (Best Model) ---
best_model_name = best['Model']
best_preds = models[best_model_name]
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_preds, alpha=0.3, color='steelblue', s=10)
lim = [min(y_test.min(), best_preds.min()) - 0.5,
       max(y_test.max(), best_preds.max()) + 0.5]
plt.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect Prediction')
plt.xlabel('Actual log(1 + Production)')
plt.ylabel('Predicted log(1 + Production)')
plt.title(f'Actual vs Predicted — {best_model_name}')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: actual_vs_predicted.png")

# --- Plot 7: Feature Importance (Random Forest) ---
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values()
plt.figure(figsize=(7, 4))
importances.plot(kind='barh', color='steelblue')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print("    Saved: feature_importance.png")

# --------------------------------------------------------------------------
# SECTION 7: SAVE RESULTS
# --------------------------------------------------------------------------
results_df.to_csv('model_results_summary.csv', index=False)
print("\n    Saved: model_results_summary.csv")

print("\n" + "=" * 65)
print("  Project Complete!")
print(f"  Best Model : {best['Model']}")
print(f"  R² Score   : {best['R2_Test']:.4f}  |  RMSE: {best['RMSE']:.4f}")
print("=" * 65)
