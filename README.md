# 🏠 House Price Prediction: Comprehensive Regression Pipeline

This project builds a full end-to-end regression system to predict house sale prices using the classic House Prices: Advanced Regression Techniques dataset from Kaggle. The notebook was developed in Google Colab and follows a complete machine learning workflow: deep EDA, professional-grade preprocessing, feature engineering, model training, evaluation, and final submission creation.

- [Google Colab Notebook](https://colab.research.google.com/drive/1kjoC66-eGSKgwyA8y9yF_7f9su3UH_pY?usp=sharing)
- [Dataser Source](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

Goal:
To understand the structure of residential property data and construct a model that accurately estimates housing prices by capturing relationships among quality, structure, geography, and age.

## 📊 What This Project Covers
### 1. Data Loading & Preparation
- Integration with Kaggle API for automated dataset retrieval
- Initial inspection of shape, types, and completeness
- Consolidation of train and test sets for consistent preprocessing
- Index alignment using property Id values

### 2. Target Variable Analysis
- To stabilize variance and improve model performance:
- Visualized the skewed distribution of SalePrice
- Identified a strong right tail due to high-priced properties
- Applied log(1 + x) transformation to normalize the target
- Verified reduced skewness and improved distribution symmetry

### 3. Exploratory Data Analysis
- Generated a correlation matrix to reveal top predictive features
- Identified influential drivers such as OverallQual, GrLivArea, and garage size
- Confirmed intuitive patterns: higher quality, larger living areas, and better amenities strongly correlate with higher sale prices

### 4. Advanced Data Preprocessing
A structured, context-aware approach was used to impute missing values:
-  Numerical values
  - Zero-imputation for basement, garage, and masonry features where absence represents “not present”
  - Neighborhood-wise median imputation for LotFrontage
- Categorical values
  - “None” for features where NA explicitly indicates absence
  - Mode for true missing categories in ratings, exterior materials, zoning, utilities, and sale type
- Final integrity check ensuring no unresolved missing data remained

### 5. Feature Engineering
To strengthen model expressiveness, several derived features were added:
- TotalSF: Combined total of basement and living area
- TotalBath: Weighted aggregation of full and half bathrooms (including basement baths)
-  Age: Years between construction and sale date

These features intentionally capture structural size, usable space, and temporal depreciation.

### 6. Categorical Encoding

- Object-type features transformed via One-Hot Encoding
- Ensured no artificial order was imposed on nominal categories
- Produced a high-dimensional, machine-learning-friendly numerical feature matrix

### 7. Model Building & Training
Two models illustrate the evolution from baseline to advanced predictive systems:

#### 1. Linear Regression (Baseline)
- Scaled inputs using StandardScaler
- Served as an interpretable, assumption-aware benchmark
- Exhibited limitations due to non-linear relationships and feature interactions

#### 2) XGBoost Regressor (Advanced)
- Leveraged gradient boosting for expressive, non-linear modeling
- Trained on unscaled data (tree-based models handle magnitudes naturally)
- Demonstrated superior accuracy across RMSE, MAE, and R²
- Successfully captured interaction effects, diminishing returns, and structural patterns unseen by linear models

### 8. Model Evaluation
Performance was assessed using standard regression metrics:
- MAE for interpretability
- RMSE for error sensitivity
- R² for variance explained

XGBoost clearly outperformed the linear model, validating its suitability for structured tabular data with mixed feature types.

#### 9. Final Prediction & Submission
- Generated predictions for the test set in log scale
- Converted them back using expm1()
- Constructed a clean Kaggle submission file: submission.csv


## Potential Next Steps
- Hyperparameter tuning with GridSearchCV or Optuna
- Richer feature engineering, including interaction terms and condition-adjusted age metrics
- Ensemble modeling using stacking or blending
- L1/L2 regression for feature importance and interpretability
