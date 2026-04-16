# IBM HR Employee Attrition - Machine Learning Analysis

This project uses the IBM HR Analytics Employee Attrition dataset to solve two key ML tasks: employee salary prediction (regression) and attrition prediction (classification). It benchmarks 13+ algorithms to evaluate performance and select the most effective models using a data-driven approach.

## Contents
- **`IBM_HR_AdvancedML_Complete_3.ipynb`**: Complete Jupyter Notebook containing data preprocessing, exploratory data analysis, and the implementation of regression (to predict `MonthlyIncome`) and classification models (to predict `Attrition`).
- **`WA_Fn-UseC_-HR-Employee-Attrition (2).csv`**: The dataset used for model training and analysis.
- **`fix_order.py`**: A python script used to organize the cell execution order of the notebook.

## Overview
Multiple machine learning models were developed and tuned, including:
- Ridge, Lasso, and ElasticNet Regression.
- Ensemble methods like Random Forest, AdaBoost, Gradient Boosting, and XGBoost.
- Principal Component Analysis (PCA) to reduce feature dimensionality.
- Classification optimization techniques including using SMOTE and threshold tuning for evaluating the business impact of the attrition predictor.

## Recent Updates
- Corrected a notebook execution issue where predictive models were evaluated before the `y_proba_tuned_cls` metrics were generated. 
- Reordered the notebook execution flow using a custom script (`fix_order.py`) to systematically run classification tuning correctly.
- Programmatically ran all notebook cells from top to bottom, guaranteeing reproducible charts, verified models, and resolved errors.
