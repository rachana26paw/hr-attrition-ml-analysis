# Advanced ML Project — IBM HR Attrition 

## What we worked on

In this project, we tried to solve two practical HR problems using machine learning:

* Predict whether an employee is likely to leave the company (**Attrition — classification**)
* Predict an employee’s salary based on their profile (**MonthlyIncome — regression**)

The focus was not just building models, but understanding how these predictions can actually help in real decision-making.

---

## Dataset

We used the IBM HR Attrition dataset (≈1400 employees, 35 features).

Some important features:

* Age, Department, JobRole
* WorkLifeBalance, JobSatisfaction
* YearsAtCompany, OverTime

Targets:

* **Attrition** → Yes/No
* **MonthlyIncome** → continuous

---

## How we approached it

Instead of relying on a single model, we compared multiple algorithms to understand what works best.

### Regression (salary prediction)

* Linear Regression (baseline)
* Ridge, Lasso, ElasticNet
* k-NN Regressor
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

### Classification (attrition prediction)

* Logistic Regression
* k-NN
* Decision Tree
* SVM (RBF kernel)
* Random Forest
* AdaBoost
* Gradient Boosting
* XGBoost

---

## What we implemented

* Data preprocessing (encoding + scaling)
* Train-test split
* Class imbalance handling (SMOTE)
* Hyperparameter tuning (GridSearchCV)
* Model comparison using:

  * Regression → R², RMSE
  * Classification → F1-score, AUC

---

## Key improvement we made

We noticed that **MonthlyIncome was right-skewed**, so we applied:

* Log transformation using `np.log1p()`
* Converted predictions back using `np.expm1()`

This helped improve linear model performance and made the results more reliable.

---

## What we learned

* Different models behave very differently on the same data
* Tree-based models handle non-linearity better
* Linear models need proper assumptions (like normal distribution)
* Metrics like F1-score and AUC matter more than accuracy in imbalanced datasets

---

## Why this matters

* Attrition prediction → helps reduce employee turnover
* Salary prediction → helps ensure fair compensation

Even small improvements here can have real financial impact for companies.

---

## Current status

* Full ML pipeline built
* Multiple models trained and compared
* Optimization and improvements applied
* Ready for final evaluation

---

## How to run

1. Download the dataset from Kaggle
2. Place the CSV in the project folder
3. Run:

```bash id="l2n8fw"
jupyter notebook IBM_HR_AdvancedML_Complete_2.ipynb
```

---

## Final note

This project helped us move from just “using models” to actually understanding how and why they work in real scenarios.
