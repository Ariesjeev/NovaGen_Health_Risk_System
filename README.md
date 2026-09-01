# 🧬 NovaGen: Classifying Health Risk with Ensemble Learning
 
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9%2B-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0%2B-darkblue?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
 
An end-to-end supervised learning project that classifies individuals as **"healthy"** or **"at higher health risk"** using a population health dataset of 9,549 participant records ([novagen_dataset.csv](novagen_dataset.csv)). Five model families — Logistic Regression, KNN, Random Forest, Gradient Boosting, and a soft-voting ensemble — were trained and compared, with **Random Forest emerging as the top performer at 95.9% recall and 93.7% accuracy**.


## 📌 Project Overview & Problem Statement
 
NovaGen Research Labs, a biomedical research institute, runs large-scale population health studies to understand how underlying health conditions drive disease risk and long-term outcomes. Every year the institute recruits thousands of volunteers for medical exams, lifestyle assessments, and clinical tests, but lacked a reliable, consistent way to distinguish healthy participants from higher-risk ones — limiting how well they could select trial participants or stratify populations for outcome analysis.
 
Engaged as a **Data Scientist**, the goal was to build a predictive model classifying individuals as healthy or unhealthy from **9,549 unique participant records** (one per individual, avoiding sampling bias) spanning **23 numerical and categorical health indicators** — physiological measurements, lifestyle habits, and medical history — to directly support:
 
- Selecting eligible participants for clinical trials and longitudinal studies
- Stratifying populations for risk-based analysis and outcome comparison
Because **missing a genuinely high-risk individual is far more costly than a false alarm** (a health-screening context), **recall** was prioritized as the key evaluation metric alongside accuracy.
 
---
 
## 🔍 The Modeling Workflow
 
1. **Feature/Target Split** — separate `Target` (healthy = 1 / at-risk = 0) from the 22 predictors.
2. **Stratified Train/Test Split** — 80/20 split, stratified on `Target` to preserve the ~52/48 class balance in both sets.
3. **Feature Scaling** — `StandardScaler` fit on the training set and applied to both train and test splits, used for the distance- and coefficient-based models (Logistic Regression, KNN, Voting Classifier); tree-based models (Random Forest, Gradient Boosting) trained on unscaled features, since splits are scale-invariant.
4. **Five Models Trained & Compared**:
   - **Logistic Regression** (L2-regularized, `liblinear` solver) — linear baseline.
   - **K-Nearest Neighbors** (`k=5`, Euclidean distance).
   - **Random Forest** (`n_estimators=200`) — bagged ensemble of decision trees.
   - **Gradient Boosting** (`n_estimators=150`, `learning_rate=0.1`, `max_depth=3`) — sequential boosted trees.
   - **Voting Classifier** (soft voting over Logistic Regression + KNN + Random Forest) — ensemble-of-ensembles.
5. **Evaluation** — accuracy, recall, and full classification report on the same 20% held-out test set (1,910 participants) for every model, to select the best candidate for deployment.
---
