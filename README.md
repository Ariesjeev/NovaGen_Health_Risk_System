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

## 📈 Dataset at a Glance
 
- **Near-Balanced Target**: The Target split is **~52.1% "healthy" vs. 47.9% "at-risk"**, close enough to balanced that accuracy remains a meaningful metric alongside recall.
- **Rich Feature Set**: 23 columns spanning core vitals (`BMI`, `Blood_Pressure`, `Cholesterol`, `Glucose_Level`, `Heart_Rate`), lifestyle factors (`Sleep_Hours`, `Exercise_Hours`, `Water_Intake`, `Stress_Level`, `Smoking`, `Alcohol`, `Diet`), and pre-engineered categorical encodings (`Diet_Type__Vegan`, `Diet_Type__Vegetarian`, `Blood_Group_AB/B/O`).
- **No Missing Values**: All 9,549 records are fully populated, so the pipeline could move directly into scaling and modeling without imputation.
---
 
## 📊 Model Evaluation & Results
 
All five models were evaluated on the same 20% test split (1,910 individuals):
 
| Model | Accuracy | Recall |
|---|:---:|:---:|
| Logistic Regression | 81.4% | 82.8% |
| KNN | 88.3% | 88.3% |
| **Random Forest** | **93.7%** | **95.9%** |
| Gradient Boosting | 93.0% | 94.9% |
| Voting Classifier | 91.6% | 93.1% |
 
### 🏆 Recommended Model: Random Forest
 
**Random Forest was the clear winner on recall (95.9%)** — the metric prioritized here, since failing to flag a genuinely at-risk participant carries far more downstream cost than a false positive in a research-screening context. It also posted the highest overall accuracy (93.7%), making it a strictly dominant choice over every other model tested, including the more complex Voting Classifier.
 
### 💡 Key Insights
 
- **Ensembles Decisively Beat Linear/Distance Models**: Both Random Forest and Gradient Boosting cleared 93%+ accuracy, a double-digit jump over Logistic Regression's 81.4% — confirming that the health-risk boundary is highly non-linear and better captured by tree-based splits than a linear decision surface or raw Euclidean distance.
- **Bagging Edged Out Boosting**: Random Forest's independent, variance-reducing trees slightly outperformed Gradient Boosting's sequential, bias-reducing trees on this dataset — suggesting the signal here benefits more from averaging over diverse trees than from iterative error correction.
- **The Voting Ensemble Underperformed Its Best Component**: Averaging Logistic Regression, KNN, and Random Forest predictions (91.6% recall) actually diluted Random Forest's individual strength (95.9%) by blending in two weaker models — a reminder that ensembling only helps when component models are comparably strong.
---
 
## 🚀 Next Steps: Production Roadmap
 
1. **Hyperparameter Tuning**: Run `GridSearchCV`/`RandomizedSearchCV` on Random Forest (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`) to push recall further.
2. **Feature Importance Analysis**: Extract and rank Random Forest feature importances to identify which physiological or lifestyle indicators most strongly predict health risk — directly useful for the research team's participant-stratification decisions.
3. **Threshold Tuning**: Since recall is the priority metric, tune the classification threshold (rather than the default 0.5) to push recall even higher at an acceptable precision cost.
4. **Stratified K-Fold Cross-Validation**: Move from a single train/test split to 5- or 10-fold CV for a more robust, generalizable performance estimate.
5. **Try Gradient Boosting Variants**: Benchmark `XGBoost`, `LightGBM`, or `CatBoost` against the current `GradientBoostingClassifier`, which may close or reverse the gap with Random Forest.
6. **Calibrated Ensemble Weighting**: If ensembling is revisited, weight the Voting Classifier's components (rather than equal soft voting) so weaker learners don't dilute the strongest one.
---
 
## 🛠️ How to Run the Project Locally
 
### 1. Clone the Repository
 
```bash
git clone https://github.com/Ariesjeev/NovaGen_Health_Risk_System.git
cd NovaGen_Health_Risk_System
```
 
### 2. Run the Jupyter Notebook
 
Open [novagen.ipynb](novagen.ipynb) in your IDE of choice (such as VS Code or Jupyter Lab) and execute the cells. Make sure `novagen_dataset.csv` is in the same directory (or update the `pd.read_csv` path in the notebook).
