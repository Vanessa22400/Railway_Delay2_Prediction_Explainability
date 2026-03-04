# Railway Delay Modeling and Explainability - Tübingen (Germany)
*Advanced machine learning and SHAP interpretability to improve delay prediction and reliability insights in Germany’s national railway system (Deutsche Bahn).*

**Dataset**: 12 months of railway operations (24,760 cleaned records)  
**Techniques**: feature reuse, boosting models (XGBoost, LightGBM, CatBoost), SHAP explainability  
**Key Result**: Random Forest reached 0.87 accuracy with 0.35 recall for critical delays; CatBoost achieved the lowest MAE (2.30 min) for delay duration

---

## Business Context

This project is a continuation of my previous analysis of railway delays at Tübingen Hbf. In the first study, I used exploratory analysis and hypothesis testing to confirm real operational patterns (weekday fragility, rush-hour amplification, service-type differences, and directional bottlenecks). That work also highlighted a practical limitation: predicting the exact delay duration can be unstable in a complex railway network.

Building on those findings, this second project focuses on **predictive reliability**, **model comparison**, and **interpretability** under real-world constraints. The goal is not only to predict delays, but also to understand **why** models make certain predictions and which operational features matter most.

---

## Problem Statement

Can advanced machine learning models improve delay prediction and critical delay detection for Tübingen Hbf, and what do explainability methods reveal about the structural drivers behind delays?

---

## Objectives

- Reuse engineered features and operational insights from the previous project
- Benchmark performance with Random Forest (baseline)
- Evaluate gradient boosting models (XGBoost, LightGBM, CatBoost)
- Compare results across **regression** (delay minutes) and **classification** (critical delays)
- Apply **SHAP** to interpret model decisions and identify the strongest drivers
- Translate results into operational insights that support planning and monitoring

---

## Methodology

1. **Data Cleaning and Consistency**:  Same cleaning rules as the previous project (timestamp conversion, removing negative delays, preserving operational realism).

2. **Exploratory Insights Recap**: Short recap of the structural patterns already validated in the first study (without repeating the full EDA).

3. **Modeling Dataset Preparation**: Remove identifiers and post-event timing fields to avoid leakage; prepare a modeling-ready dataset.

5. **Feature Engineering**:
   - One-hot encoding for service categories (train type)  
   - **Cyclical time encoding** using sine/cosine transformations for hour-of-day  
   - Binary target creation for critical delays (> 5 minutes)

6. **Model Selection and Validation**:  
   - Baseline: Random Forest (regression and classification)  
   - Boosting: XGBoost, LightGBM, CatBoost  
   - Evaluation with error metrics (MAE/RMSE) and classification metrics (precision/recall/F1)

7. **Model Interpretation (SHAP)**: Use SHAP values to explain feature impact globally and on individual predictions.

---

## Tools & Technologies

- Python (Pandas, NumPy)
- Scikit-learn
- XGBoost, LightGBM, CatBoost
- SHAP (Model Explainability)
- Matplotlib, Seaborn
- Data Source: Deutsche Bahn data (public repository) and curated dataset from the previous project

---

## Exploratory Data Analysis Highlights

This project does not repeat the full EDA from the previous study, but key structural behaviors remain important for modeling:

- **Extreme events are rare but important:** the distribution is strongly right-skewed (max delay 331 minutes).
- **Service-type behavior differs:** express and regional services show distinct delay dynamics.
- **Early-week fragility persists:** Mondays show higher average delays than other weekdays.
- **Rush hour amplifies impact:** peak periods increase mean delays (3.83 vs. 2.77 minutes).

These patterns shape both model framing and interpretation.

Optional figure (if you want one visual in this section):  
![Weekly delay pattern](images/Average_Train_Delay_Weekday.png)  
*Average delay by weekday. Mondays show consistently higher delays.*

---

## Modeling Approach

This project uses two predictive tasks to reflect real operational needs:

**Regression Task (Predicting Delay Duration)**  
Estimates expected delay minutes under typical conditions. This supports baseline performance monitoring and operational planning.

**Classification Task (Predicting Critical Delay Risk)**  
Labels delays above 5 minutes as critical delays to better align modeling with passenger decision-making and disruption risk.

A key design choice in this project is **cyclical time representation** (hour_sin/hour_cos), which helps models treat late-night and early-morning hours as adjacent rather than distant.

---

## Model Performance

**Baseline: Random Forest**

Regression  
- MAE: 2.42 minutes  
- RMSE: 5.41 minutes  

Classification (Critical delays > 5 minutes)  
- Accuracy: 0.87  
- Critical delay precision (Class 1): 0.67  
- Critical delay recall (Class 1): 0.35  

Interpretation  
The baseline models capture routine operational patterns reasonably well. However, recall for critical delays remains limited, reflecting the reality that severe disruptions are often driven by irregular events not present in the dataset.

**Boosting Models: XGBoost, LightGBM, CatBoost**

Regression (MAE)  
- XGBoost: 2.42  
- LightGBM: 2.48  
- **CatBoost: 2.30 (best MAE)**  

Classification  
Overall accuracy remained high across models, but **critical delay recall did not improve meaningfully**. This suggests that the main barrier is not algorithm choice, but structural uncertainty in the available features.

---

## Key Insights

- **Model choice had limited impact on critical delay detection:** boosting did not consistently improve recall for critical delays, reinforcing that rare disruptions are difficult to anticipate without external variables.
- **Routine delays are more predictable than extremes:** low MAE shows models perform well on typical patterns, while RMSE reflects sensitivity to rare high-impact events.
- **Cyclical time encoding supports operational realism:** using sine/cosine for hour improves how models represent daily rhythms.
- **Explainability adds practical value:** SHAP reveals which operational variables dominate predictions, making model outputs easier to trust and communicate.

---

## Business Impact & Applications

**Predictive delay monitoring**  
Use regression predictions to estimate routine delay minutes and track performance trends over time.

**Route-level risk assessment**  
Route structure variables (ride and station sequence) indicate corridor-dependent behavior, helping identify recurring bottlenecks.

**Operational optimization**  
Different service categories show different delay contributions, supporting differentiated buffer strategies and schedule planning.

**Decision support and communication**  
Explainability improves transparency, making it easier to communicate why certain situations are predicted as higher risk.

Optional figures (recommended, because they look strong for recruiters):  
![SHAP summary plot](images/fig_shap_summary.png)  
*Global SHAP summary showing which features drive delay predictions.*  

![SHAP bar importance](images/fig_shap_bar.png)  
*Global feature importance ranked by SHAP contribution.*

---

## Next Steps

- Integrate external data (weather, maintenance schedules, disruption logs) to improve critical delay recall
- Add uncertainty estimation to better communicate prediction confidence
- Test time-aware validation to reflect operational sequencing
- Build a monitoring dashboard to track reliability by service category and route corridor

---

## Repository Structure
