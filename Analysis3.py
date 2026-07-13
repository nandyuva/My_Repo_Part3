# Part 3 - Advanced Modeling: Ensembles, Tuning, Pipeline
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import *

# 1. LOAD DATA
df = pd.read_csv("cleaned_data.csv")

# CHANGE THIS to your target column from Part 2
TARGET_COL = 'Sales'  # <-- replace with your column

y_reg = df[TARGET_COL]
median_val = y_reg.median()
y = (y_reg > median_val).astype(int) # Binary target

X = df.drop(TARGET_COL, axis=1)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("="*60)
print("1. DECISION TREE BASELINE")
print("="*60)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
print("Train Acc:", accuracy_score(y_train, dt.predict(X_train)))
print("Test Acc:", accuracy_score(y_test, pred_dt))
print("Interpretation: If Train >> Test, it's overfitting = high variance")


print("\n" + "="*60)
print("2. CONTROLLED DECISION TREE")
print("="*60)
dt_control = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt_control.fit(X_train, y_train)
print("Train Acc:", accuracy_score(y_train, dt_control.predict(X_train)))
print("Test Acc:", accuracy_score(y_test, dt_control.predict(X_test)))
print("Interpretation: Limiting depth reduces variance, helps generalization")


print("\n" + "="*60)
print("3. ENSEMBLE COMPARISON")
print("="*60)
# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

models = {"RandomForest": rf, "GradientBoosting": gb}
ensemble_results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    ensemble_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1': f1_score(y_test, pred),
        'AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    })
ensemble_df = pd.DataFrame(ensemble_results)
print(ensemble_df)

# Feature Importance from RF
importances = rf.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
print("\nTop 5 Features by Importance:")
print(feat_df.sort_values('Importance', ascending=False).head(5))


print("\n" + "="*60)
print("4. HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*60)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [5, 10, None],
    'clf__min_samples_split': [2, 5]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best CV AUC:", grid.best_score_)

best_model = grid.best_estimator_
pred_best = best_model.predict(X_test)
print("Test AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]))


print("\n" + "="*60)
print("5. SAVE BEST MODEL")
print("="*60)
joblib.dump(best_model, 'best_model.pkl')
print("Model saved as best_model.pkl")

# Code block for README to reload
print("\n# To load model later:")
print("model = joblib.load('best_model.pkl')")
print("pred = model.predict(new_data)")


print("\n" + "="*60)
print("6. SUMMARY TABLE FOR README")
print("="*60)
summary = pd.DataFrame({
    'Model': ['DecisionTree', 'Controlled_DT', 'RandomForest', 'GradientBoost', 'Tuned_RF'],
    'Accuracy': [accuracy_score(y_test, pred_dt), accuracy_score(y_test, dt_control.predict(X_test)), 
                 ensemble_df.loc[0,'Accuracy'], ensemble_df.loc[1,'Accuracy'], accuracy_score(y_test, pred_best)],
    'AUC': [roc_auc_score(y_test, dt.predict_proba(X_test)[:,1]), roc_auc_score(y_test, dt_control.predict_proba(X_test)[:,1]),
            ensemble_df.loc[0,'AUC'], ensemble_df.loc[1,'AUC'], roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1])]
})
print(summary)
print("\nRecommendation: Pick the model with highest AUC and stable train/test gap")