import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import numpy as np
import os

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def display_top_features(model, feature_names, n=10):
    importances = model.named_steps['xgb'].feature_importances_
    top_idx = np.argsort(importances)[::-1][:n]
    print("\nTop Features:")
    for i in top_idx:
        print(f"- {feature_names[i]}: {importances[i]:.3f}")

def train_evaluate_save(data_path: str, model_path: str, report_path: str):
    df = load_data(data_path)
    target_col = 'Label'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42,
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85
        ))
    ])

    print('Running cross-validation...')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=skf)
    print(f'CV ROC-AUC mean: {cv_auc.mean():.4f}  std: {cv_auc.std():.4f}')

    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f'Test ROC-AUC: {auc:.4f}')


    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f'Model and pipeline saved to {model_path}')


    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'Classification report saved to {report_path}')

    display_top_features(pipeline, X.columns.tolist())

if __name__ == '__main__':
    train_evaluate_save(
        data_path='data/Loan_Default_Processed.csv',
        model_path='models/xgb_pipeline.joblib',
        report_path='models/classification_report.txt'
    )
