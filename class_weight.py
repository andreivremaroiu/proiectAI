from data_preprocessing import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

X_train, X_val, y_train, y_val, _ = load_and_preprocess_data()

rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    min_samples_split=5, 
    min_samples_leaf=2, 
    bootstrap=True,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
y_pred_prob = rf.predict_proba(X_val)[:, 1]

print("Validation AUC with Class Weight:", roc_auc_score(y_val, y_pred_prob))
print("Classification Report:")
print(classification_report(y_val, y_pred))
