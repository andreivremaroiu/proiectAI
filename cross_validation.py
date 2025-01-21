from data_preprocessing import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X_train, X_val, y_train, y_val, _ = load_and_preprocess_data()

rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    min_samples_split=5, 
    min_samples_leaf=2, 
    bootstrap=True,
    random_state=42
)

cv_scores = cross_val_score(rf, X_train, y_train, scoring='roc_auc', cv=5, n_jobs=-1)

print("Cross-Validation AUC Scores:", cv_scores)
print("Mean AUC:", cv_scores.mean())
print("Standard Deviation of AUC:", cv_scores.std())
