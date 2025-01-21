from data_preprocessing import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report

X_train, X_val, y_train, y_val, _ = load_and_preprocess_data()

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='roc_auc', cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_val)
y_pred_prob = best_rf.predict_proba(X_val)[:, 1]

print("Best Parameters:", grid_search.best_params_)
print("Validation AUC:", roc_auc_score(y_val, y_pred_prob))
print("Classification Report:")
print(classification_report(y_val, y_pred))
