from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from data_preprocessing import X_train, X_val, y_train, y_val


y_train = y_train.map({'Yes': 1, 'No': 0})
y_val = y_val.map({'Yes': 1, 'No': 0})


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_val)
y_pred_proba_rf = rf_model.predict_proba(X_val)[:, 1]  


print("Random Forest Accuracy:", accuracy_score(y_val, y_pred_rf))
print("Random Forest AUC:", roc_auc_score(y_val, y_pred_proba_rf))
