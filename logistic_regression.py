from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from data_preprocessing import X_train, X_val, y_train, y_val

y_train = y_train.map({'Yes': 1, 'No': 0})
y_val = y_val.map({'Yes': 1, 'No': 0})

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_val)
print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred))
print("Logistic Regression AUC:", roc_auc_score(y_val, y_pred))
