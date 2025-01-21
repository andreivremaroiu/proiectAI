from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from data_preprocessing import X_train, X_val, y_train, y_val

y_train = y_train.map({'Yes': 1, 'No': 0})
y_val = y_val.map({'Yes': 1, 'No': 0})

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_val)
print("Decision Tree Accuracy:", accuracy_score(y_val, y_pred_dt))
print("Decision Tree AUC:", roc_auc_score(y_val, y_pred_dt))
