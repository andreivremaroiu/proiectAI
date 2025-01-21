from data_preprocessing import load_and_preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_train, X_val, y_train, y_val, _ = load_and_preprocess_data()

elastic_net = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000, random_state=42)
elastic_net.fit(X_train, y_train)

y_pred = elastic_net.predict(X_val)

print("Elastic Net Results")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:")
print(classification_report(y_val, y_pred))
