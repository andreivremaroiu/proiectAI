from data_preprocessing import load_and_preprocess_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

X_train, X_val, y_train, y_val, _ = load_and_preprocess_data()

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_val)

print("Support Vector Machine Results")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:")
print(classification_report(y_val, y_pred))
