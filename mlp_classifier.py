from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import X_train, X_val, y_train, y_val

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train) 
y_val = label_encoder.transform(y_val)

mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

y_pred_mlp = mlp_model.predict(X_val)

print("Neural Network Accuracy:", accuracy_score(y_val, y_pred_mlp))

y_pred_mlp_prob = mlp_model.predict_proba(X_val)[:, 1] 
print("Neural Network AUC:", roc_auc_score(y_val, y_pred_mlp_prob))

