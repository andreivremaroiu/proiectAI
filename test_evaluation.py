from data_preprocessing import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report


X_train, X_val, y_train, y_val, X_test = load_and_preprocess_data()


train_ids = X_train.pop('id') if 'id' in X_train.columns else None
val_ids = X_val.pop('id') if 'id' in X_val.columns else None
test_ids = X_test.pop('id') if 'id' in X_test.columns else None


rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    min_samples_split=5, 
    min_samples_leaf=2, 
    bootstrap=True,
    random_state=42
)
rf.fit(X_train, y_train)


y_val_pred = rf.predict(X_val)
y_val_pred_prob = rf.predict_proba(X_val)[:, 1]


y_test_pred = rf.predict(X_test)
y_test_pred_prob = rf.predict_proba(X_test)[:, 1]


print("Validation AUC:", roc_auc_score(y_val, y_val_pred_prob))
print("Classification Report for Validation Data:")
print(classification_report(y_val, y_val_pred))


print("Test Predictions:")
print("Predicted Probabilities:", y_test_pred_prob)
print("Predicted Classes:", y_test_pred)

if test_ids is not None:
    test_results = X_test.copy()
    test_results['id'] = test_ids
    test_results['predicted_class'] = y_test_pred
    test_results['predicted_probability'] = y_test_pred_prob
    print(test_results.head())
