## RANDOM FORESTS
Random Forest Accuracy: 0.7835880933226066
Random Forest AUC: 0.800492547787286

## DECISION_TREE
Decision Tree Accuracy: 0.7864038616251006
Decision Tree AUC: 0.7293780922158912

## Logistic Regression
Logistic Regression Accuracy: 0.7892196299275945
Logistic Regression AUC: 0.6535976731971913

## MLP Classifier
Neural Network Accuracy: 0.7799678197908286
Neural Network AUC: 0.7789078326573307

## KNN

K-Nearest Neighbors Results
Accuracy: 0.7506033789219629
Classification Report:
              precision    recall  f1-score   support

          No       0.81      0.88      0.84      1885
         Yes       0.48      0.34      0.40       601

    accuracy                           0.75      2486
   macro avg       0.64      0.61      0.62      2486
weighted avg       0.73      0.75      0.73      2486


## Elastic Net
Elastic Net Results
Accuracy: 0.7855993563958166
Classification Report:
              precision    recall  f1-score   support

          No       0.82      0.93      0.87      1885
         Yes       0.60      0.34      0.44       601

    accuracy                           0.79      2486
   macro avg       0.71      0.64      0.65      2486
weighted avg       0.76      0.79      0.76      2486

## SVM
Support Vector Machine Results
Accuracy: 0.7775543041029767
Classification Report:
              precision    recall  f1-score   support

          No       0.83      0.89      0.86      1885
         Yes       0.55      0.43      0.48       601

    accuracy                           0.78      2486
   macro avg       0.69      0.66      0.67      2486
weighted avg       0.76      0.78      0.77      2486


## RANDOM FOREST TUNING
Best Parameters: {'bootstrap': True, 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}
Validation AUC: 0.8079840407455302
Classification Report:
              precision    recall  f1-score   support

          No       0.84      0.89      0.87      1885
         Yes       0.58      0.46      0.51       601

    accuracy                           0.79      2486
   macro avg       0.71      0.68      0.69      2486

weighted avg       0.78      0.79      0.78      2486

## CROSS VALIDATION
Cross-Validation AUC Scores: [0.79793015 0.81812808 0.80661703 0.78842336 0.78593897]
Mean AUC: 0.7994075212224597
Standard Deviation of AUC: 0.01188841997694555

## CLASS WEIGHT
Validation AUC with Class Weight: 0.8077267330752901
Classification Report:
              precision    recall  f1-score   support

          No       0.91      0.79      0.84      1885
         Yes       0.53      0.77      0.63       601

    accuracy                           0.78      2486
   macro avg       0.72      0.78      0.74      2486
weighted avg       0.82      0.78      0.79      2486


## TEST EVALUATION

Validation AUC: 0.8054577472558998
Classification Report for Validation Data:
              precision    recall  f1-score   support

          No       0.81      0.92      0.86      1885
         Yes       0.57      0.32      0.41       601

    accuracy                           0.78      2486
   macro avg       0.69      0.62      0.64      2486
weighted avg       0.75      0.78      0.75      2486

Test Predictions:
Predicted Probabilities: [0.12741395 0.13374299 0.47563285 ... 0.1726512  0.39908214 0.13501314]
Predicted Classes: ['No' 'No' 'No' ... 'No' 'No' 'No']
   age  relative_wage  hours_of_training  is_certified  ...  type_of_company_startup_wo_funding     id  predicted_class  predicted_probability
0   32         166.20                 22             1  ...                                 0.0  12428               No               0.127414       
1   33         153.10                152             1  ...                                 0.0  12429               No               0.133743       
2   36         110.45                 23             0  ...                                 0.0  12430               No               0.475633       
3   30         162.84                 21             1  ...                                 0.0  12431               No               0.101249       
4   33         163.90                 15             1  ...                                 0.0  12432               No               0.096719       

[5 rows x 196 columns]