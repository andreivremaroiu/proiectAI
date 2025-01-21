import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "Model": [
        "Random Forests", "Decision Tree", "Logistic Regression",
        "MLP Classifier", "KNN", "Elastic Net", "SVM",
        "Tuned Random Forest", "Random Forest w/ Class Weights"
    ],
    "Accuracy": [0.784, 0.786, 0.789, 0.780, 0.751, 0.786, 0.778, None, None],
    "AUC": [0.800, 0.729, 0.654, 0.779, None, None, None, 0.808, 0.808],
    "Recall (Minority Class)": [
        None, None, None, None, 0.61, 0.64, 0.66, 0.46, 0.78
    ],
    "Macro Avg Recall": [
        None, None, None, None, None, None, None, 0.68, 0.78
    ]
}

results_df = pd.DataFrame(data)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x="Accuracy", y="Model", data=results_df, palette="Blues_d")
plt.title("Model Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Model")

plt.subplot(1, 2, 2)
sns.barplot(x="AUC", y="Model", data=results_df, palette="Greens_d")
plt.title("Model AUC")
plt.xlabel("AUC")
plt.ylabel("Model")

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(x="Recall (Minority Class)", y="Model", data=results_df, palette="Oranges_d")
plt.title("Recall for Minority Class")
plt.xlabel("Recall")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="Macro Avg Recall", y="Model", data=results_df, palette="Purples_d")
plt.title("Macro Average Recall")
plt.xlabel("Macro Average Recall")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

results_df.to_csv("model_results_summary.csv", index=False)
print("Results saved to 'model_results_summary.csv'.")
