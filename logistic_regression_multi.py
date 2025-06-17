import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load full iris dataset
df = sns.load_dataset('iris')

# Encode species as numbers (multiclass)
df["species"] = df["species"].map({
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
})

# Features and labels
X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
log_reg = LogisticRegression(multi_class='auto', solver='saga', max_iter=5000)

params = {
    "penalty": ["l1", "l2", "elasticnet"],
    "C": [0.1, 1, 10],
    "l1_ratio": [0.5],  # Only relevant when penalty is elasticnet
}

grid = GridSearchCV(log_reg, params, cv=5, scoring='accuracy', error_score='raise')
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

print("Best Parameters:", grid.best_params_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

sns.pairplot(df, hue="species")
plt.show()
