import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load and prepare dataset
df = fetch_california_housing()
dataset = pd.DataFrame(df["data"], columns=df.feature_names)
dataset["MedVal"] = df.target

# Split features and target
X = dataset.drop(columns=["MedVal"])
y = dataset["MedVal"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate model
model = LinearRegression()
mse_scores = cross_val_score(model, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=5)
# print(mse_scores)
print(np.mean(mse_scores))

model.fit(X_test_scaled,y_test)
y_pred=model.predict(X_test_scaled)


sns.displot(y_pred-y_test,kind="kde")
# Show the plot
plt.show()

score=r2_score(y_pred,y_test)
print(score)
