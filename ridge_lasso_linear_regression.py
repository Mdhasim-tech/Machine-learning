import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

#Load the dataset
df=fetch_california_housing()
dataset=pd.DataFrame(df['data'],columns=df.feature_names)
dataset["MedVal"]=df.target

#Features and target
X=dataset.drop(columns=["MedVal"])
y=dataset["MedVal"]

#Train/Test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#Standardize features
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#Define models
models={
    "LinearRegression":LinearRegression(),
    "Ridge":Ridge(alpha=1.0),
    "Lasso":Lasso(alpha=0.1)
}

#Store results
results=[]

#Train and evaluate each model
for name,model in models.items():
    model.fit(X_train_scaled,y_train)

    y_train_pred=model.predict(X_train_scaled)
    y_test_pred=model.predict(X_test_scaled)

    train_r2=r2_score(y_train,y_train_pred)
    test_r2=r2_score(y_test,y_test_pred)

    train_mse=mean_squared_error(y_train,y_train_pred)
    test_mse=mean_squared_error(y_test,y_test_pred)

    results.append({
        "Model": name,
        "Train R²": train_r2,
        "Test R²": test_r2,
        "Train MSE": train_mse,
        "Test MSE": test_mse
    })

# Display results
results_df = pd.DataFrame(results)
print(results_df)

# Optional: Plot test R² scores
plt.figure(figsize=(8, 4))
plt.bar(results_df["Model"], results_df["Test R²"], color=["skyblue", "lightgreen", "lightcoral"])
plt.title("Test R² Comparison")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.show()
