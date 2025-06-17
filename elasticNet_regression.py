import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# 1. Load the Diabetes dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="DiseaseProgression")

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Feature scaling (important for ElasticNet)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. ElasticNet model + hyperparameter tuning
elastic_net = ElasticNet()
params = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],   # regularization strength
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1]   # balance between L1 and L2
}
grid = GridSearchCV(elastic_net, params, cv=5, scoring='r2')
grid.fit(X_train_scaled, y_train)

# 5. Results
print("Best R² score on training data:", grid.best_score_)
print("Best parameters:", grid.best_params_)

# 6. Evaluate on test data
from sklearn.metrics import r2_score
y_pred = grid.predict(X_test_scaled)
print("Test R² score:", r2_score(y_test, y_pred))
