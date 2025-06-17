import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score,mean_squared_error

dataset=fetch_california_housing()
df=pd.DataFrame(dataset["data"],columns=dataset.feature_names)

df["MedVal"]=dataset.target
X=df.drop(columns=["MedVal"])
y=df["MedVal"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

lasso=Lasso()
params= {"alpha": [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50]}
lassocv=GridSearchCV(lasso,params,scoring="neg_mean_squared_error",cv=5)
lassocv.fit(X_train,y_train)

print(lassocv.best_score_,lassocv.best_params_)

lassocv_pred=lassocv.predict(X_test)
r2_score=r2_score(lassocv_pred,y_test)
mse=mean_squared_error(lassocv_pred,y_test)
print("r2 score:",r2_score)
print("mse score:",mse)
sns.displot(lassocv_pred-y_test,kind="kde")

plt.show()