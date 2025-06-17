from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd


df=fetch_california_housing()
dataset=pd.DataFrame(df["data"],columns=df.feature_names)
dataset["MedVal"]=df.target

X = dataset.drop(columns=["MedVal"])
y = dataset["MedVal"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


ridge_regressor=Ridge()
parameters={"alpha":[1,2,3,4,5,10,20,30,40,50,60,70,80,90]}
grid=GridSearchCV(ridge_regressor,parameters,scoring="neg_mean_squared_error",cv=5)
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_score_)

ridge_pred=grid.predict(X_test)
print(ridge_pred)


# sns.displot(ridge_pred-y_test,kind="kde")
# plt.show()
score=r2_score(ridge_pred,y_test)
print(score)







