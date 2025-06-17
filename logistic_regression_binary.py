import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df=sns.load_dataset('iris')

df=df[df["species"]!="setosa"]
df["species"]=df['species'].map({
    "versicolor":0,"virginica":1
})


X=df.drop(columns=["species"])
y=df["species"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
logistic_regressor=LogisticRegression()
#Hyperparameter tuning
params={
    "penalty":["l1","l2","elasticnet"],
    "C":[1,2,3,4,5,6,10,20,30,40,50,60,70,80,90],
    "max_iter":[100,200,300],


}
model=GridSearchCV(logistic_regressor,params,scoring="accuracy",cv=10)
model.fit(X_train,y_train)
# print(model.best_score_)


y_pred=model.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
print(accuracy)

report=classification_report(y_pred,y_test)
print(report)
sns.pairplot(df, hue="species")
plt.show()

print(df.corr())