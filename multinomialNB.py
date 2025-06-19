import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report


df=pd.read_csv('sentiment_data.csv',encoding='latin1')
df_clean=df.dropna(subset=['text','sentiment'])
X=df_clean["text"].dropna()
y=df_clean["sentiment"].dropna()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
vectorizer=CountVectorizer()
model=MultinomialNB()

pipeline=make_pipeline(vectorizer,model)

pipeline.fit(X_train,y_train)


pred=pipeline.predict(X_test)
# print(pred)

report=classification_report(y_test,pred)
# print(report)
vector=["i love you!"]
print(pipeline.predict(vector))