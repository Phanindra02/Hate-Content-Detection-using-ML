from tkinter import CENTER
from turtle import width
import streamlit as st
import pandas as pd 
import time
from nltk.tokenize import word_tokenize
import nltk
import string
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
df=pd.read_csv("train.csv")
df['tweet'] = df['tweet'].apply(lambda x : ' '.join([tweet for tweet in x.split()if not tweet.startswith("@")]))
df['tweet'] = df['tweet'].apply(lambda y:' '.join([k for k in nltk.word_tokenize(y) if k  not in (stop_words or string.punctuation )]))
df['tweet'] = df['tweet'].apply(lambda y:' '.join([k for k in nltk.word_tokenize(y) if k.isalpha()]))
x=df.iloc[:,-1].values
y=df.iloc[:,-2].values

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,train_size=0.7,test_size=0.3)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words = 'english')
x_train_vect=vect.fit_transform(x_train)
x_test_vect=vect.transform(x_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='lbfgs', max_iter=24783)
model.fit(x_train_vect,y_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 70,random_state=0)
classifier.fit(x_train_vect, y_train)

from sklearn.svm import SVC
mode = SVC()
mode.fit(x_train_vect, y_train) 

st.title("#Hate tweet detection",CENTER)
st.image("twitter.jpg", width=50)
navi = st.sidebar.radio("Select",["Home","Tweet it"])

if navi=="Home":
    graph=st.selectbox("Select the model to fit",["Select","Random forest","Logistic regression","SVC"])
    
    if graph == "Logistic regression":
       st.write("Applying the logistic regression model on data")
       y_pred=model.predict(x_test_vect)
       result=accuracy_score(y_pred,y_test)
       progress=st.progress(0)
       for i in range(100):
           time.sleep(0.01)
           progress.progress(i+1)
       st.warning(f"**THE ACCURACY SCORE IS** ::**{result}**")

    if graph =="Random forest":
        st.write("Applying the random forest  model on data")
        y_pred1 = classifier.predict(x_test_vect)
        result=accuracy_score(y_pred1,y_test)
        progress=st.progress(0)
        for i in range(100):
           time.sleep(0.01)
           progress.progress(i+1)
        st.warning(f"**THE ACCURACY SCORE IS** ::**{result}**")

    if graph =="SVC":
        st.write("Applying the svc  model on data")
        y_pred2= mode.predict(x_test_vect)
        result=accuracy_score(y_pred2,y_test)
        progress=st.progress(0)
        for i in range(100):
           time.sleep(0.01)
           progress.progress(i+1)
        st.warning(f"**THE ACCURACY SCORE IS** ::**{result}**")
    

           
if navi=="Tweet it":
    tweet=st.text_area("Type the tweet")
    sp=st.selectbox("Select the model to test",["Select","Random forest","Logistic regression","SVC"])

    if sp == "Logistic regression":
       st.write("Applying the logistic regression model on tweet")
       vect1=vect.transform([tweet])
       out1 = model.predict(vect1)
       if out1 == 0:
           st.success("**POSTIVE TWEET**")
       else:
           st.error("**NEGATIVE TWEET**")

    if sp == "SVC":
       st.write("Applying the svc  model on tweet")
       vect2=vect.transform([tweet])
       out2 = mode.predict(vect2)
       if out2 == 0:
           st.success("**POSTIVE TWEET**")
       else:
           st.error("**NEGATIVE TWEET**")
      

    if sp == "Random forest":
       st.write("Applying the random forest model on tweet")
       vect3=vect.transform([tweet])
       out3 = classifier.predict(vect3)
       if out3 == 0:
           st.success("**POSTIVE TWEET**")
       else:
           st.error("**NEGATIVE TWEET**")