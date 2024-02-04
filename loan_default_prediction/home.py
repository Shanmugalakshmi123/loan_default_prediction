import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
st.title("Loan Default Prediction")
df=pd.read_csv("loan_default_prediction_project.csv")
#df.info()
mod=df['Gender'].mode()
df["Gender"]=df["Gender"].fillna(mod[0])
df=df[df['Employment_Status'].notnull()]
#df.info()
le=LabelEncoder()
df['Employment_Status']=le.fit_transform(df['Employment_Status'])
df['Location']=le.fit_transform(df['Location'])
df['Gender']=le.fit_transform(df['Gender'])
df=df[df['Income']<120000]
x=df.iloc[:,[0,1,2,3,4,5,6,7,9,10,11]]
y=df.iloc[:,[8]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=42)
x_train_over,y_train_over=sm.fit_resample(x_train,y_train)
x_train=pd.DataFrame(x_train_over,columns=x_train.columns)
lr=RandomForestClassifier()
lr.fit(x_train_over,y_train_over)
y_pred=lr.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
acc=accuracy_score(y_test,y_pred)
print(acc)
scores=cross_val_score(lr,x_train_over,y_train_over,cv=10)
print("cross-validation-scores",scores)
c1,c2,c3=st.columns(3)
age=int(c1.text_input("Enter Age",value=56))
inc=float(c2.text_input("Enter income",value=71266.105))
emp=int(c3.text_input("Enter employment status",value=0))
loc=int(c1.text_input("Enter location",value=1))
credit_score=int(c2.text_input("Enter credit score",value=639))
ratio=float(c3.text_input("Enter debt to income ratio",value=0.0071421))
bal=float(c1.text_input("Enter loan balance",value=27060.188))
amt=float(c2.text_input("Enter loan amount",value=13068.331))
rate=float(c3.text_input("Enter interest rate",value=18.185533))
duration=int(c1.text_input("Enter loan duration months",value=59))
gender=int(c2.text_input("Enter gender",value=1))
test=np.array([[age,gender,inc,emp,loc,credit_score,ratio,bal,amt,rate,duration]])
t=c1.button("Predict")
if t:
    y_pred=lr.predict(test)
    #st.write(y_pred)
    st.write("Loan Status:",y_pred[0])
    st.write("Accuracy",acc*100,"%")