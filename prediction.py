import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import sklearn
import pickle


@st.cache
def load_model(file):
    model = pickle.load(open(file,"rb"))
    return model

lrmodel = load_model("LRPF.sav")
dtmodel = load_model("DTPF.sav")
svmmodel= load_model("SVMPF.sav")

df = pd.read_csv('./data/PassFail.csv')
st.title('Pass or Fail Prediction')
st.write('Feature 1:Self Study Hours(Daily)')
st.write('Feature 2:Tution Study Hours(Monthly)')


st.sidebar.title("Please Select")
image = Image.open("BinaryCla.png")
st.image(image,use_column_width=True)
st.markdown('<style>body{background-color: yellow;}</style>',unsafe_allow_html=True)


algotype = st.sidebar.selectbox('Select Algorithm Type',('LogisticR','SVM','DecisionTree'))
SS_select = st.sidebar.selectbox('Select Self Study Hours',df['Self_Study_Daily'].unique())
TS_select = st.sidebar.selectbox('Select Tuition Study Hours',df['Tution_Monthly'].unique())

pfd={1:"Pass",0:"Fail"}


if algotype=='LogisticR':
    pred=lrmodel.predict([[int(SS_select),int(TS_select)]])
    st.write("Prediction result=",pfd[pred.ravel()[0]])
elif algotype=='SVM':
    pred=svmmodel.predict([[int(SS_select),int(TS_select)]])
    st.write("Prediction result=",pfd[pred.ravel()[0]])
elif algotype=='DecisionTree':
    pred=dtmodel.predict([[int(SS_select),int(TS_select)]])
    st.write("Prediction result=",pfd[pred.ravel()[0]])


        


