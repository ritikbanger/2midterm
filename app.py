# -*- coding: utf-8 -*-
"""

@author: ritik
"""

import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('ritikmid2.pkl','rb'))   
dataset= pd.read_csv('clustering dataset1.csv')
X = dataset.iloc[:,0:8].values
# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NaN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:,2:7]) 
#Replacing missing data with the calculated mean value  
X[:,2:7]= imputer.transform(X[:,2:7])  


# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:,0])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age):
  output= model.predict([[Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age]])
  print("Disease Category is",output)
  if output==[0]:
    prediction="Disease Category is 0"
  else:
    prediction="Disease Category is 1"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center>
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Patient Health Prediction")
    Age = st.number_input('Insert an Age',18,60)
    Glucose = st.number_input('Insert Glucose',0,200)
    BP= st.number_input('Insert BP',80,120)
    SkinThickness = st.number_input('Insert SkinThickness',0,100)
    Insulin = st.number_input('Insert Insulin',0,300)
    Gender = st.number_input('Insert 0 for Male 1 for Female',0,1)
    BMI= st.number_input('Insert BMI',10,50)
    PedigreeFunction= st.number_input('Insert a PedigreeFunction')
    
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Ritik Banger")
      st.subheader("Department of Computer Engineering")
    html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">II Midterm Exam - 6th Sem</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()
