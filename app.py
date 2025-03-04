
import streamlit as st
import numpy as np
import pandas as pd
import pickle 

#Load the instamces that were created
with open('final_model.pkl','rb') as file:
    model=pickle.load(file)

with open('lda.pkl','rb') as file:
    LDA=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

def prediction (input_data):
    scaled_data=scaler.transform(input_data)
    lda_data= LDA.transform(scaled_data)
    pred=model.predict(lda_data)[0]

    if pred==1:
        return 'Wine 1'
    elif pred==2:
        return 'Wine 2'
    else:
        return 'Wine 3'

def main():
    st.title('Wine Classification')
    st.subheader('This application will give classifiy the wine based on their chemical constituents.')
    alc=st.text_input('Enter Alcohol percentage:')
    mal_acid=st.text_input('Enter Malic acid percentage:')
    ash=st.text_input('Enter Ash percentage:')
    alc_ash=st.text_input('Enter Alcalinity of ash:')
    mag=st.text_input('Enter Magnesium percentage:')
    phe=st.text_input('Enter Total phenols:')
    fla=st.text_input('Enter Total Flavanoids:')
    nfla=st.text_input('Enter Total Nonflavanoid phenols:')
    pro=st.text_input('Enter Proanthocyanins percentage:')
    co_i=st.text_input('Enter Total Color_intensity:')
    hue=st.text_input('Enter Hue:')
    od=st.text_input('Enter OD280/OD315 of Diluted wine:')
    proline=st.text_input('Enter Total Proline:')

    input_list=[[alc,mal_acid,ash,alc_ash,mag,phe,fla,nfla,pro,co_i,hue,od,proline]]

    if st.button('predict'):
        response=prediction(input_list)
        st.success(response)

if __name__=='__main__':
    main()
