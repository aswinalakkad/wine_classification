import streamlit as st
import numpy as np
import pickle 

# Load models
with open('final_model.pkl','rb') as file:
    model = pickle.load(file)

with open('lda.pkl','rb') as file:
    LDA = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

def prediction(input_data):
    scaled_data = scaler.transform(input_data)
    lda_data = LDA.transform(scaled_data)
    pred = model.predict(lda_data)[0]

    if pred == 1:
        return 'Wine 1'
    elif pred == 2:
        return 'Wine 2'
    else:
        return 'Wine 3'

def main():
    st.title('Wine Classification')
    st.subheader('This application classifies wine based on chemical constituents.')

    alc = st.number_input('Alcohol')
    mal_acid = st.number_input('Malic Acid')
    ash = st.number_input('Ash')
    alc_ash = st.number_input('Alcalinity of Ash')
    mag = st.number_input('Magnesium')
    phe = st.number_input('Total Phenols')
    fla = st.number_input('Flavanoids')
    nfla = st.number_input('Nonflavanoid Phenols')
    pro = st.number_input('Proanthocyanins')
    co_i = st.number_input('Color Intensity')
    hue = st.number_input('Hue')
    od = st.number_input('OD280/OD315')
    proline = st.number_input('Proline')

    if st.button('Predict'):
        try:
            input_list = [[
                alc, mal_acid, ash, alc_ash, mag,
                phe, fla, nfla, pro, co_i,
                hue, od, proline
            ]]
            
            input_array = np.array(input_list, dtype=float)
            response = prediction(input_array)
            st.success(response)

        except Exception:
            st.error("Invalid input. Please check all fields.")

if __name__ == '__main__':
    main()
