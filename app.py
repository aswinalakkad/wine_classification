import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Load Saved Models
# -----------------------------
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('lda.pkl', 'rb') as file:
    LDA = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# -----------------------------
# Prediction Function
# -----------------------------
def prediction(input_data):
    scaled_data = scaler.transform(input_data)
    lda_data = LDA.transform(scaled_data)
    pred = model.predict(lda_data)[0]

    if pred == 1:
        return "Wine 1"
    elif pred == 2:
        return "Wine 2"
    else:
        return "Wine 3"


# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("🍷 Wine Classification App")
    st.subheader("Classify wine based on its chemical properties")

    alc = st.number_input(
        "Alcohol",
        value=None,
        help="Alcohol percentage present in wine"
    )

    mal_acid = st.number_input(
        "Malic Acid",
        value=None,
        help="Amount of malic acid in wine"
    )

    ash = st.number_input(
        "Ash",
        value=None,
        help="Total ash content"
    )

    alc_ash = st.number_input(
        "Alcalinity of Ash",
        value=None,
        help="Alkalinity level of ash"
    )

    mag = st.number_input(
        "Magnesium",
        value=None,
        help="Magnesium concentration"
    )

    phe = st.number_input(
        "Total Phenols",
        value=None,
        help="Total phenolic compounds"
    )

    fla = st.number_input(
        "Flavanoids",
        value=None,
        help="Amount of flavanoids"
    )

    nfla = st.number_input(
        "Nonflavanoid Phenols",
        value=None,
        help="Non-flavanoid phenolic compounds"
    )

    pro = st.number_input(
        "Proanthocyanins",
        value=None,
        help="Proanthocyanin concentration"
    )

    co_i = st.number_input(
        "Color Intensity",
        value=None,
        help="Wine color intensity"
    )

    hue = st.number_input(
        "Hue",
        value=None,
        help="Hue value of wine"
    )

    od = st.number_input(
        "OD280/OD315",
        value=None,
        help="OD ratio of diluted wine"
    )

    proline = st.number_input(
        "Proline",
        value=None,
        help="Proline concentration"
    )

    if st.button("Predict"):
        inputs = [
            alc, mal_acid, ash, alc_ash, mag,
            phe, fla, nfla, pro, co_i,
            hue, od, proline
        ]

        if None in inputs:
            st.warning("Please fill all fields.")
        else:
            input_array = np.array([inputs], dtype=float)
            result = prediction(input_array)
            st.success(f"Predicted Class: {result}")


if __name__ == "__main__":
    main()
