
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cancer Prediction Prototype", layout="wide")

# ---- Custom CSS ----
st.markdown(
    """
    <style>
    body, .stApp {
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Arial Black', sans-serif;
    }
    .stButton>button {
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Logos + Teams ----
st.markdown("## Partner Research Groups")


col1, col2, _ = st.columns([1, 1, 6])

with col1:
    st.image("mllogo.JPG", width=150)
    st.markdown("[**ML-Healthcare Group**](https://mlhealthcare.framer.website/)")

with col2:
    st.image("logo_TP.png", width=150)
    st.markdown("[**SpentaGen Group**](https://spentagen.com/)")
    

st.title("Cancer Prediction Prototype")
st.caption("⚠️ This tool is a research prototype only. It does not replace medical diagnosis or consultation.")

# ---- Load model ----
with open("model.pkl", "rb") as f:
    model, top_features, X_test, y_test, scaler_choice = pickle.load(f)

st.subheader("Enter values for the 10 selected features (raw input; normalized internally with " + scaler_choice + ")")
inputs = {}
cols = st.columns(2)
for i, f in enumerate(top_features):
    with cols[i % 2]:
        val = st.number_input(f, value=float(X_test[f].mean()))
        inputs[f] = val

if st.button("Run Prediction"):
    user_df = pd.DataFrame([inputs])
    proba = model.predict_proba(user_df)[0,1]
    pred = int(proba >= 0.5)

    if pred == 1:
        st.error(f"🔴 Model result (prototype): High cancer risk (probability = {proba:.3f})")
    else:
        st.success(f"🟢 Model result (prototype): Low cancer risk (probability = {1-proba:.3f})")

    st.info("⚠️ This result is experimental. Please consult a physician for further evaluation.")

    # Classification report
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    # ROC curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1],"--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# ---- References ----
st.markdown("## References")
st.markdown(""" 
- [WHFDL: an explainable method based on World Hyper-heuristic and Fuzzy Deep Learning approaches for gastric cancer detection using metabolomics data](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-025-00486-1)  
- [Metabolomic machine learning predictor for diagnosis and prognosis of gastric cancer](https://www.nature.com/articles/s41467-024-46043-y)
""")
