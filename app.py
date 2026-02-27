# ==========================================
# Elite Diabetes Prediction Dashboard
# Enhanced UI / UX Version
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------------------------
# Page Config
# ------------------------------------------
st.set_page_config(
    page_title="Diabetes AI Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------
# Custom Styling
# ------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.metric-card {
    background-color: #1C1F26;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# Hero Section
# ------------------------------------------
st.markdown("""
# 🩺 Diabetes Risk Intelligence Dashboard
### AI-Powered Clinical Decision Support System
""")

st.markdown("---")

# ------------------------------------------
# Load Data
# ------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

# ------------------------------------------
# Train Model
# ------------------------------------------
@st.cache_resource
def train_model():
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    cm = confusion_matrix(y_test, model.predict(X_test))

    return model, accuracy, cm, X.columns

model, accuracy, cm, feature_names = train_model()

# ------------------------------------------
# Sidebar
# ------------------------------------------
st.sidebar.header("📋 Patient Clinical Inputs")

preg = st.sidebar.slider("Pregnancies", 0, 20, 1)
glu = st.sidebar.slider("Glucose Level", 0, 200, 110)
bp = st.sidebar.slider("Blood Pressure", 0, 150, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
ins = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.slider("Age", 1, 100, 30)

input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])

predict_clicked = st.sidebar.button("🔍 Run Prediction")

# ------------------------------------------
# Tabs Layout
# ------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Model Insights", "📂 Dataset"])

# ==========================================
# TAB 1 – PREDICTION DASHBOARD
# ==========================================
with tab1:

    if predict_clicked:

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        col1, col2, col3 = st.columns(3)

        # Metric Cards
        with col1:
            if prediction == 1:
                st.error("🛑 High Risk Patient")
            else:
                st.success("✅ Low Risk Patient")

        with col2:
            st.metric("Risk Probability", f"{probability*100:.2f}%")

        with col3:
            st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

        st.markdown("---")

        # Gauge + Risk Interpretation Side by Side
        col_gauge, col_text = st.columns([2, 1])

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability*100,
                number={'suffix': "%"},
                title={'text': "Diabetes Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF4B4B"},
                    'steps': [
                        {'range': [0, 40], 'color': "#2ECC71"},
                        {'range': [40, 70], 'color': "#F39C12"},
                        {'range': [70, 100], 'color': "#E74C3C"}
                    ],
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_text:
            st.markdown("### 🧠 Risk Interpretation")
            if probability < 0.4:
                st.success("Low clinical concern.")
            elif probability < 0.7:
                st.warning("Moderate monitoring recommended.")
            else:
                st.error("Immediate medical consultation advised.")

    else:
        st.info("Enter patient data in the sidebar and click **Run Prediction**.")

# ==========================================
# TAB 2 – MODEL INSIGHTS
# ==========================================
with tab2:

    colA, colB = st.columns(2)

    # Feature Importance
    importance = model.coef_[0]
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=True)

    with colA:
        fig2 = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Bluered",
            title="Feature Impact on Prediction"
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)

    # Confusion Matrix
    with colB:
        fig3 = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="reds",
            labels=dict(x="Predicted", y="Actual"),
            title="Confusion Matrix"
        )
        fig3.update_layout(height=500)
        st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# TAB 3 – DATASET
# ==========================================
with tab3:
    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.markdown("### 👩‍💻 Developed by Priyanka Sharma | Data Scientist")