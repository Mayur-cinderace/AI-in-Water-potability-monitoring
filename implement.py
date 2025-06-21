import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Load model + preprocessing ---
model     = joblib.load("rf_model_indianwater.pkl")
scaler    = joblib.load("scaler_indianwater.pkl")
imputer   = joblib.load("imputer_indianwater.pkl")
features  = joblib.load("features_indianwater.pkl")

# Mapping raw CSV columns → model feature names
COLUMN_RENAME = {
    "Temp": "Temp",
    "D.O. (mg/l)": "DO",
    "PH": "pH",
    "CONDUCTIVITY": "Conductivity",
    "B.O.D. (mg/l)": "BOD",
    "NITRATENAN N+ NITRITENANN (mg/l)": "NitrateNitrite",
    "FECAL COLIFORM (MPN/100ml)": "FecalColiform",
    "TOTAL COLIFORM (MPN/100ml)Mean": "TotalColiform"
}

st.set_page_config(page_title="Water Potability Predictor", layout="wide")
st.title("💧 Water Potability Prediction Dashboard")

# --- Sidebar: Batch Upload & Download ---
st.sidebar.header("📁 Batch Prediction")
uploaded = st.sidebar.file_uploader(
    "Upload CSV with raw columns:\n" + "\n".join(COLUMN_RENAME.keys()),
    type=["csv"]
)
if uploaded:
    df_batch = pd.read_csv(uploaded, encoding="latin1")
    df_batch = df_batch.rename(columns=COLUMN_RENAME)
    missing = set(features) - set(df_batch.columns)
    if missing:
        st.sidebar.error(f"Missing columns: {missing}")
    else:
        Xb = df_batch[features]
        Xb_imp = imputer.transform(Xb)
        Xb_scl = scaler.transform(Xb_imp)
        preds = model.predict(Xb_scl)
        probs = model.predict_proba(Xb_scl)[:, 1]
        df_batch["Prediction"] = np.where(preds == 1, "SAFE", "UNSAFE")
        df_batch["Confidence"] = np.round(probs, 4)
        st.sidebar.success(f"Processed {len(df_batch)} rows")
        st.sidebar.download_button(
            "⬇️ Download Results",
            df_batch.to_csv(index=False),
            file_name="batch_results.csv",
            mime="text/csv"
        )

# --- Single Sample Input ---
st.subheader("🔧 Single Sample Input")

col1, col2, col3 = st.columns(3)
with col1:
    state        = st.selectbox("State", ["Karnataka","Maharashtra","Delhi","Tamil Nadu","Other"])
    station_code = st.text_input("Station Code", "ST1234")
with col2:
    location     = st.text_input("Location", "Generic River Point")
    year         = st.number_input("Year", 1990, 2030, 2024)
with col3:
    source_type  = st.selectbox("Water Source Type", ["River","Groundwater","Lake","Municipal Tap","Well","Other"])
    season       = st.selectbox("Season", ["Summer","Monsoon","Winter"])
    treatment    = st.radio("Pre-treated?", ["Yes","No"])

st.markdown("---")

user_input = {}
for feat, (mn, mx, df) in {
    "Temp": (5.0, 50.0, 25.0),
    "DO": (0.0, 15.0, 6.0),
    "pH": (0.0, 14.0, 7.0),
    "Conductivity": (50, 2500, 750),
    "BOD": (0.0, 20.0, 3.0),
    "NitrateNitrite": (0.0, 50.0, 10.0),
    "FecalColiform": (0, 10000, 1000),
    "TotalColiform": (0, 10000, 1500)
}.items():
    user_input[feat] = st.slider(f"{feat}", float(mn), float(mx), float(df))

# Prepare data
input_df = pd.DataFrame([user_input])[features]
X_imp    = imputer.transform(input_df)
X_scl    = scaler.transform(X_imp)

# Prediction
pred = model.predict(X_scl)[0]
pro  = model.predict_proba(X_scl)[0][1]

st.markdown("### 🧪 Prediction Result")
if pred == 1:
    st.success(f"✅ SAFE to drink | Confidence: {pro:.2f}")
    st.balloons()
else:
    st.error(f"⚠️ UNSAFE to drink | Confidence: {1-pro:.2f}")
    st.warning("⚠️ Alert: Please consider treatment or alternative source.")

# Interactive Visualizations
st.subheader("📊 Input Feature Values")
st.bar_chart(input_df.T)

# Session history
if "history" not in st.session_state:
    st.session_state.history = []
if "preds" not in st.session_state:
    st.session_state.preds = []
st.session_state.history.append(pro)
st.session_state.preds.append("SAFE" if pred==1 else "UNSAFE")
st.line_chart(pd.DataFrame({"Confidence": st.session_state.history}))

# Prediction count chart
st.subheader("📈 Prediction Counts This Session")
counts = pd.Series(st.session_state.preds).value_counts()
st.bar_chart(counts)

# Last-record log
full_record = {**user_input,
               "State": state, "Location": location, "StationCode": station_code,
               "Year": year, "SourceType": source_type, "Season": season,
               "PreTreated": treatment,
               "Prediction": "SAFE" if pred==1 else "UNSAFE",
               "Confidence": pro}
df_log = pd.DataFrame([full_record])

st.markdown("---")
st.subheader("📋 Last Record Details")
st.write(df_log)

st.download_button(
    "⬇️ Download Last Record",
    df_log.to_csv(index=False),
    file_name="last_record.csv",
    mime="text/csv"
)
