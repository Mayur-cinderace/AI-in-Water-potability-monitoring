import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

df = pd.read_csv("water_dataX.csv", encoding='latin1')
df.columns = df.columns.str.strip()

df = df.rename(columns={
    "D.O. (mg/l)": "DO",
    "PH": "pH",
    "CONDUCTIVITY": "Conductivity",
    "B.O.D. (mg/l)": "BOD",
    "NITRATENAN N+ NITRITENANN (mg/l)": "NitrateNitrite",
    "FECAL COLIFORM (MPN/100ml)": "FecalColiform",
    "TOTAL COLIFORM (MPN/100ml)Mean": "TotalColiform"
})

features = ["Temp", "DO", "pH", "Conductivity", "BOD", "NitrateNitrite", "FecalColiform", "TotalColiform"]
df_selected = df[features].apply(pd.to_numeric, errors='coerce')

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(df_selected), columns=features)

def classify_safe(row):
    return int(
        (6.5 <= row["pH"] <= 8.5) and
        (row["DO"] >= 5.0) and
        (row["BOD"] <= 3.0) and
        (row["FecalColiform"] <= 2500)
    )

X_imputed["Potability"] = X_imputed.apply(classify_safe, axis=1)

X = X_imputed.drop("Potability", axis=1)
y = X_imputed["Potability"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n[Classification Report]")
print(classification_report(y_test, y_pred))
print("\n[Confusion Matrix]")
print(confusion_matrix(y_test, y_pred))
print("\n[ROC AUC Score]")
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

joblib.dump(clf, "rf_model_indianwater.pkl")
joblib.dump(scaler, "scaler_indianwater.pkl")
joblib.dump(imputer, "imputer_indianwater.pkl")
joblib.dump(features, "features_indianwater.pkl")

print("\n✅ Model + Scaler + Imputer + Feature list saved successfully.")
