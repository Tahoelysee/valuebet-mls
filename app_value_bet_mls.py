
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Value Bet MLS - Over 2.5", layout="wide")

st.title("⚽ Value Bet Detector - MLS Over 2.5")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset_mls.csv")
        return df
    except:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("📂 Le fichier dataset_mls.csv est introuvable ou vide.")
else:
    st.success(f"✅ {len(df)} matchs chargés.")

    # Préparation des données
    features = df.drop(columns=["date", "home_team", "away_team", "over25", "goals_home", "goals_away", "total_goals"], errors="ignore")
    features = features.select_dtypes(include=["number"]).fillna(0)
    target = df["over25"]

    # Division et entraînement rapide d’un modèle IA
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.subheader("🔍 Analyse IA - Probabilités des matchs Over 2.5")

    # Prédictions sur les 10 derniers matchs
    last_matches = df.tail(10).copy()
    last_features = last_matches[features.columns].fillna(0)
    preds = model.predict_proba(last_features)[:, 1]
    last_matches["proba_over25 (%)"] = (preds * 100).round(1)
    last_matches["cote_book"] = last_matches["over25_odd"]
    last_matches["value_bet"] = (preds * last_matches["cote_book"]) > 1.1

    st.dataframe(last_matches[[
        "date", "home_team", "away_team", "goals_home", "goals_away",
        "proba_over25 (%)", "cote_book", "value_bet"
    ]].sort_values(by="proba_over25 (%)", ascending=False))

    st.info("🧠 Value bet détecté si : (Proba x Cote) > 1.1")
