"""
Streamlit demo: enter match stats, get a predicted placement percentile.

Run with:
    streamlit run app/streamlit_app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PUBG Placement Predictor", page_icon="🎮")
st.title("🎮 PUBG/BGMI Placement Predictor")
st.write("Enter your match stats to estimate your final placement percentile.")

@st.cache_resource
def load_model():
    bundle = joblib.load("best_model.pkl")
    return bundle["model"], bundle["features"]

model, feature_cols = load_model()

col1, col2 = st.columns(2)
with col1:
    kills = st.number_input("Kills", 0, 50, 2)
    damage_dealt = st.number_input("Damage dealt", 0, 5000, 250)
    walk_distance = st.number_input("Walk distance (m)", 0, 20000, 1200)
    ride_distance = st.number_input("Ride distance (m)", 0, 40000, 0)
with col2:
    heals = st.number_input("Heal items used", 0, 50, 2)
    boosts = st.number_input("Boost items used", 0, 50, 2)
    weapons_acquired = st.number_input("Weapons acquired", 0, 20, 3)
    match_type = st.selectbox(
        "Match type", ["solo", "duo", "squad", "solo-fpp", "duo-fpp", "squad-fpp"]
    )

if st.button("Predict placement"):
    # Build a single-row input matching the training feature space.
    # Missing engineered features (group aggregates, match-relative ranks)
    # default to the raw player's own stats, since there's no live match
    # context for a standalone prediction.
    row = {col: 0 for col in feature_cols}
    row.update({
        "kills": kills,
        "damageDealt": damage_dealt,
        "walkDistance": walk_distance,
        "rideDistance": ride_distance,
        "heals": heals,
        "boosts": boosts,
        "weaponsAcquired": weapons_acquired,
        "healsAndBoosts": heals + boosts,
        "totalDistance": walk_distance + ride_distance,
        "damagePerKill": damage_dealt / (kills + 1),
        "killsPerWalkDistance": kills / (walk_distance + 1),
    })
    match_type_col = f"matchType_{match_type}"
    if match_type_col in row:
        row[match_type_col] = 1

    X = pd.DataFrame([row])[feature_cols]
    pred = float(np.clip(model.predict(X)[0], 0, 1))

    st.metric("Predicted placement percentile", f"{pred:.2%}")
    st.caption(
        f"This means you're predicted to finish ahead of roughly "
        f"{pred*100:.0f}% of the field in this match."
    )
