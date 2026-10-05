"""
Streamlit Web Application for Battle Royale Placement Prediction & Post-Match Analysis.
Includes preset player profiles, interactive match statistics sliders,
placement percentile gauges, and tactical explanations.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="PUBG // Tactical Match Placement Engine",
    page_icon="⚡",
    layout="wide"
)

# Custom Tactical Theme Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;700&family=Rajdhani:wght@600;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 50% 10%, #0d1527 0%, #05070c 100%);
        font-family: 'Rajdhani', sans-serif;
        color: #e2e8f0;
    }
    
    h1, h2, h3 {
        font-family: 'Chakra Petch', sans-serif !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Glowing metric cards */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5), inset 0 0 10px rgba(245, 158, 11, 0.05);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em;
    }
    
    div[data-testid="stMetricValue"] {
        color: #f59e0b !important;
        font-family: 'Chakra Petch', sans-serif !important;
        font-weight: 700 !important;
        text-shadow: 0 0 12px rgba(245, 158, 11, 0.5);
    }
    
    /* Tactical button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        color: #000000 !important;
        font-family: 'Chakra Petch', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.1em;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.8) !important;
    }
</style>
""", unsafe_allow_html=True)

# Root path and model loading
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_model.joblib"

@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

st.markdown("""
<div style="border-bottom: 1px solid rgba(245,158,11,0.3); padding-bottom: 12px; margin-bottom: 20px;">
    <h1 style="color: #ffffff; margin: 0; display: flex; align-items: center; gap: 10px;">
        <span style="color: #f59e0b;">⚡ PUBG</span> TACTICAL PLACEMENT ENGINE
        <span style="font-size: 12px; background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid rgba(245,158,11,0.5); padding: 2px 8px; border-radius: 4px; vertical-align: middle;">v2.0 PRO</span>
    </h1>
    <p style="color: #94a3b8; font-size: 14px; margin: 4px 0 0 0; letter-spacing: 0.05em;">
        POST-MATCH STATISTICAL DEBRIEF & ESTIMATION • ML-POWERED RELATIVE PLACEMENT RANKING
    </p>
</div>
""", unsafe_allow_html=True)

# Preset Profiles
PROFILES = {
    "Custom / Manual Input": {},
    "🔥 The Aggressive Rusher (Hot-Dropper)": {
        "matchType": "squad-fpp", "walkDistance": 850.0, "rideDistance": 0.0,
        "kills": 6, "damageDealt": 680.0, "boosts": 2, "heals": 2,
        "weaponsAcquired": 5, "headshotKills": 3, "assists": 1, "DBNOs": 4, "killPlace": 8
    },
    "🌿 The Stealth Survivalist (Tactical Camper)": {
        "matchType": "squad-fpp", "walkDistance": 2800.0, "rideDistance": 1200.0,
        "kills": 1, "damageDealt": 110.0, "boosts": 6, "heals": 5,
        "weaponsAcquired": 4, "headshotKills": 0, "assists": 0, "DBNOs": 0, "killPlace": 32
    },
    "🏆 The Balanced Team MVP": {
        "matchType": "squad-fpp", "walkDistance": 2400.0, "rideDistance": 1800.0,
        "kills": 4, "damageDealt": 490.0, "boosts": 5, "heals": 4,
        "weaponsAcquired": 6, "headshotKills": 2, "assists": 2, "DBNOs": 3, "killPlace": 12
    },
    "💀 The Early Casualty": {
        "matchType": "squad-fpp", "walkDistance": 120.0, "rideDistance": 0.0,
        "kills": 0, "damageDealt": 25.0, "boosts": 0, "heals": 0,
        "weaponsAcquired": 1, "headshotKills": 0, "assists": 0, "DBNOs": 0, "killPlace": 85
    }
}

# Sidebar Preset Selector
st.sidebar.header("🎮 Player Profiles")
selected_profile = st.sidebar.selectbox("Choose a playstyle template:", list(PROFILES.keys()))
profile_vals = PROFILES[selected_profile]

st.sidebar.markdown("---")
st.sidebar.subheader("Match Mode")
match_type = st.sidebar.selectbox(
    "Match Type",
    ["squad-fpp", "duo-fpp", "solo-fpp", "squad", "duo", "solo"],
    index=0 if "matchType" not in profile_vals else ["squad-fpp", "duo-fpp", "solo-fpp", "squad", "duo", "solo"].index(profile_vals["matchType"])
)

# Input columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏃 Movement & Mobility")
    walk_dist = st.slider("Walk Distance (meters)", 0.0, 5000.0, float(profile_vals.get("walkDistance", 1400.0)), step=50.0)
    ride_dist = st.slider("Ride Distance (meters)", 0.0, 8000.0, float(profile_vals.get("rideDistance", 400.0)), step=100.0)
    swim_dist = st.slider("Swim Distance (meters)", 0.0, 1000.0, 0.0, step=10.0)

with col2:
    st.subheader("⚔️ Combat Performance")
    kills = st.slider("Kills", 0, 20, int(profile_vals.get("kills", 2)))
    damage = st.slider("Damage Dealt", 0.0, 2500.0, float(profile_vals.get("damageDealt", 260.0)), step=25.0)
    headshots = st.slider("Headshot Kills", 0, kills, min(int(profile_vals.get("headshotKills", 1)), kills))
    assists = st.slider("Assists", 0, 10, int(profile_vals.get("assists", 1)))
    dbnos = st.slider("DBNOs (Knockdowns)", 0, 15, int(profile_vals.get("DBNOs", 2)))
    kill_place = st.slider("Kill Place Ranking in Match", 1, 100, int(profile_vals.get("killPlace", 28)))

with col3:
    st.subheader("🎒 Tactical & Survival Items")
    boosts = st.slider("Boost Items Used", 0, 20, int(profile_vals.get("boosts", 3)))
    heals = st.slider("Healing Items Used", 0, 30, int(profile_vals.get("heals", 3)))
    weapons = st.slider("Weapons Acquired", 0, 25, int(profile_vals.get("weaponsAcquired", 4)))
    revives = st.slider("Teammates Revived", 0, 5, 0)

st.markdown("---")

if st.button("🚀 Predict Match Placement", use_container_width=True, type="primary"):
    artifact = load_model_artifact()
    if artifact is None:
        st.error("Model artifact not found. Please train the model first.")
    else:
        model = artifact["model"]
        feature_cols = artifact["features"]
        
        # Calculate engineered features
        total_dist = walk_dist + ride_dist + swim_dist
        row = {
            'assists': assists, 'boosts': boosts, 'damageDealt': damage, 'DBNOs': dbnos,
            'headshotKills': headshots, 'heals': heals, 'killPlace': kill_place,
            'killPoints': 1000, 'kills': kills, 'killStreaks': 1, 'longestKill': 40.0,
            'matchDuration': 1400, 'maxPlace': 28 if "squad" in match_type else 48 if "duo" in match_type else 96,
            'numGroups': 27 if "squad" in match_type else 46 if "duo" in match_type else 92,
            'rankPoints': 1400, 'revives': revives, 'rideDistance': ride_dist, 'roadKills': 0,
            'swimDistance': swim_dist, 'teamKills': 0, 'vehicleDestroys': 0,
            'walkDistance': walk_dist, 'weaponsAcquired': weapons, 'winPoints': 1000,
            'players_in_match': 95, 'groups_in_match': 27, 'group_size': 4 if "squad" in match_type else 2 if "duo" in match_type else 1,
            'totalDistance': total_dist,
            'damage_per_kill': damage / (kills + 1.0),
            'headshot_rate': headshots / (kills + 1.0),
            'kills_per_walkDistance': kills / (walk_dist + 1.0),
            'heals_per_walkDistance': heals / (walk_dist + 1.0),
            'boosts_per_walkDistance': boosts / (walk_dist + 1.0),
            'heals_and_boosts': heals + boosts,
            'items_per_walkDistance': (heals + boosts + weapons) / (walk_dist + 1.0),
            'kills_per_totalDistance': kills / (total_dist + 1.0),
            'walkDistance_team_mean': walk_dist,
            'walkDistance_team_max': walk_dist,
            'walkDistance_team_min': walk_dist,
            'walkDistance_team_sum': walk_dist * (4 if "squad" in match_type else 2 if "duo" in match_type else 1),
            'totalDistance_team_mean': total_dist,
            'totalDistance_team_max': total_dist,
            'totalDistance_team_min': total_dist,
            'totalDistance_team_sum': total_dist * (4 if "squad" in match_type else 2 if "duo" in match_type else 1),
            'kills_team_mean': float(kills),
            'kills_team_max': float(kills),
            'kills_team_min': float(kills),
            'kills_team_sum': float(kills * (4 if "squad" in match_type else 2 if "duo" in match_type else 1)),
            'damageDealt_team_mean': damage,
            'damageDealt_team_max': damage,
            'damageDealt_team_min': damage,
            'damageDealt_team_sum': damage * (4 if "squad" in match_type else 2 if "duo" in match_type else 1),
            'boosts_team_mean': float(boosts), 'boosts_team_max': float(boosts), 'boosts_team_min': float(boosts), 'boosts_team_sum': float(boosts * 4),
            'heals_team_mean': float(heals), 'heals_team_max': float(heals), 'heals_team_min': float(heals), 'heals_team_sum': float(heals * 4),
            'weaponsAcquired_team_mean': float(weapons), 'weaponsAcquired_team_max': float(weapons), 'weaponsAcquired_team_min': float(weapons), 'weaponsAcquired_team_sum': float(weapons * 4),
            'DBNOs_team_mean': float(dbnos), 'DBNOs_team_max': float(dbnos), 'DBNOs_team_min': float(dbnos), 'DBNOs_team_sum': float(dbnos * 4),
            'killPlace_team_mean': float(kill_place), 'killPlace_team_max': float(kill_place), 'killPlace_team_min': float(kill_place), 'killPlace_team_sum': float(kill_place * 4),
            'walkDistance_match_mean_ratio': walk_dist / 1150.0,
            'walkDistance_match_max_ratio': walk_dist / 4000.0,
            'walkDistance_match_rank_perc': min(walk_dist / 3000.0, 1.0),
            'totalDistance_match_mean_ratio': total_dist / 1750.0,
            'totalDistance_match_max_ratio': total_dist / 6000.0,
            'totalDistance_match_rank_perc': min(total_dist / 5000.0, 1.0),
            'damageDealt_match_mean_ratio': damage / 130.0,
            'damageDealt_match_max_ratio': damage / 800.0,
            'damageDealt_match_rank_perc': min(damage / 600.0, 1.0),
            'kills_match_mean_ratio': kills / 0.9,
            'kills_match_max_ratio': kills / 8.0,
            'kills_match_rank_perc': min(kills / 6.0, 1.0),
            'boosts_match_mean_ratio': boosts / 1.1,
            'boosts_match_max_ratio': boosts / 7.0,
            'boosts_match_rank_perc': min(boosts / 6.0, 1.0),
            'heals_match_mean_ratio': heals / 1.3,
            'heals_match_max_ratio': heals / 8.0,
            'heals_match_rank_perc': min(heals / 7.0, 1.0),
            'weaponsAcquired_match_mean_ratio': weapons / 3.6,
            'weaponsAcquired_match_max_ratio': weapons / 12.0,
            'weaponsAcquired_match_rank_perc': min(weapons / 10.0, 1.0),
            'killPlace_match_mean_ratio': kill_place / 47.0,
            'killPlace_match_max_ratio': kill_place / 96.0,
            'killPlace_match_rank_perc': 1.0 - (kill_place / 96.0)
        }
        
        input_df = pd.DataFrame([row])
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0.0
                
        input_df = input_df[feature_cols].astype(np.float32)
        X_pred = np.ascontiguousarray(input_df.values, dtype=np.float32)
        pred_percentile = float(np.clip(model.predict(X_pred)[0], 0.0, 1.0))
        
        # Display Prediction Metrics
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Predicted Placement Percentile", f"{pred_percentile * 100:.1f}%")
        with res_col2:
            top_pct = (1.0 - pred_percentile) * 100
            st.metric("Estimated Standing", f"Top {top_pct:.1f}% of match")
        with res_col3:
            if pred_percentile >= 0.85:
                tier = "🌟 Top 15% (Challenger Tier)"
            elif pred_percentile >= 0.60:
                tier = "🥈 Above Average Finisher"
            elif pred_percentile >= 0.35:
                tier = "🥉 Mid-Tier Finisher"
            else:
                tier = "⚠️ Early Elimination"
            st.metric("Play Performance Tier", tier)
            
        # Visual Charts (Dark Tactical Gauge + Polar Radar Chart)
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_percentile * 100,
                title={'text': "<b>PLACEMENT PERCENTILE</b>", 'font': {'color': '#f59e0b', 'size': 16}},
                number={'suffix': "%", 'font': {'color': '#ffffff', 'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#94a3b8'},
                    'bar': {'color': "#f59e0b"},
                    'bgcolor': "#090d16",
                    'borderwidth': 1,
                    'bordercolor': "rgba(245,158,11,0.3)",
                    'steps': [
                        {'range': [0, 25], 'color': "rgba(239,68,68,0.25)"},
                        {'range': [25, 50], 'color': "rgba(234,179,8,0.25)"},
                        {'range': [50, 75], 'color': "rgba(6,182,212,0.25)"},
                        {'range': [75, 100], 'color': "rgba(16,185,129,0.35)"}
                    ],
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with chart_col2:
            categories = ['Combat', 'Mobility', 'Survival', 'Headshot', 'Inventory']
            stat_values = [
                min(damage / 700.0, 1.0) * 100,
                min(walk_dist / 3000.0, 1.0) * 100,
                min(boosts / 6.0, 1.0) * 100,
                min((headshots / (kills + 1.0)) * 1.5, 1.0) * 100,
                min((boosts + heals + weapons) / 15.0, 1.0) * 100
            ]
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=stat_values,
                theta=categories,
                fill='toself',
                fillcolor='rgba(245, 158, 11, 0.35)',
                line=dict(color='#f59e0b', width=2),
                marker=dict(color='#ffffff', size=6)
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color='#64748b', showticklabels=False),
                    bgcolor='rgba(15, 23, 42, 0.6)'
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=30, r=30, t=40, b=20),
                title=dict(text="<b>TACTICAL RADAR ATTRIBUTES</b>", font=dict(color='#06b6d4', size=16), x=0.5)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Key Drivers
        st.subheader("📊 Key Tactical Drivers")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🟢 Positive Contributors")
            if walk_dist > 1500:
                st.write(f"- **High Mobility ({walk_dist:.0f}m)**: Excellent zone repositioning and circle survival.")
            if boosts >= 3:
                st.write(f"- **Active Boost Usage ({boosts} boosts)**: Maintained sprint speed and continuous health regen.")
            if kills >= 3:
                st.write(f"- **Combat Impact ({kills} kills, {damage:.0f} dmg)**: Eliminated threats directly.")
            if kill_place <= 25:
                st.write(f"- **Top Kill Rank (#{kill_place})**: High kill standing protected team position.")
        with c2:
            st.markdown("##### 🔴 Negative / Limiting Factors")
            if walk_dist < 600:
                st.write(f"- **Low Mobility ({walk_dist:.0f}m)**: Limited zone coverage; risk of circle pinch.")
            if boosts == 0:
                st.write("- **Zero Boost Items**: Slower movement and vulnerability to zone damage ticks.")
            if kill_place > 60:
                st.write(f"- **Low Kill Rank (#{kill_place})**: Reduced combat agency.")
