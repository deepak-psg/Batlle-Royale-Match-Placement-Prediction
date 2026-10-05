"""
FastAPI Serving Endpoint for PUBG Match Placement Prediction.
Accepts player match statistics, computes real-time feature transformations,
and returns placement percentile predictions alongside key explanatory factors.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI(
    title="Battle Royale Placement Predictor API",
    description="Production endpoint for PUBG finish placement percentile regression and tactical analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(__file__).resolve().parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

reports_path = Path(__file__).resolve().parent.parent / "reports"
if reports_path.exists():
    app.mount("/reports", StaticFiles(directory=str(reports_path)), name="reports")

@app.get("/")
def get_index():
    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Battle Royale Placement API is running. Access /docs for API documentation."}

# Global model store
MODEL_DATA = None

class PlayerStatsInput(BaseModel):
    matchType: str = Field(default="squad-fpp", description="PUBG match type (squad-fpp, duo-fpp, solo-fpp, etc.)")
    walkDistance: float = Field(default=1200.0, ge=0.0, description="Total distance walked in meters")
    rideDistance: float = Field(default=0.0, ge=0.0, description="Total distance driven in meters")
    swimDistance: float = Field(default=0.0, ge=0.0, description="Total distance swam in meters")
    kills: int = Field(default=2, ge=0, description="Number of enemy players killed")
    damageDealt: float = Field(default=250.0, ge=0.0, description="Total damage dealt to enemies")
    boosts: int = Field(default=2, ge=0, description="Number of boost items used")
    heals: int = Field(default=3, ge=0, description="Number of healing items used")
    weaponsAcquired: int = Field(default=4, ge=0, description="Number of weapons picked up")
    headshotKills: int = Field(default=1, ge=0, description="Number of enemies killed with headshots")
    assists: int = Field(default=1, ge=0, description="Number of enemy players damaged that were killed by teammates")
    DBNOs: int = Field(default=1, ge=0, description="Number of enemy players knocked down")
    revives: int = Field(default=0, ge=0, description="Number of times revived a teammate")
    killPlace: int = Field(default=30, ge=1, le=100, description="Ranking in match in terms of kills")
    killStreaks: int = Field(default=1, ge=0, description="Max number of enemy players killed in a short time")
    longestKill: float = Field(default=45.0, ge=0.0, description="Longest distance of a kill in meters")
    teamKills: int = Field(default=0, ge=0, description="Number of teammates killed")
    vehicleDestroys: int = Field(default=0, ge=0, description="Number of vehicles destroyed")

    # Optional team stats
    teammates_count: int = Field(default=3, ge=0, le=3, description="Number of other teammates in squad")
    team_avg_kills: Optional[float] = Field(default=None, description="Average kills across team")
    team_avg_walkDistance: Optional[float] = Field(default=None, description="Average walk distance across team")

class PredictionResponse(BaseModel):
    predicted_placement_percentile: float
    percentile_tier: str
    summary_message: str
    top_positive_factors: List[Dict[str, Any]]
    top_negative_factors: List[Dict[str, Any]]

def load_model():
    global MODEL_DATA
    if MODEL_DATA is None:
        model_path = Path(__file__).resolve().parent.parent / "models" / "lgbm_model.joblib"
        if not model_path.exists():
            raise RuntimeError(f"Model file not found at {model_path}. Train the model first.")
        MODEL_DATA = joblib.load(model_path)
    return MODEL_DATA

@app.on_event("startup")
def startup_event():
    try:
        load_model()
        print("[+] Model loaded successfully at startup.")
    except Exception as e:
        print(f"[!] Warning: Model could not be loaded at startup: {e}")

@app.get("/health")
def health():
    model_loaded = MODEL_DATA is not None
    return {"status": "ok", "model_loaded": model_loaded}

@app.post("/predict", response_model=PredictionResponse)
def predict_placement(player: PlayerStatsInput):
    data = load_model()
    model = data["model"]
    feature_cols = data["features"]
    
    # Map input into full engineered feature vector
    row = {
        'assists': player.assists,
        'boosts': player.boosts,
        'damageDealt': player.damageDealt,
        'DBNOs': player.DBNOs,
        'headshotKills': player.headshotKills,
        'heals': player.heals,
        'killPlace': player.killPlace,
        'killPoints': 1000,
        'kills': player.kills,
        'killStreaks': player.killStreaks,
        'longestKill': player.longestKill,
        'matchDuration': 1400,
        'maxPlace': 28 if "squad" in player.matchType else 48 if "duo" in player.matchType else 96,
        'numGroups': 27 if "squad" in player.matchType else 46 if "duo" in player.matchType else 92,
        'rankPoints': 1400,
        'revives': player.revives,
        'rideDistance': player.rideDistance,
        'roadKills': 0,
        'swimDistance': player.swimDistance,
        'teamKills': player.teamKills,
        'vehicleDestroys': player.vehicleDestroys,
        'walkDistance': player.walkDistance,
        'weaponsAcquired': player.weaponsAcquired,
        'winPoints': 1000,
        'players_in_match': 95,
        'groups_in_match': 27,
        'group_size': player.teammates_count + 1
    }
    
    total_dist = player.walkDistance + player.rideDistance + player.swimDistance
    row['totalDistance'] = total_dist
    row['damage_per_kill'] = player.damageDealt / (player.kills + 1.0)
    row['headshot_rate'] = player.headshotKills / (player.kills + 1.0)
    row['kills_per_walkDistance'] = player.kills / (player.walkDistance + 1.0)
    row['heals_per_walkDistance'] = player.heals / (player.walkDistance + 1.0)
    row['boosts_per_walkDistance'] = player.boosts / (player.walkDistance + 1.0)
    row['heals_and_boosts'] = player.heals + player.boosts
    row['items_per_walkDistance'] = (player.heals + player.boosts + player.weaponsAcquired) / (player.walkDistance + 1.0)
    row['kills_per_totalDistance'] = player.kills / (total_dist + 1.0)
    
    # Team aggregates approximation (defaulting to individual stats if team stats omitted)
    team_walk = player.team_avg_walkDistance if player.team_avg_walkDistance is not None else player.walkDistance
    team_kills = player.team_avg_kills if player.team_avg_kills is not None else float(player.kills)
    
    row['walkDistance_team_mean'] = team_walk
    row['walkDistance_team_max'] = max(team_walk, player.walkDistance)
    row['walkDistance_team_min'] = min(team_walk, player.walkDistance)
    row['walkDistance_team_sum'] = team_walk * (player.teammates_count + 1)
    
    row['totalDistance_team_mean'] = team_walk
    row['totalDistance_team_max'] = team_walk
    row['totalDistance_team_min'] = team_walk
    row['totalDistance_team_sum'] = team_walk * (player.teammates_count + 1)
    
    row['kills_team_mean'] = team_kills
    row['kills_team_max'] = max(team_kills, float(player.kills))
    row['kills_team_min'] = min(team_kills, float(player.kills))
    row['kills_team_sum'] = team_kills * (player.teammates_count + 1)
    
    row['damageDealt_team_mean'] = player.damageDealt
    row['damageDealt_team_max'] = player.damageDealt
    row['damageDealt_team_min'] = player.damageDealt
    row['damageDealt_team_sum'] = player.damageDealt * (player.teammates_count + 1)
    
    row['boosts_team_mean'] = float(player.boosts)
    row['boosts_team_max'] = float(player.boosts)
    row['boosts_team_min'] = float(player.boosts)
    row['boosts_team_sum'] = float(player.boosts * (player.teammates_count + 1))
    
    row['heals_team_mean'] = float(player.heals)
    row['heals_team_max'] = float(player.heals)
    row['heals_team_min'] = float(player.heals)
    row['heals_team_sum'] = float(player.heals * (player.teammates_count + 1))
    
    row['weaponsAcquired_team_mean'] = float(player.weaponsAcquired)
    row['weaponsAcquired_team_max'] = float(player.weaponsAcquired)
    row['weaponsAcquired_team_min'] = float(player.weaponsAcquired)
    row['weaponsAcquired_team_sum'] = float(player.weaponsAcquired * (player.teammates_count + 1))
    
    row['DBNOs_team_mean'] = float(player.DBNOs)
    row['DBNOs_team_max'] = float(player.DBNOs)
    row['DBNOs_team_min'] = float(player.DBNOs)
    row['DBNOs_team_sum'] = float(player.DBNOs * (player.teammates_count + 1))
    
    row['killPlace_team_mean'] = float(player.killPlace)
    row['killPlace_team_max'] = float(player.killPlace)
    row['killPlace_team_min'] = float(player.killPlace)
    row['killPlace_team_sum'] = float(player.killPlace * (player.teammates_count + 1))
    
    # Match-relative approximations (using dataset match norms)
    row['walkDistance_match_mean_ratio'] = player.walkDistance / 1150.0
    row['walkDistance_match_max_ratio'] = player.walkDistance / 4000.0
    row['walkDistance_match_rank_perc'] = min(player.walkDistance / 3000.0, 1.0)
    
    row['totalDistance_match_mean_ratio'] = total_dist / 1750.0
    row['totalDistance_match_max_ratio'] = total_dist / 6000.0
    row['totalDistance_match_rank_perc'] = min(total_dist / 5000.0, 1.0)
    
    row['damageDealt_match_mean_ratio'] = player.damageDealt / 130.0
    row['damageDealt_match_max_ratio'] = player.damageDealt / 800.0
    row['damageDealt_match_rank_perc'] = min(player.damageDealt / 600.0, 1.0)
    
    row['kills_match_mean_ratio'] = player.kills / 0.9
    row['kills_match_max_ratio'] = player.kills / 8.0
    row['kills_match_rank_perc'] = min(player.kills / 6.0, 1.0)
    
    row['boosts_match_mean_ratio'] = player.boosts / 1.1
    row['boosts_match_max_ratio'] = player.boosts / 7.0
    row['boosts_match_rank_perc'] = min(player.boosts / 6.0, 1.0)
    
    row['heals_match_mean_ratio'] = player.heals / 1.3
    row['heals_match_max_ratio'] = player.heals / 8.0
    row['heals_match_rank_perc'] = min(player.heals / 7.0, 1.0)
    
    row['weaponsAcquired_match_mean_ratio'] = player.weaponsAcquired / 3.6
    row['weaponsAcquired_match_max_ratio'] = player.weaponsAcquired / 12.0
    row['weaponsAcquired_match_rank_perc'] = min(player.weaponsAcquired / 10.0, 1.0)
    
    row['killPlace_match_mean_ratio'] = player.killPlace / 47.0
    row['killPlace_match_max_ratio'] = player.killPlace / 96.0
    row['killPlace_match_rank_perc'] = 1.0 - (player.killPlace / 96.0)
    
    # Construct ordered feature DataFrame
    input_df = pd.DataFrame([row])
    # Fill any missing columns with 0
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0.0
            
    input_df = input_df[feature_cols].astype(np.float32)
    X_pred = np.ascontiguousarray(input_df.values, dtype=np.float32)
    
    pred_val = float(np.clip(model.predict(X_pred)[0], 0.0, 1.0))
    
    # Tier classification
    if pred_val >= 0.90:
        tier = "Champion / Top 10% (Elite)"
    elif pred_val >= 0.75:
        tier = "Top 25% (High Placement)"
    elif pred_val >= 0.50:
        tier = "Top 50% (Above Average)"
    elif pred_val >= 0.25:
        tier = "Bottom 50% (Mid-Game Exit)"
    else:
        tier = "Bottom 25% (Early Elimination)"
        
    summary = f"Predicted Finish: Top {((1.0 - pred_val) * 100):.1f}% of match participants ({pred_val:.3f} percentile)."
    
    # Simulated top drivers based on key feature thresholds
    pos_factors = []
    neg_factors = []
    
    if player.walkDistance > 1500:
        pos_factors.append({"factor": "High Mobility / Zone Rotation", "detail": f"{player.walkDistance:.0f}m covered"})
    elif player.walkDistance < 500:
        neg_factors.append({"factor": "Low Survival Mobility", "detail": f"Only {player.walkDistance:.0f}m moved"})
        
    if player.boosts >= 3:
        pos_factors.append({"factor": "Active Boost Item Usage", "detail": f"{player.boosts} boosts used"})
    elif player.boosts == 0:
        neg_factors.append({"factor": "No Boost Consumption", "detail": "0 boosts consumed"})
        
    if player.kills >= 3:
        pos_factors.append({"factor": "Combat Impact", "detail": f"{player.kills} kills"})
        
    if player.killPlace > 70:
        neg_factors.append({"factor": "Low Kill Ranking", "detail": f"Kill rank #{player.killPlace}"})
    elif player.killPlace < 20:
        pos_factors.append({"factor": "Top Kill Ranking", "detail": f"Kill rank #{player.killPlace}"})
        
    return PredictionResponse(
        predicted_placement_percentile=round(pred_val, 4),
        percentile_tier=tier,
        summary_message=summary,
        top_positive_factors=pos_factors,
        top_negative_factors=neg_factors
    )
