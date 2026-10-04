# PUBG/BGMI Match Placement Prediction

Supervised regression project predicting a player's final match placement
percentile (`winPlacePerc`, 0 to 1) from match stats.

## Setup

1. Download the dataset from Kaggle:
   https://www.kaggle.com/c/pubg-finish-placement-prediction/data
2. Place `train_V2.csv` in the `data/` folder.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Train the models

```
cd src
python train.py --data ../data/train_V2.csv --sample_frac 0.1
```

- `--sample_frac` controls what fraction of matches to use (by matchId, so
  full matches stay together). Start with `0.1` to iterate quickly;
  drop the flag entirely for a final full-dataset run.
- Trains Linear Regression, Random Forest, and LightGBM, prints validation
  MAE for each, and saves the best model to `best_model.pkl`.

## Run the demo

```
streamlit run app/streamlit_app.py
```

Enter match stats in the sidebar to get a live predicted placement
percentile.

## Project structure

```
pubg-placement/
├── data/              # put train_V2.csv here (not committed)
├── src/
│   ├── features.py    # feature engineering pipeline
│   └── train.py        # trains + compares the 3 models
├── app/
│   └── streamlit_app.py  # demo UI
├── requirements.txt
└── README.md
```

## Pipeline summary

1. Clean data (drop cheater rows, cap unrealistic kill counts)
2. Normalize stats by match size (`playersJoined`)
3. Engineer efficiency ratios (damage-per-kill, kills-per-distance, etc.)
4. Compute team (group) aggregates across teammates
5. Convert group aggregates into **match-relative percentile ranks**
   (the key feature set — placement itself is a rank, not an absolute number)
6. Split train/validation **by matchId** to avoid teammate leakage
7. Train and compare Linear Regression, Random Forest, LightGBM (MAE)
8. Clip/snap predictions to the valid [0, 1] range
9. (Next step) SHAP for feature importance / interpretability
