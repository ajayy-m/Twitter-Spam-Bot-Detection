import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

# ================= LOAD RANDOM FOREST MODEL =================
model = joblib.load("random_forest_bot_classifier.pkl")
feature_list = joblib.load("random_forest_feature_list.pkl")
ml_cols = joblib.load("random_forest_ml_feature_cols.pkl")

# ================= LOAD SCALERS =================
robust_scaler = joblib.load("robust_scaler.pkl")
age_scaler = joblib.load("age_scaler.pkl")
embedding_normalizer = joblib.load("embedding_normalizer.pkl")

# ================= LOAD DATA =================
ref_emb = pd.read_csv("training_embeddings_reference.csv")
train_tabular = pd.read_csv("training_tabular_reference.csv")
api_df = pd.read_excel("api_file.xlsx", dtype={"user_id": str})

# Rename emb_0 -> 0 if needed
rename_map = {f"emb_{i}": str(i) for i in range(64)}
ref_emb = ref_emb.rename(columns=rename_map)

# ================= PREPARE NN =================
emb_cols = [c for c in ref_emb.columns if c.isdigit()]
DROP_SIM_COLS = ["user_id", "user_name", "Label", "dataset"]
similarity_cols = [c for c in train_tabular.columns if c not in DROP_SIM_COLS]

nn = NearestNeighbors(n_neighbors=10, metric="cosine")
nn.fit(train_tabular[similarity_cols])

# 🔥 FREEZE QUANTILE (important fix)
followers_95_quantile = train_tabular["followers_count"].quantile(0.95)

results = []
all_probs = []  # 🔥 fixed placement

# ================= LOOP =================
for idx in range(len(api_df)):

    row = api_df.iloc[[idx]].copy()

    # -------- Feature Engineering --------
    row["tweets_per_day"] = row["tweets_count"] / row["account_age_days"]
    row["followers_per_day"] = row["followers_count"] / row["account_age_days"]

    row["log_tweets_per_day"] = np.log1p(row["tweets_per_day"])
    row["log_followers_per_day"] = np.log1p(row["followers_per_day"])

    row["followers_spike"] = row["followers_count"] / np.sqrt(row["account_age_days"])
    row["tweet_spike"] = row["tweets_count"] / np.sqrt(row["account_age_days"])

    row["extreme_activity_score"] = (
        row["log_followers_per_day"] + row["log_tweets_per_day"]
    )

    row["young_account_flag"] = (row["account_age_days"] < 90).astype(int)

    # 🔥 USE FROZEN QUANTILE (fixed)
    row["huge_followers_flag"] = (
        row["followers_count"] > followers_95_quantile
    ).astype(int)

    row["young_and_popular"] = (
        row["young_account_flag"] * row["huge_followers_flag"]
    )

    # -------- Nearest Neighbor Embedding --------
    for c in similarity_cols:
        if c not in row.columns:
            row[c] = 0

    row[similarity_cols] = row[similarity_cols].fillna(0)

    _, idx_nn = nn.kneighbors(row[similarity_cols])
    neighbor_ids = train_tabular.iloc[idx_nn[0]]["user_id"]

    new_embedding = (
        ref_emb[ref_emb["user_id"].isin(neighbor_ids)][emb_cols]
        .mean()
        .values
    )

    emb_df = pd.DataFrame([new_embedding], columns=emb_cols, index=row.index)
    row = pd.concat([row, emb_df], axis=1)

    # -------- L2 Normalize Embeddings --------
    row[emb_cols] = embedding_normalizer.transform(row[emb_cols])

    # -------- Scaling --------
    robust_cols = robust_scaler.feature_names_in_

    for c in robust_cols:
        if c not in row.columns:
            row[c] = 0

    row[robust_cols] = robust_scaler.transform(row[robust_cols])

    if "account_age_days" in row.columns:
        row[["account_age_days"]] = age_scaler.transform(
            row[["account_age_days"]]
        )

    # -------- Ensure Feature Order --------
    for c in feature_list:
        if c not in row.columns:
            row[c] = 0

    X_user = row[feature_list]

    # -------- Prediction --------
    prob = model.predict_proba(X_user)[:, 1][0]
    all_probs.append(prob)  # 🔥 correct tracking

    label = "HUMAN" if prob >= 0.39 else "BOT"

    results.append(
        {
            "user_id": row["user_id"].values[0],
            "probability": prob,
            "prediction": label,
        }
    )

# ================= SAVE OUTPUT =================
results_df = pd.DataFrame(results)
results_df.to_excel("batch_predictions.xlsx", index=False)

print("Batch prediction completed.")
print("Saved to: batch_predictions.xlsx")

# 🔥 Proper probability diagnostics
print("Min prob:", min(all_probs))
print("Max prob:", max(all_probs))
print("Mean prob:", sum(all_probs) / len(all_probs))
print("Non-zero feature count:", (X_user != 0).sum(axis=1).values[0])
print("Model classes:", model.classes_)