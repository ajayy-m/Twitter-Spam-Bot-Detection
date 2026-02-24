import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
import lime
import lime.lime_tabular

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

# Freeze quantile on training data (never recompute on test data)
followers_95_quantile = train_tabular["followers_count"].quantile(0.95)

# ================= LIME FEATURE NAMES =================
# LIME operates on RAW features (what the api_file contains).
# The pipeline function handles all engineering + scaling internally.
# We exclude user_id, user_name, Label, dataset — not model inputs.
RAW_INPUT_COLS = [c for c in api_df.columns if c not in
                  ["user_id", "user_name", "Label", "label", "dataset"]]


# ================= SHARED PIPELINE FUNCTION =================
def run_pipeline(row: pd.DataFrame) -> float:
    """
    Takes a single-row DataFrame with raw api_file columns,
    runs the full pipeline, and returns P(HUMAN).
    """
    row = row.copy()

    # -------- Safety clamp: account_age_days must be > 0 for division --------
    row["account_age_days"] = row["account_age_days"].clip(lower=1.0)
    row["tweets_count"]     = row["tweets_count"].clip(lower=0.0)
    row["followers_count"]  = row["followers_count"].clip(lower=0.0)

    # -------- Feature Engineering --------
    row["tweets_per_day"]        = row["tweets_count"]    / row["account_age_days"]
    row["followers_per_day"]     = row["followers_count"] / row["account_age_days"]
    row["log_tweets_per_day"]    = np.log1p(row["tweets_per_day"])
    row["log_followers_per_day"] = np.log1p(row["followers_per_day"])
    row["followers_spike"]       = row["followers_count"] / np.sqrt(row["account_age_days"])
    row["tweet_spike"]           = row["tweets_count"]    / np.sqrt(row["account_age_days"])
    row["extreme_activity_score"] = row["log_followers_per_day"] + row["log_tweets_per_day"]
    row["young_account_flag"]    = (row["account_age_days"] < 90).astype(int)
    row["huge_followers_flag"]   = (row["followers_count"] > followers_95_quantile).astype(int)
    row["young_and_popular"]     = row["young_account_flag"] * row["huge_followers_flag"]

    # -------- Nearest Neighbor Embedding --------
    for c in similarity_cols:
        if c not in row.columns:
            row[c] = 0
    row[similarity_cols] = row[similarity_cols].replace([np.inf, -np.inf], 0).fillna(0)

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
        row[["account_age_days"]] = age_scaler.transform(row[["account_age_days"]])

    # -------- Ensure Feature Order --------
    for c in feature_list:
        if c not in row.columns:
            row[c] = 0

    X_user = row[feature_list]
    return model.predict_proba(X_user)[:, 1][0]


# ================= LIME PREDICT FUNCTION =================
def full_pipeline_for_lime(raw_array: np.ndarray) -> np.ndarray:
    """
    Called by LIME with shape (n_perturbations, n_raw_features).
    Reconstructs a DataFrame from raw values, runs full pipeline,
    returns (n_perturbations, 2) class probabilities.
    """
    all_probs = []
    bg_medians = pd.DataFrame(background_data, columns=RAW_INPUT_COLS).median()
    for i in range(raw_array.shape[0]):
        row = pd.DataFrame([raw_array[i]], columns=RAW_INPUT_COLS)
        # Impute any NaNs introduced by LIME perturbation
        for col in binary_cols:
            if col in row.columns:
                row[col] = row[col].fillna(0)
        row = row.fillna(bg_medians)
        prob = run_pipeline(row)
        all_probs.append([1 - prob, prob])
    return np.array(all_probs)


# ================= BUILD LIME BACKGROUND DATA =================
# Background must be in the same RAW space as RAW_INPUT_COLS.
# train_tabular is already processed, so we only use columns that
# exist in both train_tabular and RAW_INPUT_COLS for the background.
# For columns not in train_tabular (engineered ones), we fill with 0 —
# LIME uses background only for distribution statistics, not pipeline input.
def build_lime_background() -> np.ndarray:
    bg = pd.DataFrame(index=train_tabular.index)
    for c in RAW_INPUT_COLS:
        if c in train_tabular.columns:
            bg[c] = train_tabular[c]
        else:
            bg[c] = 0
    bg = bg.replace([np.inf, -np.inf], 0).fillna(0)
    return bg.values

background_data = build_lime_background()

# ================= INITIALISE LIME EXPLAINER =================
binary_cols = ["verified", "has_description", "has_prof_url", "has_location",
               "has_prof_img", "young_account_flag", "huge_followers_flag", "young_and_popular"]
categorical_feature_indices = [
    i for i, name in enumerate(RAW_INPUT_COLS) if name in binary_cols
]

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=background_data,
    feature_names=RAW_INPUT_COLS,
    class_names=["BOT", "HUMAN"],
    categorical_features=categorical_feature_indices,
    mode="classification",
    discretize_continuous=True,
    random_state=42,
)

# ================= PRE-FLIGHT CHECKS =================
dup_ids = api_df[api_df["user_id"].duplicated(keep=False)]["user_id"].unique()
if len(dup_ids) > 0:
    print(f"WARNING: {len(dup_ids)} duplicate user_id(s) found: {list(dup_ids)}")
    print("   Only the LAST explanation per duplicate will be kept.")

null_count = api_df["user_id"].isna().sum()
if null_count > 0:
    print(f"WARNING: {null_count} row(s) with null user_id — will be SKIPPED.")

# ================= MAIN LOOP =================
results = []
all_probs = []
lime_explanations = {}
X_user = None  # for diagnostics

for idx in range(len(api_df)):

    row = api_df.iloc[[idx]].copy()

    # -------- Skip null user_ids --------
    user_id = row["user_id"].values[0]
    if pd.isna(user_id) or str(user_id).strip() in ("", "nan"):
        print(f"Skipping row {idx} — null/empty user_id.")
        continue

    # -------- Predict --------
    prob = run_pipeline(row)
    all_probs.append(prob)

    label = "HUMAN" if prob >= 0.39 else "BOT"

    results.append({
        "user_id":     user_id,
        "probability": prob,
        "prediction":  label,
    })

    # -------- LIME Explanation --------
    # Extract raw input columns; fillna before passing to LIME:
    #   - Binary/categorical cols (e.g. verified) → 0 (safe default for missing flag)
    #   - Continuous cols → column median from training data background
    lime_row = row[RAW_INPUT_COLS].copy()
    for col in binary_cols:
        if col in lime_row.columns:
            lime_row[col] = lime_row[col].fillna(0)
    # Fill any remaining NaNs in continuous cols with the training background median
    bg_medians = pd.DataFrame(background_data, columns=RAW_INPUT_COLS).median()
    lime_row = lime_row.fillna(bg_medians)
    lime_instance = lime_row.values[0].astype(float)

    lime_exp = explainer.explain_instance(
        data_row=lime_instance,
        predict_fn=full_pipeline_for_lime,
        num_features=15,
        num_samples=300,
        labels=(1,),        # explain P(HUMAN); negate weight to read P(BOT)
    )

    lime_explanations[user_id] = lime_exp

    # Print summary
    print(f"\n{'='*60}")
    print(f"User: {user_id} | Prediction: {label} | P(HUMAN)={prob:.4f}")
    print("Top LIME features  [+ pushes toward HUMAN | - pushes toward BOT]")
    for feat, weight in lime_exp.as_list(label=1):
        direction = "HUMAN" if weight > 0 else "BOT  "
        print(f"  [{direction}]  {weight:+.4f}  |  {feat}")

# ================= SAVE PREDICTIONS =================
results_df = pd.DataFrame(results)
results_df.to_excel("batch_predictions.xlsx", index=False)

print("\nBatch prediction completed.")
print("Saved to: batch_predictions.xlsx")
print("Min prob :", min(all_probs))
print("Max prob :", max(all_probs))
print("Mean prob:", sum(all_probs) / len(all_probs))
print("Model classes:", model.classes_)

# ================= SAVE LIME EXPLANATIONS =================
lime_rows = []
for uid, exp in lime_explanations.items():
    prediction = next(r["prediction"] for r in results if r["user_id"] == uid)
    prob_val   = next(r["probability"] for r in results if r["user_id"] == uid)
    for rank, (feat, weight) in enumerate(exp.as_list(label=1), start=1):
        lime_rows.append({
            "user_id":       uid,
            "prediction":    prediction,
            "p_human":       round(prob_val, 4),
            "rank":          rank,
            "feature":       feat,
            "lime_weight":   round(weight, 6),
            "pushes_toward": "HUMAN" if weight > 0 else "BOT",
        })

lime_df = pd.DataFrame(lime_rows)
lime_df = lime_df.drop_duplicates(subset=["user_id", "feature"])
lime_df = lime_df.sort_values(["user_id", "rank"]).reset_index(drop=True)
lime_df.to_excel("lime_explanations.xlsx", index=False)
print("LIME explanations saved to: lime_explanations.xlsx")

# ================= OPTIONAL: HTML PER USER =================
# import os
# os.makedirs("lime_html", exist_ok=True)
# for uid, exp in lime_explanations.items():
#     exp.save_to_file(f"lime_html/{uid}.html")
# print("HTML files saved to lime_html/")
