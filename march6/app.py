"""
Bot Detection API — Flask backend
Install: pip install flask flask-cors
Run:     python app.py
Then open index.html in your browser.
"""

import os, time, requests
import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.neighbors import NearestNeighbors
import lime, lime.lime_tabular

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ── CONFIG ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY   = "OPENROUTER_API_KEY"
LLM_MODEL            = "stepfun/step-3.5-flash:free"
LLM_MAX_TOKENS       = 800
LLM_TEMPERATURE      = 0.3
LLM_RETRY_BASE_DELAY = 10
LLM_RETRIES          = 5

# ── CLASSIFICATION THRESHOLD ─────────────────────────────────────────────────
# Optimal threshold derived from ROC/precision-recall analysis.
#   P(Human) >= 0.4237 → HUMAN
#   P(Human) <  0.4237 → BOT
THRESHOLD = 0.42

def classify(prob: float) -> str:
    return "HUMAN" if prob >= THRESHOLD else "BOT"

# ── LOAD ARTIFACTS ────────────────────────────────────────────────────────────
print("Loading model artifacts...")
model          = joblib.load("Models/random_forest_bot_classifier.pkl")
feature_list   = joblib.load("Models/random_forest_feature_list.pkl")
ml_cols        = joblib.load("Models/random_forest_ml_feature_cols.pkl")
robust_scaler  = joblib.load("Models/robust_scaler.pkl")
age_scaler     = joblib.load("Models/age_scaler.pkl")
emb_normalizer = joblib.load("Models/embedding_normalizer.pkl")

ref_emb       = pd.read_csv("Dataset/training_embeddings_reference.csv")
train_tabular = pd.read_csv("Dataset/training_tabular_reference.csv")
api_df        = pd.read_excel("api_file.xlsx", dtype={"user_id": str})

rename_map = {f"emb_{i}": str(i) for i in range(64)}
ref_emb    = ref_emb.rename(columns=rename_map)

emb_cols        = [c for c in ref_emb.columns if c.isdigit()]
DROP_SIM_COLS   = ["user_id", "user_name", "Label", "dataset"]
similarity_cols = [c for c in train_tabular.columns if c not in DROP_SIM_COLS]

nn = NearestNeighbors(n_neighbors=10, metric="cosine")
nn.fit(train_tabular[similarity_cols])
followers_95_quantile = train_tabular["followers_count"].quantile(0.95)

RAW_INPUT_COLS = [c for c in api_df.columns if c not in
                  ["user_id", "user_name", "Label", "label", "dataset"]]
binary_cols    = ["verified", "has_description", "has_prof_url", "has_location",
                  "has_prof_img", "young_account_flag", "huge_followers_flag", "young_and_popular"]

# ── LIME SETUP ────────────────────────────────────────────────────────────────
def _build_background():
    bg = pd.DataFrame(index=train_tabular.index)
    for c in RAW_INPUT_COLS:
        bg[c] = train_tabular[c] if c in train_tabular.columns else 0
    return bg.replace([np.inf, -np.inf], 0).fillna(0).values

background_data = _build_background()
bg_medians      = pd.DataFrame(background_data, columns=RAW_INPUT_COLS).median()
cat_indices     = [i for i, n in enumerate(RAW_INPUT_COLS) if n in binary_cols]

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=background_data, feature_names=RAW_INPUT_COLS,
    class_names=["BOT", "HUMAN"], categorical_features=cat_indices,
    mode="classification", discretize_continuous=True, random_state=42,
)
print("Ready — http://localhost:5000")

# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
def run_pipeline(row: pd.DataFrame) -> float:
    row = row.copy()
    row["account_age_days"] = row["account_age_days"].clip(lower=1.0)
    row["tweets_count"]     = row["tweets_count"].clip(lower=0.0)
    row["followers_count"]  = row["followers_count"].clip(lower=0.0)
    row["tweets_per_day"]         = row["tweets_count"]    / row["account_age_days"]
    row["followers_per_day"]      = row["followers_count"] / row["account_age_days"]
    row["log_tweets_per_day"]     = np.log1p(row["tweets_per_day"])
    row["log_followers_per_day"]  = np.log1p(row["followers_per_day"])
    row["followers_spike"]        = row["followers_count"] / np.sqrt(row["account_age_days"])
    row["tweet_spike"]            = row["tweets_count"]    / np.sqrt(row["account_age_days"])
    row["extreme_activity_score"] = row["log_followers_per_day"] + row["log_tweets_per_day"]
    row["young_account_flag"]     = (row["account_age_days"] < 90).astype(int)
    row["huge_followers_flag"]    = (row["followers_count"] > followers_95_quantile).astype(int)
    row["young_and_popular"]      = row["young_account_flag"] * row["huge_followers_flag"]
    for c in similarity_cols:
        if c not in row.columns: row[c] = 0
    row[similarity_cols] = row[similarity_cols].replace([np.inf, -np.inf], 0).fillna(0)
    _, idx_nn    = nn.kneighbors(row[similarity_cols])
    neighbor_ids = train_tabular.iloc[idx_nn[0]]["user_id"]
    emb    = ref_emb[ref_emb["user_id"].isin(neighbor_ids)][emb_cols].mean().values
    emb_df = pd.DataFrame([emb], columns=emb_cols, index=row.index)
    row    = pd.concat([row, emb_df], axis=1)
    row[emb_cols] = emb_normalizer.transform(row[emb_cols])
    robust_cols = robust_scaler.feature_names_in_
    for c in robust_cols:
        if c not in row.columns: row[c] = 0
    row[robust_cols] = robust_scaler.transform(row[robust_cols])
    if "account_age_days" in row.columns:
        row[["account_age_days"]] = age_scaler.transform(row[["account_age_days"]])
    for c in feature_list:
        if c not in row.columns: row[c] = 0
    return model.predict_proba(row[feature_list])[:, 1][0]

def _lime_predict(raw_array):
    out = []
    for i in range(raw_array.shape[0]):
        r = pd.DataFrame([raw_array[i]], columns=RAW_INPUT_COLS)
        for col in binary_cols:
            if col in r.columns: r[col] = r[col].fillna(0)
        r    = r.fillna(bg_medians)
        prob = run_pipeline(r)
        out.append([1 - prob, prob])
    return np.array(out)

def get_lime_features(row):
    lr = row[RAW_INPUT_COLS].copy()
    for col in binary_cols:
        if col in lr.columns: lr[col] = lr[col].fillna(0)
    lr  = lr.fillna(bg_medians)
    exp = explainer.explain_instance(
        data_row=lr.values[0].astype(float), predict_fn=_lime_predict,
        num_features=15, num_samples=300, labels=(1,)
    )
    return [{"rank": i+1, "feature": f, "weight": round(w, 6),
             "direction": "HUMAN" if w > 0 else "BOT"}
            for i, (f, w) in enumerate(exp.as_list(label=1))]

FEATURE_DESC = {
    "followers_count":"number of followers","friends_count":"number of accounts followed",
    "tweets_count":"total tweets posted","listed_count":"times added to Twitter lists",
    "hashtag_count":"total hashtags used","mentions_count":"total mentions of other users",
    "retweet_count":"total retweets received","reply_count":"total replies received",
    "url_count":"total URLs shared","ff_ratio":"follower-to-following ratio",
    "avg_hashtag":"avg hashtags per tweet","avg_mentions":"avg mentions per tweet",
    "avg_retweet":"avg retweets per tweet","avg_reply":"avg replies per tweet",
    "avg_url":"avg URLs per tweet","avg_user_engagement":"avg user engagement score",
    "has_description":"account has a bio","has_prof_url":"account has a profile URL",
    "has_location":"account has a location set","has_prof_img":"account has a profile image",
    "profile_completeness":"profile completeness","avg_polarity":"avg tweet sentiment polarity",
    "avg_subjectivity":"avg tweet subjectivity","unique_word_count":"total unique words used",
    "unique_word_use":"ratio of unique words","punctuation_count":"total punctuation used",
    "avg_sentence_length":"avg sentence length","punctuation_density":"punctuation density",
    "account_age_days":"account age in days","verified":"account is verified",
    "tweets_per_day":"avg tweets per day","followers_per_day":"avg follower gain per day",
    "followers_spike":"follower spike vs account age","tweet_spike":"tweet spike vs account age",
    "extreme_activity_score":"combined activity spike score",
    "young_account_flag":"account under 90 days old",
    "huge_followers_flag":"follower count in top 5%","young_and_popular":"new and high-follower account",
}

def _h(raw):
    for key, desc in FEATURE_DESC.items():
        if raw.startswith(key) or f" {key} " in raw or raw.endswith(key):
            cond = raw.replace(key,"").strip().lstrip("=").strip()
            return f"{desc} ({cond})" if cond else desc
    return raw

def get_llm_explanation(user_id, prediction, p_human, lime_features):
    bots   = [(f["feature"], f["weight"]) for f in lime_features if f["weight"] < 0]
    humans = [(f["feature"], f["weight"]) for f in lime_features if f["weight"] > 0]
    bot_lines   = "\n".join(f"  - {_h(f)} (influence: {abs(w):.4f})" for f,w in sorted(bots,   key=lambda x:x[1])[:5])
    human_lines = "\n".join(f"  - {_h(f)} (influence: {abs(w):.4f})" for f,w in sorted(humans, key=lambda x:-x[1])[:5])

    if prediction == "HUMAN":
        verdict    = "likely human"
        confidence = "narrow" if p_human <= 0.46 else "moderate"
    else:  # BOT
        verdict    = "likely automated (bot)"
        confidence = "narrow" if p_human >= 0.38 else "moderate"

    prompt = f"""You are an expert analyst writing a clear, polished report on a Twitter account's authenticity.

A machine learning model analysed account {user_id} and concluded it is {verdict}, with a {confidence} margin of confidence.

Evidence suggesting automated (bot) behaviour:
{bot_lines or "  - None identified"}

Evidence suggesting genuine human behaviour:
{human_lines or "  - None identified"}

Write a polished 4-5 sentence paragraph a non-technical reader would understand. Do NOT mention feature names, numbers, or model scores. Instead:
- Open with a clear one-sentence verdict on whether this account appears human, bot, or uncertain.
- Explain in plain English what behavioural patterns led to this conclusion.
- Acknowledge contradictory signals naturally if present.
- Close with a sentence on confidence level and any caveats.
Professional journalistic tone. Complete every sentence."""

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}",
               "Content-Type": "application/json", "HTTP-Referer": "https://bot-detection"}
    payload = {"model": LLM_MODEL, "max_tokens": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE,
               "messages": [{"role": "user", "content": prompt}]}

    for attempt in range(1, LLM_RETRIES + 1):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                choices = r.json().get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content")
                    text = content.strip() if isinstance(content, str) else ""
                    if text: return text
                if attempt < LLM_RETRIES: time.sleep(LLM_RETRY_BASE_DELAY); continue
            elif r.status_code == 429:
                time.sleep(LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            else:
                return f"LLM error {r.status_code}"
        except Exception as e:
            return f"Request failed: {e}"
    return "LLM explanation unavailable."

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/api/users")
def list_users():
    users = (api_df[["user_id","user_name"]]
             .dropna(subset=["user_id"])
             .assign(user_id=lambda d: d["user_id"].astype(str))
             .drop_duplicates("user_id"))
    return jsonify(users.to_dict(orient="records"))

@app.route("/api/analyse/<user_id>")
def analyse(user_id):
    row = api_df[api_df["user_id"] == str(user_id)]
    if row.empty:
        return jsonify({"error": f"User {user_id} not found"}), 404
    row       = row.iloc[[0]].copy()
    user_name = str(row["user_name"].values[0]) if "user_name" in row.columns else user_id
    prob          = run_pipeline(row)
    label         = classify(prob)
    lime_features = get_lime_features(row)
    llm_text      = get_llm_explanation(user_id, label, prob, lime_features)
    # Raw input features for frontend display (exclude ID/label cols)
    raw_display_cols = [c for c in RAW_INPUT_COLS if c not in
                        ["user_id","user_name","Label","label","dataset"]]
    raw_features = {}
    for c in raw_display_cols:
        v = row[c].values[0] if c in row.columns else None
        # convert numpy types to native Python for JSON serialisation
        if hasattr(v, 'item'):
            v = v.item()
        # replace NaN / Inf with None so Flask produces valid JSON
        if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')):
            v = None
        raw_features[c] = v

    return jsonify({
        "user_id": user_id, "user_name": user_name,
        "prediction": label, "p_human": round(prob, 4), "p_bot": round(1-prob, 4),
        "lime_features": lime_features, "llm_explanation": llm_text,
        "raw_features": raw_features,
    })

from flask import send_from_directory

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(debug=False, port=5000)
