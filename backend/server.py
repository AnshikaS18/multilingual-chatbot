# backend/server.py
import os
import json
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend")

app = Flask(__name__)
CORS(app)

# -----------------------
# CONFIG
# -----------------------
HF_API_KEY = os.environ.get("HF_API_KEY")  # set this before running
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMB_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

BASE_DIR = os.path.dirname(__file__)
QA_PATH = os.path.join(BASE_DIR, "qa.json")
EMB_CACHE_PATH = os.path.join(BASE_DIR, "qa_embs.npy")     # cached embeddings
QA_META_PATH = os.path.join(BASE_DIR, "qa_meta.json")      # meta (questions/ids)

# -----------------------
# Helpers
# -----------------------
def hf_get_embeddings(texts, timeout=30):
    """
    Call HF inference API to get sentence embeddings.
    Accepts a list of strings, returns numpy array (n, d).
    """
    if not isinstance(texts, list):
        texts = [texts]
    resp = requests.post(EMB_URL, headers=HEADERS, json={"inputs": texts}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # data could be:
    # - list of lists (embeddings per input), OR
    # - nested token embeddings (rare). We'll try to handle both.
    def to_vec(x):
        # if x is a list of scalars -> sentence vector
        if isinstance(x, list) and all(isinstance(v, (int, float)) for v in x):
            return np.array(x, dtype=float)
        # if x is list of lists (tokens), average across tokens
        if isinstance(x, list) and all(isinstance(v, list) for v in x):
            return np.mean(np.array(x, dtype=float), axis=0)
        raise ValueError("Unexpected embedding shape from HF API")
    if isinstance(data, list):
        vecs = [to_vec(item) for item in data]
        return np.vstack(vecs)
    # fallback single item
    return np.array([to_vec(data)], dtype=float)

def l2_normalize_rows(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

# -----------------------
# Load QA and embeddings (try cache)
# -----------------------
log.info("Loading QA from %s", QA_PATH)
with open(QA_PATH, "r", encoding="utf-8") as f:
    QA = json.load(f)

questions = [q["q"] for q in QA]
answers = [q["a"] for q in QA]
ids = [q.get("id", str(i)) for i,q in enumerate(QA)]

q_embs = None
if os.path.exists(EMB_CACHE_PATH) and os.path.exists(QA_META_PATH):
    try:
        log.info("Loading cached QA embeddings from %s", EMB_CACHE_PATH)
        q_embs = np.load(EMB_CACHE_PATH)
        with open(QA_META_PATH, "r", encoding="utf-8") as m:
            meta = json.load(m)
        if meta.get("questions") != questions:
            log.info("QA changed; will re-compute QA embeddings")
            q_embs = None
    except Exception as e:
        log.warning("Failed loading cache: %s", e)
        q_embs = None

if q_embs is None:
    log.info("Computing QA embeddings via HF API (this may take a moment)...")
    # call HF API in batch
    q_embs = hf_get_embeddings(questions)
    q_embs = l2_normalize_rows(q_embs)
    np.save(EMB_CACHE_PATH, q_embs)
    with open(QA_META_PATH, "w", encoding="utf-8") as m:
        json.dump({"questions": questions}, m)
    log.info("Saved QA embeddings to cache (%s)", EMB_CACHE_PATH)

# -----------------------
# Endpoints
# -----------------------
@app.route("/health")
def health():
    return jsonify({"status":"ok"})

@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json(force=True)
        text = (data.get("text") or "").strip()
        top_k = int(data.get("top_k", 1)) if data.get("top_k") else 1
        if not text:
            return jsonify({"error":"no text provided"}), 400

        # get embedding for query (single)
        q_vec = hf_get_embeddings([text])[0]
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)

        # cosine similarity via dot product (QA rows are normalized)
        sims = np.dot(q_embs, q_vec)
        best_idx = int(np.argmax(sims))
        # optionally return top_k
        top_idxs = list(np.argsort(-sims)[:top_k])
        results = []
        for i in top_idxs:
            results.append({
                "id": ids[i],
                "q": questions[i],
                "a": answers[i],
                "score": float(sims[i])
            })
        return jsonify({"results": results})
    except requests.exceptions.RequestException as e:
        log.exception("HF API request failed")
        return jsonify({"error":"embedding service error", "details": str(e)}), 500
    except Exception as ex:
        log.exception("Server error")
        return jsonify({"error":"server error", "details": str(ex)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

