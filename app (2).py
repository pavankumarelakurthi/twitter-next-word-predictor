"""
Twitter Next Word Predictor — Streamlit App
==========================================
Run:  streamlit run app.py
Requires saved_model/ folder with:
  - next_word_model.keras
  - tokenizer.jsongit add .
  - config.json
"""

import streamlit as st
import numpy as np
import json
import re
import os
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐦 Twitter Next Word Predictor",
    page_icon="🐦",
    layout="centered",
)

# ── Load model (cached — loads only once) ─────────────────────────────────────
@st.cache_resource
def load_artifacts(model_dir="saved_modell"):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    model_path     = os.path.join(model_dir, r"C:\pknextword\save_modell\next_word_modell.keras")
    tokenizer_path = os.path.join(model_dir, r"C:\pknextword\save_modell\tokenizer.json")
    config_path    = os.path.join(model_dir, r"C:\pknextword\save_modell\config.json")

    if not os.path.exists(model_path):
        return None, None, None

    model = load_model(model_path)

    with open(tokenizer_path, encoding="utf-8") as f:
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        tokenizer = tokenizer_from_json(f.read())

    with open(config_path) as f:
        cfg = json.load(f)

    return model, tokenizer, cfg


# ── Prediction helpers ────────────────────────────────────────────────────────
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_next_word(model, tokenizer, text, max_seq_len, top_k=5):
    if not isinstance(text, str) or not text.strip():
        return {"top_word": None, "top_k": []}

    text_clean = re.sub(r"[^a-z\s]", "", text.lower()).strip()
    if not text_clean:
        return {"top_word": None, "top_k": []}

    token_list = tokenizer.texts_to_sequences([text_clean])[0]
    if not token_list:
        return {"top_word": None, "top_k": []}

    token_list = token_list[-(max_seq_len - 1):]
    padded_seq = pad_sequences([token_list], maxlen=max_seq_len - 1, padding="pre")
    probs      = model.predict(padded_seq, verbose=0)[0]

    top_indices = np.argsort(probs)[::-1][:top_k]
    candidates  = [(tokenizer.index_word.get(int(i), "<unk>"), float(probs[i]))
                   for i in top_indices]
    return {"top_word": candidates[0][0], "top_k": candidates}


def predict_sentence(model, tokenizer, seed, max_seq_len, n_words=5):
    generated = seed
    for _ in range(n_words):
        res    = predict_next_word(model, tokenizer, generated, max_seq_len)
        next_w = res["top_word"]
        if next_w is None or next_w.startswith("<"):
            break
        generated += " " + next_w
    return generated


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐦 Twitter Next Word Predictor")
st.markdown("Trained on **~30,000 tweets** using a **Bidirectional LSTM** model.")
st.markdown("---")

# Load model
model, tokenizer, cfg = load_artifacts()

if model is None:
    st.error(
        "❌ **Model not found!**\n\n"
        "Make sure the `saved_model/` folder is in the same directory as `app.py` and contains:\n"
        "- `next_word_model.keras`\n"
        "- `tokenizer.json`\n"
        "- `config.json`\n\n"
        "Run the Jupyter notebook first to train and save the model."
    )
    st.stop()

MAX_SEQ_LEN = cfg["max_seq_len"]
st.success(f"✅ Model loaded  |  Vocab: {cfg['vocab_size']:,}  |  Max sequence: {MAX_SEQ_LEN}")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Next Word", "📝 Auto-Complete", "📊 Top-K Chart"])

# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Predict the Next Word")
    seed_input = st.text_input(
        "Enter a phrase:",
        value="government should take",
        placeholder="Type anything..."
    )
    top_k = st.slider("Number of predictions to show", 1, 10, 5)

    if st.button("⚡ Predict", use_container_width=True, key="btn_predict"):
        if seed_input.strip():
            with st.spinner("Predicting..."):
                res = predict_next_word(model, tokenizer, seed_input, MAX_SEQ_LEN, top_k=top_k)

            if not res["top_k"]:
                st.warning("No predictions — try different words.")
            else:
                st.markdown(f"### 🏆 Top prediction: `{res['top_word']}`")
                st.markdown("**All candidates:**")
                for rank, (word, prob) in enumerate(res["top_k"], 1):
                    bar_len = int(prob * 300)
                    st.markdown(
                        f"`#{rank}` &nbsp; **{word}** &nbsp;"
                        f"<span style='color:#0ea5e9'>{prob:.2%}</span> "
                        f"<span style='background:#0ea5e9;display:inline-block;"
                        f"width:{bar_len}px;height:10px;border-radius:4px'></span>",
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("Please enter some text first.")

# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Auto-Complete a Sentence")
    seed_auto = st.text_input(
        "Enter a seed phrase:",
        value="india needs more",
        placeholder="Start of a sentence...",
        key="auto_seed"
    )
    n_words = st.slider("Words to generate", 1, 15, 5, key="n_words_slider")

    if st.button("✍️ Complete Sentence", use_container_width=True, key="btn_complete"):
        if seed_auto.strip():
            with st.spinner("Generating..."):
                completed = predict_sentence(model, tokenizer, seed_auto, MAX_SEQ_LEN, n_words=n_words)

            original_words = len(seed_auto.split())
            all_words      = completed.split()

            # Colour the generated part differently
            original_part  = " ".join(all_words[:original_words])
            generated_part = " ".join(all_words[original_words:])

            st.markdown("**Result:**")
            st.markdown(
                f"<p style='font-size:1.3rem'>"
                f"{original_part} "
                f"<span style='color:#34d399;font-weight:700'>{generated_part}</span>"
                f"</p>",
                unsafe_allow_html=True,
            )
            st.info(f"Seed: \"{seed_auto}\"  →  Generated: \"{generated_part}\"")
        else:
            st.warning("Please enter a seed phrase first.")

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Top-K Probability Chart")
    seed_chart = st.text_input(
        "Enter seed for chart:",
        value="election results will",
        key="chart_seed"
    )
    top_k_chart = st.slider("Top-K", 3, 15, 8, key="topk_chart")

    if st.button("📊 Generate Chart", use_container_width=True, key="btn_chart"):
        if seed_chart.strip():
            with st.spinner("Running model..."):
                res = predict_next_word(model, tokenizer, seed_chart, MAX_SEQ_LEN, top_k=top_k_chart)

            if not res["top_k"]:
                st.warning("No predictions returned.")
            else:
                words_list = [r[0] for r in res["top_k"]]
                probs_list = [r[1] * 100 for r in res["top_k"]]

                fig, ax = plt.subplots(figsize=(8, max(3, top_k_chart * 0.5)))
                bars = ax.barh(words_list[::-1], probs_list[::-1], color="#0ea5e9", edgecolor="white")
                ax.set_xlabel("Probability (%)")
                ax.set_title(f'Top-{top_k_chart} Next Word Predictions\nSeed: "{seed_chart}"')
                for bar, prob in zip(bars, probs_list[::-1]):
                    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                            f"{prob:.2f}%", va="center", fontsize=9)
                ax.set_facecolor("#0f172a")
                fig.patch.set_facecolor("#1e293b")
                ax.title.set_color("white")
                ax.xaxis.label.set_color("white")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#334155")
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Please enter a seed phrase first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#64748b;font-size:.85rem'>"
    "Bidirectional LSTM | Twitter Sentiment Dataset | Built with TensorFlow + Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
