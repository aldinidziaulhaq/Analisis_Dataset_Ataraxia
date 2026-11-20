import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Analisis Emosi Teks", layout="wide")

LABELS = ["senang", "sedih", "marah", "takut", "kaget", "jijik", "cape", "netral"]
ATARAXIA = "data/dataset_clean_ataraxia.csv"
MARKUS = "data/dataset_markus_erelius_clean.csv"


def clean(text):
    return " ".join(str(text).lower().split())


@st.cache_data
def load_data():
    ata = pd.read_csv(ATARAXIA)
    ata["text"] = ata["text"].map(clean)

    # Berkas Marcus Aurelius dapat berformat non-standar (dibungkus tanda kutip
    # dan diakhiri titik koma), sehingga dibaca baris demi baris.
    rows = []
    with open(MARKUS, encoding="utf-8") as f:
        for line in f.read().splitlines()[1:]:
            s = line.strip().rstrip(";").strip()
            s = s[1:] if s.startswith('"') else s
            s = s[:-1] if s.endswith('"') else s
            label, _, text = s.replace('""', '"').partition(",")
            label, text = label.strip().strip('"'), clean(text.strip().strip('"'))
            if label in LABELS and text:
                rows.append((label, text))
    return ata, pd.DataFrame(rows, columns=["label", "text"])


@st.cache_resource
def train(ata):
    X_tr, X_te, y_tr, y_te = train_test_split(
        ata["text"], ata["label"], test_size=0.2, stratify=ata["label"], random_state=42)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(max_iter=2000, C=5))
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    report = pd.DataFrame(classification_report(y_te, pred, labels=LABELS, output_dict=True)).T
    return model, accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro"), report


ata, quotes = load_data()
model, acc, f1, report = train(ata)

st.title("Analisis Emosi Teks Bahasa Indonesia")
tab_pred, tab_data, tab_eval = st.tabs(["Prediksi", "Data", "Evaluasi"])

with tab_pred:
    text = st.text_area("Masukkan teks", height=100,
                        placeholder="Contoh: uangku habis dan aku marah.")
    if st.button("Prediksi", type="primary"):
        text = clean(text)
        if len(text.split()) < 2:
            st.warning("Masukkan minimal dua kata.")
        else:
            proba = model.predict_proba([text])[0]
            emotion = model.classes_[proba.argmax()]
            st.metric("Emosi terprediksi", emotion, f"keyakinan {proba.max():.1%}", delta_color="off")
            st.bar_chart(pd.Series(proba, index=model.classes_).reindex(LABELS))

            cand = quotes[quotes["label"] == emotion].reset_index(drop=True)
            tfidf = TfidfVectorizer(sublinear_tf=True).fit(cand["text"].tolist() + [text])
            sim = cosine_similarity(tfidf.transform([text]), tfidf.transform(cand["text"]))[0]
            st.subheader("Kutipan Marcus Aurelius dengan label yang sama")
            for i, idx in enumerate(np.argsort(-sim)[:3], 1):
                st.write(f"{i}. {cand.loc[idx, 'text']}")

with tab_data:
    c1, c2 = st.columns(2)
    c1.write("Distribusi label: Ataraxia")
    c1.bar_chart(ata["label"].value_counts().reindex(LABELS))
    c2.write("Distribusi label: Marcus Aurelius")
    c2.bar_chart(quotes["label"].value_counts().reindex(LABELS))
    st.write("Statistik jumlah kata per teks")
    st.dataframe(pd.DataFrame({
        "Ataraxia": ata["text"].str.split().str.len().describe(),
        "Marcus Aurelius": quotes["text"].str.split().str.len().describe()}).round(2))
    st.write("Cuplikan data")
    st.dataframe(ata.head(50), hide_index=True)

with tab_eval:
    m1, m2 = st.columns(2)
    m1.metric("Akurasi (data uji)", f"{acc:.2%}")
    m2.metric("F1 makro (data uji)", f"{f1:.2%}")
    st.dataframe(report.round(3))
    st.caption("Pembagian data 80% latih dan 20% uji. Pada kutipan Marcus Aurelius akurasi model "
               "hanya sekitar 25%, sehingga model hanya andal untuk kalimat pendek bergaya "
               "dataset Ataraxia.")