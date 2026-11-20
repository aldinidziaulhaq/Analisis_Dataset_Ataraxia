import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Analisis Emosi Teks", layout="wide")

DATA_PATH = "data/dataset_clean_ataraxia.csv"
LABELS = ["senang", "sedih", "marah", "takut", "kaget", "jijik", "cape", "netral"]


def clean(text):
    return " ".join(str(text).lower().split())


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].map(clean)
    return df


@st.cache_resource
def train(_df):
    X_tr, X_te, y_tr, y_te = train_test_split(
        _df["text"], _df["label"], test_size=0.2, stratify=_df["label"], random_state=42)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(max_iter=2000, C=5))
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    report = pd.DataFrame(classification_report(y_te, pred, labels=LABELS, output_dict=True)).T
    cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=LABELS),
                      index=[f"asli: {l}" for l in LABELS],
                      columns=[f"prediksi: {l}" for l in LABELS])
    return model, accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro"), report, cm


df = load_data()
model, acc, f1, report, cm = train(df)

st.title("Analisis Emosi Teks Bahasa Indonesia")
st.caption("Dataset: dataset_clean_ataraxia.csv")
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
            st.metric("Emosi terprediksi", emotion, f"keyakinan {proba.max():.1%}",
                      delta_color="off")
            st.write("Probabilitas per kelas")
            st.bar_chart(pd.Series(proba, index=model.classes_).reindex(LABELS))

with tab_data:
    st.write(f"Jumlah baris: {len(df):,} | Jumlah kelas: {df['label'].nunique()}")
    c1, c2 = st.columns(2)
    c1.write("Distribusi label")
    c1.bar_chart(df["label"].value_counts().reindex(LABELS))
    c2.write("Statistik jumlah kata per teks")
    c2.dataframe(df["text"].str.split().str.len().describe().round(2).rename("jumlah kata"))
    st.write("Cuplikan data")
    label_filter = st.selectbox("Filter label", ["semua"] + LABELS)
    view = df if label_filter == "semua" else df[df["label"] == label_filter]
    st.dataframe(view.head(50), hide_index=True)

with tab_eval:
    m1, m2, m3 = st.columns(3)
    m1.metric("Akurasi (data uji)", f"{acc:.2%}")
    m2.metric("F1 makro (data uji)", f"{f1:.2%}")
    m3.metric("Jumlah data uji", f"{int(len(df) * 0.2):,}")
    st.write("Laporan klasifikasi per kelas")
    st.dataframe(report.round(3))
    st.write("Confusion matrix")
    st.dataframe(cm)
    st.caption("Pembagian data 80% latih dan 20% uji dengan stratifikasi label. Kalimat pada "
               "dataset memiliki pola yang seragam, sehingga akurasi tinggi belum menjamin "
               "performa yang sama pada teks dengan gaya berbeda.")