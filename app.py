import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path("data/dataset_clean_ataraxia.csv")

STOPWORDS = {
    "aku", "saya", "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan",
    "untuk", "tidak", "ada", "sekali", "banget", "tapi", "karena", "juga",
    "pada", "dalam", "kami", "kita", "dia", "mereka", "akan", "sudah", "lagi",
    "yg", "nya", "atau", "bisa", "hari", "lalu", "ku", "oleh", "saja", "mau",
    "lebih", "jadi", "waktu", "tadi", "sangat", "bikin", "kok", "masih",
}

st.set_page_config(
    page_title="Dataset Emosi Ataraxia",
    layout="wide",
)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["jumlah_karakter"] = df["text"].str.len()
    df["jumlah_kata"] = df["text"].str.split().str.len()
    return df


def tokenize(text: str):
    return re.findall(r"[a-zA-Z\u00c0-\u024f]+", text.lower())


@st.cache_data
def top_words(df: pd.DataFrame, label: str | None, n: int) -> pd.DataFrame:
    subset = df if label is None else df[df["label"] == label]
    counter = Counter()
    for t in subset["text"]:
        counter.update(w for w in tokenize(t) if w not in STOPWORDS and len(w) > 2)
    return pd.DataFrame(counter.most_common(n), columns=["kata", "frekuensi"])


# ---------------------------------------------------------------- load
if not DATA_PATH.exists():
    st.error(f"File tidak ditemukan: {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)
labels = sorted(df["label"].unique())

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih halaman",
    ["Ringkasan", "Distribusi Kelas", "Karakteristik Teks", "Analisis Kata", "Penjelajah Data"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Sumber: {DATA_PATH.as_posix()}")

# ---------------------------------------------------------------- header
st.title("Dataset Klasifikasi Emosi Ataraxia")
st.caption("Presentasi data teks bahasa Indonesia berlabel emosi yang telah melalui proses pembersihan.")

# ---------------------------------------------------------------- pages
if page == "Ringkasan":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah baris", f"{len(df):,}")
    c2.metric("Jumlah kelas", len(labels))
    c3.metric("Rata-rata kata", f"{df['jumlah_kata'].mean():.1f}")
    c4.metric("Rata-rata karakter", f"{df['jumlah_karakter'].mean():.1f}")

    st.subheader("Tentang dataset")
    st.write(
        "Dataset berisi kalimat pendek berbahasa Indonesia yang masing-masing diberi satu label emosi. "
        "Data mentah telah dibersihkan dengan langkah berikut: penyeragaman label, pembuangan label "
        "tidak valid, penghapusan URL dan mention, penghapusan karakter tak terlihat, perapian spasi, "
        "konversi ke huruf kecil, serta penghapusan teks duplikat."
    )

    st.subheader("Kelas emosi")
    st.write(", ".join(labels))

    st.subheader("Contoh data")
    st.dataframe(df[["label", "text"]].sample(10, random_state=42), use_container_width=True, hide_index=True)

    st.subheader("Kualitas data")
    q1, q2, q3 = st.columns(3)
    q1.metric("Nilai kosong", int(df[["label", "text"]].isna().sum().sum()))
    q2.metric("Teks duplikat", int(df["text"].duplicated().sum()))
    q3.metric("Teks mengandung URL atau mention", int(df["text"].str.contains(r"https?://|@\w+").sum()))

elif page == "Distribusi Kelas":
    st.subheader("Jumlah data per kelas")
    dist = df["label"].value_counts().rename_axis("label").reset_index(name="jumlah")
    left, right = st.columns([2, 1])
    with left:
        st.bar_chart(dist.set_index("label")["jumlah"])
    with right:
        dist["persentase"] = (dist["jumlah"] / len(df) * 100).round(2).astype(str) + "%"
        st.dataframe(dist, use_container_width=True, hide_index=True)

    if dist["jumlah"].nunique() == 1:
        st.info("Dataset seimbang sempurna: setiap kelas memiliki jumlah data yang sama.")
    else:
        rasio = dist["jumlah"].max() / dist["jumlah"].min()
        st.warning(f"Dataset tidak seimbang. Rasio kelas terbesar terhadap terkecil: {rasio:.2f}.")

elif page == "Karakteristik Teks":
    st.subheader("Panjang teks")
    metric = st.radio("Ukuran", ["jumlah_kata", "jumlah_karakter"], horizontal=True,
                      format_func=lambda x: "Jumlah kata" if x == "jumlah_kata" else "Jumlah karakter")

    st.markdown("**Statistik per kelas**")
    stats = (
        df.groupby("label")[metric]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
        .reset_index()
    )
    stats.columns = ["label", "jumlah", "rata-rata", "median", "minimum", "maksimum"]
    st.dataframe(stats, use_container_width=True, hide_index=True)

    st.markdown("**Rata-rata per kelas**")
    st.bar_chart(df.groupby("label")[metric].mean())

    st.markdown("**Sebaran keseluruhan**")
    bins = st.slider("Jumlah interval", 5, 50, 20)
    hist = pd.cut(df[metric], bins=bins).value_counts().sort_index()
    hist.index = [f"{int(i.left)}-{int(i.right)}" for i in hist.index]
    st.bar_chart(hist)

elif page == "Analisis Kata":
    st.subheader("Kata yang paling sering muncul")
    col_a, col_b = st.columns(2)
    pilihan = col_a.selectbox("Kelas", ["Semua kelas"] + labels)
    n = col_b.slider("Jumlah kata", 5, 30, 15)

    words = top_words(df, None if pilihan == "Semua kelas" else pilihan, n)
    left, right = st.columns([2, 1])
    with left:
        st.bar_chart(words.set_index("kata")["frekuensi"])
    with right:
        st.dataframe(words, use_container_width=True, hide_index=True)
    st.caption("Kata umum seperti kata ganti dan kata sambung telah disaring.")

    st.markdown("**Perbandingan 5 kata teratas antar kelas**")
    rows = {}
    for lb in labels:
        rows[lb] = ", ".join(top_words(df, lb, 5)["kata"])
    st.dataframe(
        pd.DataFrame({"label": rows.keys(), "kata teratas": rows.values()}),
        use_container_width=True, hide_index=True,
    )

elif page == "Penjelajah Data":
    st.subheader("Telusuri data")
    c1, c2 = st.columns([1, 2])
    pilih = c1.multiselect("Filter kelas", labels, default=labels)
    kata = c2.text_input("Cari kata dalam teks")

    view = df[df["label"].isin(pilih)]
    if kata.strip():
        view = view[view["text"].str.contains(re.escape(kata.strip().lower()), na=False)]

    st.write(f"Menampilkan {len(view):,} dari {len(df):,} baris")
    st.dataframe(view[["label", "text", "jumlah_kata"]], use_container_width=True, hide_index=True, height=450)

    st.download_button(
        "Unduh hasil filter (CSV)",
        view[["label", "text"]].to_csv(index=False).encode("utf-8"),
        file_name="hasil_filter_ataraxia.csv",
        mime="text/csv",
    )