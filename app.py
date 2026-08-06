import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

# Sayfa Yapılandırması
st.set_page_config(page_title="ALFA Terminal", layout="wide")

# --- ÖZGÜN CSS (Pointer İşareti ve Modern Navigasyon) ---
st.markdown(
    """
    <style>
    /* Radio butonlarını özelleştir ve el işareti yap */
    div[role="radiogroup"] > label {
        cursor: pointer !important;
        padding: 10px;
        border-radius: 5px;
    }
    div[role="radiogroup"] > label:hover {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- ARKA PLAN VERİLERİ (Caching) ---
@st.cache_data(ttl=3600)
def get_makro_data():
    try:
        df = yf.download("XU100.IS", period="3mo", progress=False)
        if df.empty: return 31.75, -1.98, 0.76
        df.index = pd.to_datetime(df.index)
        bugun = pd.Timestamp.today()
        day_t = df.index.asof(bugun)
        day_t2w = df.index.asof(bugun - pd.Timedelta(days=14))
        day_t2m = df.index.asof(bugun - pd.Timedelta(days=60))
        
        get_p = lambda d: float(df.loc[d]['Close'].iloc[0] if isinstance(df.loc[d]['Close'], pd.Series) else df.loc[d]['Close'])
        price_today, price_t2w, price_t2m = get_p(day_t), get_p(day_t2w), get_p(day_t2m)
        return 31.75, ((price_today - price_t2w) / price_t2w) * 100, ((price_today - price_t2m) / price_t2m) * 100
    except: return 31.75, -1.98, 0.76

tufe, oto_2h, oto_2a = get_makro_data()

# --- SESSION STATE ---
if "df" not in st.session_state: st.session_state.df = None
if "upload_time" not in st.session_state: st.session_state.upload_time = None

# --- HEADER ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("📈 BIST Algoritmik Hisse Seçim Modeli")
with col_h2:
    if st.session_state.upload_time:
        st.markdown(f"**Veri Yüklenme Zamanı:**<br>{st.session_state.upload_time}", unsafe_allow_html=True)
    st.markdown(f"**Makro:** TÜFE: %{tufe:.2f} | XU100 2A: %{oto_2a:.2f}")

st.markdown("---")

# --- SOL MENÜ ---
st.sidebar.markdown("### ⚡ ALFA Terminal")
menu = st.sidebar.radio("Navigasyon", ["📊 Radar & Taramalar", "🔍 Hisse Teşhis Paneli", "📁 Veri Yönetimi (Excel Yükle)"], label_visibility="collapsed")

# --- VERİ İŞLEME MANTIĞI (Her zaman çalışır) ---
def process_files(f1, f2):
    df1 = pd.read_excel(f1).iloc[:, :23]
    df1.columns = ["Kod", "ROE_0", "ROE_1", "ROE_4", "BrutEFK_0", "BrutEFK_1", "EFK_0", "EFK_1", "EFK_4", "FAVOK_0", "FAVOK_1", "NetSatisBuyume", "FAVOKBuyume", "BrutEFKBuyume", "EFKBuyume", "PDDD", "NetBorc_FAVOK", "HAOran", "Getiri_2h", "Getiri_1a", "Getiri_2a", "Getiri_6a", "Kapanis"]
    df2 = pd.read_excel(f2).iloc[:, :7]
    df2.columns = ["Kod", "ROE_2", "BrutEFK_2", "EFK_2", "FAVOK_2", "ROE_5", "EFK_5"]
    df = pd.merge(df1, df2, on="Kod", how="left").fillna(0)
    df["Bilanco_Durum"] = np.where((df["BrutEFK_0"] == df["BrutEFK_1"]), "GELMEDİ (Eski)", "GELDİ (Yeni)")
    return df

# --- SEKME İÇERİKLERİ ---
if menu == "📁 Veri Yönetimi (Excel Yükle)":
    st.header("📁 Veri Yönetimi")
    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Temel Analiz Dosyası", type=["xlsx"])
    f2 = c2.file_uploader("Eski Dönemler Dosyası", type=["xlsx"])
    
    if f1 and f2:
        if st.button("🚀 Verileri İşle"):
            st.session_state.df = process_files(f1, f2)
            st.session_state.upload_time = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            st.success("Veriler hafızaya alındı!")

elif menu == "📊 Radar & Taramalar":
    if st.session_state.df is not None:
        df = st.session_state.df
        # Filtreleme (Örnek)
        df["pdddLimit"] = np.where(df["ROE_0"] > 90, 8 + (df["ROE_0"] - 90) * 0.07, 8)
        filtreliler = (df["Getiri_2a"] > oto_2a) & (df["ROE_0"] > tufe)
        df_temel = df[filtreliler].copy()
        
        st.metric("Temel Filtreyi Geçen", f"{len(df_temel)} Hisse")
        st.dataframe(df_temel.head(10).style.format(precision=2), use_container_width=True)
    else:
        st.warning("Henüz veri yüklenmemiş. 'Veri Yönetimi' sekmesinden dosyaları yükleyin.")

elif menu == "🔍 Hisse Teşhis Paneli":
    if st.session_state.df is not None:
        kod = st.text_input("Hisse Kodu Girin (Örn: ATATP)").upper()
        if kod:
            res = st.session_state.df[st.session_state.df["Kod"] == kod]
            if not res.empty:
                st.write(res.T)
            else:
                st.error("Hisse bulunamadı.")
    else:
        st.warning("Veri bulunamadı.")
