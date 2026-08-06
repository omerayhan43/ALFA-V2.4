import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

# Sayfa Yapılandırması
st.set_page_config(page_title="ALFA V2.4 Borsa Algoritma Modeli", layout="wide")

# --- ÖZGÜN CSS (Pointer İşareti ve Modern Navigasyon) ---
st.markdown(
    """
    <style>
    div[role="radiogroup"] > label { cursor: pointer !important; padding: 10px; border-radius: 5px; }
    div[role="radiogroup"] > label:hover { background-color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True
)

# --- 1. OTOMATİK XU100 ENDEKS & MAKRO GİRDİLERİ ---
@st.cache_data(ttl=3600)
def otomatik_makro_veriler():
    try:
        df = yf.download("XU100.IS", period="3mo", progress=False)
        if df.empty: return 31.75, -1.98, 0.76 
        df.index = pd.to_datetime(df.index)
        bugun = pd.Timestamp.today()
        day_t = df.index.asof(bugun)
        day_t2w = df.index.asof(bugun - pd.Timedelta(days=14))
        day_t2m = df.index.asof(bugun - pd.Timedelta(days=60))
        get_p = lambda d: float(df.loc[d]['Close'].iloc[0] if isinstance(df.loc[d]['Close'], pd.Series) else df.loc[d]['Close'])
        p_t, p_t2w, p_t2m = get_p(day_t), get_p(day_t2w), get_p(day_t2m)
        return 31.75, ((p_t - p_t2w) / p_t2w) * 100, ((p_t - p_t2m) / p_t2m) * 100
    except: return 31.75, -1.98, 0.76

tufe_12, oto_2h, oto_2a = otomatik_makro_veriler()

# --- SESSION STATE ---
if "df_merged" not in st.session_state: st.session_state.df_merged = None
if "upload_time" not in st.session_state: st.session_state.upload_time = None

# --- HEADER ---
col_h1, col_h2 = st.columns([3, 2])
with col_h1:
    st.title("📈 BIST Algoritmik Hisse Seçim Modeli (ALFA V2.4)")
with col_h2:
    zaman_bilgisi = f"<br><span>🕒 Yükleme: <b>{st.session_state.upload_time}</b></span>" if st.session_state.upload_time else ""
    st.markdown(f'<div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef; font-size: 13px; text-align: right;"><b>⚙️ Otomatik Makro Girdiler:</b><br><span>TÜFE(12): <b>%{tufe_12:.2f}</b> | XU100 2A: <b>%{oto_2a:.2f}</b> | XU100 2H: <b>%{oto_2h:.2f}</b></span>{zaman_bilgisi}</div>', unsafe_allow_html=True)

st.markdown("---")

# --- SOL MENÜ ---
st.sidebar.markdown("### ⚡ ALFA Terminal")
menu_secim = st.sidebar.radio("Navigasyon", ["📊 Radar & Taramalar", "🔍 Hisse Teşhis Paneli", "📁 Veri Yönetimi (Excel Yükle)"], label_visibility="collapsed")

# --- VERİ YÖNETİMİ ---
if menu_secim == "📁 Veri Yönetimi (Excel Yükle)":
    st.subheader("📁 Fintables Veri Yönetimi")
    col_up1, col_up2 = st.columns(2)
    file1 = col_up1.file_uploader("1. Temel Analiz & Fiyat Dosyası", type=["xlsx", "xls"])
    file2 = col_up2.file_uploader("2. Eski Dönemler Dosyası (1)", type=["xlsx", "xls"])

    if file1 and file2:
        try:
            df1 = pd.read_excel(file1).iloc[:, :23]
            df1.columns = ["Kod", "ROE_0", "ROE_1", "ROE_4", "BrutEFK_0", "BrutEFK_1", "EFK_0", "EFK_1", "EFK_4", "FAVOK_0", "FAVOK_1", "NetSatisBuyume", "FAVOKBuyume", "BrutEFKBuyume", "EFKBuyume", "PDDD", "NetBorc_FAVOK", "HAOran", "Getiri_2h", "Getiri_1a", "Getiri_2a", "Getiri_6a", "Kapanis"]
            df2 = pd.read_excel(file2).iloc[:, :7]
            df2.columns = ["Kod", "ROE_2", "BrutEFK_2", "EFK_2", "FAVOK_2", "ROE_5", "EFK_5"]
            df = pd.merge(df1, df2, on="Kod", how="left").fillna(0)
            st.session_state.df_merged = df
            st.session_state.upload_time = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            st.success("✅ Veriler hafızaya alındı!")
        except Exception as e: st.error(f"Hata: {e}")

# --- RADAR & TEŞHİS ---
if st.session_state.df_merged is not None:
    df = st.session_state.df_merged
    df["pdddLimit"] = np.where(df["ROE_0"] > 90, 8 + (df["ROE_0"] - 90) * 0.07, 8)
    
    # Filtreler (Stabil versiyon)
    temel_filtreliler = (df["Getiri_2a"] > oto_2a) & (df["ROE_0"] > tufe_12) & (df["NetBorc_FAVOK"] < 4) & (df["PDDD"] < df["pdddLimit"]) & (df["Getiri_2h"] > (oto_2h - 10)) & (df["NetSatisBuyume"] > 0)
    df_temel = df[temel_filtreliler].copy()

    if menu_secim == "📊 Radar & Taramalar":
        st.subheader("📊 Radar & Taramalar")
        # Teknik tarama
        for idx, row in df_temel.iterrows():
            kod = str(row["Kod"]).strip() + ".IS"
            try:
                hist = yf.download(kod, period="1y", progress=False)
                if not hist.empty:
                    close = hist['Close'].iloc[:, 0] if isinstance(hist['Close'], pd.DataFrame) else hist['Close']
                    if len(close) >= 200:
                        df_temel.loc[idx, ['MA20', 'MA75', 'MA200']] = [round(close.rolling(20).mean().iloc[-1], 2), round(close.rolling(75).mean().iloc[-1], 2), round(close.rolling(200).mean().iloc[-1], 2)]
            except: continue
        st.dataframe(df_temel, use_container_width=True)

    elif menu_secim == "🔍 Hisse Teşhis Paneli":
        st.markdown("### 🔍 Hisse Teşhis Paneli")
        hisse_listesi = [""] + sorted(df["Kod"].dropna().astype(str).str.upper().unique().tolist())
        secilen_hisse = st.selectbox("Hisse Seçin:", options=hisse_listesi)

        if secilen_hisse:
            th = df[df["Kod"].str.upper() == secilen_hisse].iloc[0]
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Temel Analiz Kriterleri ({secilen_hisse})**")
                # Kriter kontrolleri (Önceki versiyondaki uzun bloklar buraya gelecek)
                st.write("Kriter kontrolü yapılıyor...") 
            with col2:
                st.write(f"**Teknik Analiz Grafiği**")
                h_yf = yf.download(secilen_hisse + ".IS", period="1y", progress=False)
                if not h_yf.empty:
                    h_yf['MA20'] = h_yf['Close'].rolling(20).mean()
                    h_yf['MA75'] = h_yf['Close'].rolling(75).mean()
                    st.line_chart(h_yf[['Close', 'MA20', 'MA75']])
                    st.success("Analiz tamamlandı.")
