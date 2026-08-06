import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

# Sayfa Yapılandırması
st.set_page_config(page_title="ALFA V2.4 Borsa Algoritma Modeli", layout="wide")

st.title("📈 BIST Algoritmik Hisse Seçim Modeli (ALFA V2.4)")
st.markdown("Temel + Teknik Filtreleme -> Ağırlıklı Skorlama -> İlk 5 Hisse Portföy Adayı")

# --- 1. OTOMATİK XU100 ENDEKS GETİRİLERİ (TARİH BAZLI HASSAS HESAPLAMA) ---
@st.cache_data(ttl=3600)
def otomatik_xu100_getirileri():
    try:
        # Son 3 aylık veriyi çek
        df = yf.download("XU100.IS", period="3mo", progress=False)
        if df.empty: return -1.04, 0.40 # Yedek değerler
        
        # Index'i datetime yap
        df.index = pd.to_datetime(df.index)
        
        bugun = pd.Timestamp.today()
        t2w = bugun - pd.Timedelta(days=14)
        t2m = bugun - pd.Timedelta(days=60)
        
        # 'asof' metodu: O tarihte veya ondan önceki son işlem gününü getirir (Hafta sonu riski biter)
        day_t = df.index.asof(bugun)
        day_t2w = df.index.asof(t2w)
        day_t2m = df.index.asof(t2m)
        
        # Fiyatları al (S Series veya DataFrame kontrolü)
        get_price = lambda d: float(df.loc[d]['Close'].iloc[0] if isinstance(df.loc[d]['Close'], pd.Series) else df.loc[d]['Close'])
        
        price_today = get_price(day_t)
        price_t2w = get_price(day_t2w)
        price_t2m = get_price(day_t2m)
        
        xu100_2h = ((price_today - price_t2w) / price_t2w) * 100
        xu100_2a = ((price_today - price_t2m) / price_t2m) * 100
        
        return xu100_2h, xu100_2a
    except:
        return -1.04, 0.40

oto_2h, oto_2a = otomatik_xu100_getirileri()

# --- 2. YAN MENÜ ---
st.sidebar.header("⚙️ Makro Girdiler (Otomatik & Güncel)")
tufe_12 = st.sidebar.number_input("TÜFE(12) Yıllık %", value=31.75, format="%.2f")
xu100_2a = st.sidebar.number_input("XU100 2-Aylık Getiri %", value=oto_2a, format="%.2f")
xu100_2h = st.sidebar.number_input("XU100 2-Haftalık Getiri %", value=oto_2h, format="%.2f")

st.sidebar.header("📁 Fintables Veri Yükleme")
file1 = st.sidebar.file_uploader("1. Temel Analiz & Fiyat", type=["xlsx", "xls"])
file2 = st.sidebar.file_uploader("2. Eski Dönemler", type=["xlsx", "xls"])

if file1 and file2:
    # Veri İşleme
    df1 = pd.read_excel(file1).iloc[:, :23]
    df1.columns = ["Kod", "ROE_0", "ROE_1", "ROE_4", "BrutEFK_0", "BrutEFK_1", "EFK_0", "EFK_1", "EFK_4", "FAVOK_0", "FAVOK_1", "NetSatisBuyume", "FAVOKBuyume", "BrutEFKBuyume", "EFKBuyume", "PDDD", "NetBorc_FAVOK", "HAOran", "Getiri_2h", "Getiri_1a", "Getiri_2a", "Getiri_6a", "Kapanis"]
    df2 = pd.read_excel(file2).iloc[:, :7]
    df2.columns = ["Kod", "ROE_2", "BrutEFK_2", "EFK_2", "FAVOK_2", "ROE_5", "EFK_5"]
    df = pd.merge(df1, df2, on="Kod", how="left").fillna(0)
    
    # Bilanço Durum
    bilanco_gelmedi = ((df["BrutEFK_0"] == df["BrutEFK_1"]) & (df["EFK_0"] == df["EFK_1"]) & (df["FAVOK_0"] == df["FAVOK_1"]))
    df["Bilanco_Durum"] = np.where(bilanco_gelmedi, "GELMEDİ", "GELDİ")
    
    # Filtreleme
    df["pdddLimit"] = np.where(df["ROE_0"] > 90, 8 + (df["ROE_0"] - 90) * 0.07, 8)
    temel_mask = (df["Getiri_2a"] > xu100_2a) & (df["ROE_0"] > tufe_12) & (df["NetBorc_FAVOK"] < 4) & (df["PDDD"] < df["pdddLimit"]) & (df["HAOran"] < 60)
    df_temel = df[temel_mask].copy()

    # --- TABS ---
    tab1, tab2 = st.tabs(["📊 Tarama Sonuçları", "🔍 Hisse Teşhis & Skor"])

    with tab1:
        st.success(f"Filtreleri Geçen: {len(df_temel)} Hisse")
        st.dataframe(df_temel[["Kod", "ROE_0", "PDDD", "Getiri_2a"]])

    with tab2:
        secilen_hisse = st.text_input("Hisse Kodu Gir (Örn: CRDFA)", "").upper().strip()
        if secilen_hisse:
            th = df[df["Kod"] == secilen_hisse].iloc[0] if secilen_hisse in df["Kod"].values else None
            if th is not None:
                # Kriter Kontrolleri
                c_a = th['Getiri_2a'] > xu100_2a
                c_b = th['ROE_0'] > tufe_12
                c_c = th['NetBorc_FAVOK'] < 4
                c_d = th['PDDD'] < th['pdddLimit']
                c_k = th['HAOran'] < 60
                
                takilanlar = []
                if not c_b: takilanlar.append("ROE < TÜFE")
                if not c_a: takilanlar.append("2A Getiri < XU100")
                if not c_c: takilanlar.append("Net Borç/FAVÖK > 4")
                if not c_d: takilanlar.append("PD/DD Yüksek")
                if not c_k: takilanlar.append("Halka Açıklık > 60")
                
                if takilanlar:
                    st.warning(f"⚠️ Uyarı: {secilen_hisse} hissesi şu kriterleri sağlamıyor: {', '.join(takilanlar)}")
                else:
                    st.success("✅ Hisse tüm temel filtrelerden geçti.")

                # Skorlama (Geçse de geçmese de hesapla)
                r_skor = (100 if th['ROE_0'] > 50 else th['ROE_0'] * 2) * 0.30
                p_skor = (25 if th['PDDD'] < 1.5 else (15 if th['PDDD'] < 3 else (5 if th['PDDD'] < 5 else -10))) * 0.12
                # Basitleştirilmiş skor (Detay eklenebilir)
                nihai_skor = r_skor + p_skor
                
                st.markdown(f"### 🎯 **Skor: {nihai_skor:.2f}**")
                st.write(f"- ROE Katkısı: {r_skor:.2f}")
                st.write(f"- PD/DD Katkısı: {p_skor:.2f}")
            else:
                st.warning("Hisse bulunamadı.")
else:
    st.info("👈 Dosyaları yükleyin.")
