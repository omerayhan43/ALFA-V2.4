import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

# Sayfa Yapılandırması
st.set_page_config(page_title="ALFA V2.4 Borsa Algoritma Modeli", layout="wide")

st.title("📈 BIST Algoritmik Hisse Seçim Modeli (ALFA V2.4)")
st.markdown("Temel + Teknik Filtreleme -> Ağırlıklı Skorlama -> İlk 5 Hisse Portföy Adayı")

# --- 1. OTOMATİK XU100 ENDEKS GETİRİLERİ (RESMİ KAPANIŞ VERİLERİ) ---
@st.cache_data(ttl=3600)
def otomatik_xu100_getirileri():
    try:
        bist100 = yf.download("XU100.IS", period="3mo", progress=False)
        if not bist100.empty and len(bist100) >= 40:
            fiyatlar = bist100['Close']
            if isinstance(fiyatlar, pd.DataFrame):
                fiyatlar = fiyatlar.iloc[:, 0]
            
            bugun_kapanis = float(fiyatlar.iloc[-1])
            iki_hafta_once_kapanis = float(fiyatlar.iloc[-10])
            iki_ay_once_kapanis = float(fiyatlar.iloc[-40])
            
            xu100_2h = float(((bugun_kapanis - iki_hafta_once_kapanis) / iki_hafta_once_kapanis) * 100)
            xu100_2a = float(((bugun_kapanis - iki_ay_once_kapanis) / iki_ay_once_kapanis) * 100)
            
            return xu100_2h, xu100_2a
    except Exception as e:
        print("Hata:", e)
    return -1.04, 0.40

oto_2h, oto_2a = otomatik_xu100_getirileri()

bugun = pd.Timestamp.today()
fmt = "%d.%m.%Y"
bugun_str = bugun.strftime(fmt)
iki_hafta_once = (bugun - pd.DateOffset(weeks=2)).strftime(fmt)
iki_ay_once = (bugun - pd.DateOffset(months=2)).strftime(fmt)

# --- 2. YAN MENÜ: MAKRO GİRDİLERİ ---
st.sidebar.header("⚙️ Makro Girdiler (Otomatik & Güncel)")
tufe_12 = st.sidebar.number_input("TÜFE(12) Yıllık %", value=31.75, format="%.2f")
xu100_2a = st.sidebar.number_input(f"XU100 2-Aylık Getiri % ({iki_ay_once} - {bugun_str})", value=oto_2a, format="%.2f")
xu100_2h = st.sidebar.number_input(f"XU100 2-Haftalık Getiri % ({iki_hafta_once} - {bugun_str})", value=oto_2h, format="%.2f")

st.sidebar.header("📁 Fintables Veri Yükleme")
file1 = st.sidebar.file_uploader("1. Temel Analiz & Fiyat Dosyası", type=["xlsx", "xls"])
file2 = st.sidebar.file_uploader("2. Eski Dönemler Dosyası (1)", type=["xlsx", "xls"])

# --- DETAYLI HİSSE TEŞHİS PANELİ (SOL MENÜ) ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Detaylı Hisse Teşhis Paneli")
aranan_hisse = st.sidebar.text_input("Hisse Kodu Kontrol Et (Örn: CRDFA, EGEGY)", "").upper().strip()

if file1 and file2:
    df1 = pd.read_excel(file1).iloc[:, :23]
    df1.columns = ["Kod", "ROE_0", "ROE_1", "ROE_4", "BrutEFK_0", "BrutEFK_1", "EFK_0", "EFK_1", "EFK_4", "FAVOK_0", "FAVOK_1", "NetSatisBuyume", "FAVOKBuyume", "BrutEFKBuyume", "EFKBuyume", "PDDD", "NetBorc_FAVOK", "HAOran", "Getiri_2h", "Getiri_1a", "Getiri_2a", "Getiri_6a", "Kapanis"]
    df2 = pd.read_excel(file2).iloc[:, :7]
    df2.columns = ["Kod", "ROE_2", "BrutEFK_2", "EFK_2", "FAVOK_2", "ROE_5", "EFK_5"]
    df = pd.merge(df1, df2, on="Kod", how="left").fillna(0)

    bilanco_gelmedi = ((df["BrutEFK_0"] == df["BrutEFK_1"]) & (df["EFK_0"] == df["EFK_1"]) & (df["FAVOK_0"] == df["FAVOK_1"]))
    df["Bilanco_Durum"] = np.where(bilanco_gelmedi, "GELMEDİ (Eski)", "GELDİ (Yeni)")

    # Filtreleme Mantığı
    df["pdddLimit"] = np.where(df["ROE_0"] > 90, 8 + (df["ROE_0"] - 90) * 0.07, 8)
    temel_filtreliler = (df["Getiri_2a"] > xu100_2a) & (df["ROE_0"] > tufe_12) & (df["NetBorc_FAVOK"] < 4) & (df["PDDD"] < df["pdddLimit"])
    df_temel = df[temel_filtreliler].copy()

    # --- ANA EKRAN SEKMELERİ ---
    tab1, tab2 = st.tabs(["📊 Tarama Sonuçları", "🔍 Hisse Teşhis"])

    with tab1:
        st.success(f"Temel Kriterleri Geçen Hisse: **{len(df_temel)}**")
        st.dataframe(df_temel[["Kod", "Bilanco_Durum", "ROE_0", "PDDD", "Getiri_2a"]])

    with tab2:
        if aranan_hisse:
            th = df[df["Kod"] == aranan_hisse].iloc[0] if aranan_hisse in df["Kod"].values else None
            if th is not None:
                st.write(f"### {aranan_hisse} Analiz")
                st.write(f"- ROE ({th['ROE_0']}) > {tufe_12}: {'✅' if th['ROE_0'] > tufe_12 else '❌'}")
                st.write(f"- 2A Getiri ({th['Getiri_2a']}) > {xu100_2a}: {'✅' if th['Getiri_2a'] > xu100_2a else '❌'}")
            else:
                st.warning("Hisse bulunamadı.")
else:
    st.info("👈 Fintables dosyalarını yükleyin.")
