import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

# Sayfa Yapılandırması
st.set_page_config(page_title="ALFA V2.4 Borsa Algoritma Modeli", layout="wide")

st.title("📈 BIST Algoritmik Hisse Seçim Modeli (ALFA V2.4)")
st.markdown("Temel + Teknik Filtreleme -> Ağırlıklı Skorlama -> İlk 5 Hisse Portföy Adayı")

# --- 1. OTOMATİK XU100 ENDEKS GETİRİLERİ HESAPLAMA ---
@st.cache_data(ttl=3600)
def otomatik_xu100_getirileri():
    try:
        bist100 = yf.download("XU100.IS", period="3mo", progress=False)
        if not bist100.empty and len(bist100) >= 40:
            fiyatlar = bist100['Close']
            if isinstance(fiyatlar, pd.DataFrame):
                fiyatlar = fiyatlar.iloc[:, 0]
            bugun_fiyat = fiyatlar.iloc[-1]
            iki_hafta_once_fiyat = fiyatlar.iloc[-10]
            iki_ay_once_fiyat = fiyatlar.iloc[-40]
            
            xu100_2h = float(((bugun_fiyat - iki_hafta_once_fiyat) / iki_hafta_once_fiyat) * 100)
            xu100_2a = float(((bugun_fiyat - iki_ay_once_fiyat) / iki_ay_once_fiyat) * 100)
            return xu100_2h, xu100_2a
    except:
        pass
    return -5.57, -1.50

oto_2h, oto_2a = otomatik_xu100_getirileri()

bugun = pd.Timestamp.today()
fmt = "%d.%m.%Y"
bugun_str = bugun.strftime(fmt)
iki_hafta_once = (bugun - pd.DateOffset(weeks=2)).strftime(fmt)
iki_ay_once = (bugun - pd.DateOffset(months=2)).strftime(fmt)
bir_yil_once = (bugun - pd.DateOffset(years=1)).strftime(fmt)

# --- 2. YAN MENÜ: MAKRO GİRDİLERİ ---
st.sidebar.header("⚙️ Makro Girdiler (Otomatik & Güncel)")
tufe_12 = st.sidebar.number_input(f"TÜFE(12) Yıllık %", value=31.75, format="%.2f")
xu100_2a = st.sidebar.number_input(f"XU100 2-Aylık Getiri % ({iki_ay_once} - {bugun_str})", value=oto_2a, format="%.2f")
xu100_2h = st.sidebar.number_input(f"XU100 2-Haftalık Getiri % ({iki_hafta_once} - {bugun_str})", value=oto_2h, format="%.2f")

# --- 3. DOSYA YÜKLEME ---
st.sidebar.header("📁 Fintables Veri Yükleme")
file1 = st.sidebar.file_uploader("1. Temel Analiz & Fiyat Dosyası", type=["xlsx", "xls"])
file2 = st.sidebar.file_uploader("2. Eski Dönemler Dosyası (1)", type=["xlsx", "xls"])

if file1 and file2:
    # Veri Omurgası & Birleştirme
    df1 = pd.read_excel(file1).iloc[:, :23]
    df1.columns = [
        "Kod", "ROE_0", "ROE_1", "ROE_4", "BrutEFK_0", "BrutEFK_1", 
        "EFK_0", "EFK_1", "EFK_4", "FAVOK_0", "FAVOK_1", 
        "NetSatisBuyume", "FAVOKBuyume", "BrutEFKBuyume", "EFKBuyume", 
        "PDDD", "NetBorc_FAVOK", "HAOran", "Getiri_2h", "Getiri_1a", 
        "Getiri_2a", "Getiri_6a", "Kapanis"
    ]

    df2 = pd.read_excel(file2).iloc[:, :7]
    df2.columns = ["Kod", "ROE_2", "BrutEFK_2", "EFK_2", "FAVOK_2", "ROE_5", "EFK_5"]

    df = pd.merge(df1, df2, on="Kod", how="left").fillna(0)

    # Bilanço Geldi mi? Kontrolü & Kaydırma Mantığı
    bilanco_gelmedi = (
        (df["BrutEFK_0"] == df["BrutEFK_1"]) & 
        (df["EFK_0"] == df["EFK_1"]) & 
        (df["FAVOK_0"] == df["FAVOK_1"]) & 
        ((df["BrutEFK_0"] != 0) | (df["EFK_0"] != 0) | (df["FAVOK_0"] != 0))
    )

    df["Bilanco_Durum"] = np.where(bilanco_gelmedi, "GELMEDİ (Eski)", "GELDİ (Yeni)")

    df["ef_EFK_1"] = np.where(bilanco_gelmedi, np.where(df["EFK_2"] != 0, df["EFK_2"], df["EFK_1"]), df["EFK_1"])
    df["ef_EFK_4"] = np.where(bilanco_gelmedi, np.where(df["EFK_5"] != 0, df["EFK_5"], df["EFK_4"]), df["EFK_4"])
    df["ef_FAVOK_1"] = np.where(bilanco_gelmedi, np.where(df["FAVOK_2"] != 0, df["FAVOK_2"], df["FAVOK_1"]), df["FAVOK_1"])
    df["ef_BrutEFK_1"] = np.where(bilanco_gelmedi, np.where(df["BrutEFK_2"] != 0, df["BrutEFK_2"], df["BrutEFK_1"]), df["BrutEFK_1"])
    df["ef_ROE_1"] = np.where(bilanco_gelmedi, np.where(df["ROE_2"] != 0, df["ROE_2"], df["ROE_1"]), df["ROE_1"])
    df["ef_ROE_4"] = np.where(bilanco_gelmedi, np.where(df["ROE_5"] != 0, df["ROE_5"], df["ROE_4"]), df["ROE_4"])

    # --- 4. TEMEL KRİTERLER (FİLTRELEME) ---
    df["pdddLimit"] = np.where(df["ROE_0"] > 90, 8 + (df["ROE_0"] - 90) * 0.07, 8)
    
    a = df["Getiri_2a"] > xu100_2a
    b = df["ROE_0"] > tufe_12
    c = df["NetBorc_FAVOK"] < 4
    d = df["PDDD"] < df["pdddLimit"]
    ee = df["Getiri_2h"] > (xu100_2h - 10)
    f = df["NetSatisBuyume"] > 0
    g = df["FAVOKBuyume"] > tufe_12
    h = (df["BrutEFKBuyume"] > tufe_12) & (df["EFKBuyume"] > tufe_12)
    gx = (df["FAVOK_0"] > df["FAVOK_1"]) & (df["FAVOK_0"] > df["FAVOK_4"]) if "FAVOK_4" in df.columns else (df["FAVOK_0"] > df["FAVOK_1"])
    hx = (df["BrutEFK_0"] > df["BrutEFK_1"]) & (df["EFK_0"] > df["EFK_1"])
    j = df["Getiri_1a"] > -15
    efkTeyit = (df["EFK_0"] >= df["ef_EFK_1"]) | (df["EFK_0"] >= df["ef_EFK_4"])
    roeTeyit = (df["ROE_0"] >= df["ef_ROE_1"]) | (df["ROE_0"] >= df["ef_ROE_4"])
    k = df["HAOran"] < 60

    temel_filtreliler = a & b & c & d & ee & f & ((h | g) | (hx | gx) | efkTeyit) & j & roeTeyit & k
    df_temel = df[temel_filtreliler].copy()

    st.success(f"Toplam {len(df)} hisse içerisinden Temel Kriterleri geçen toplam hisse sayısı: **{len(df_temel)}**")

    if len(df_temel) > 0:
        # Temelden geçen tüm hisseleri bilanço durumuyla göster
        st.markdown("### 📋 Temel Kriterleri Geçen Tüm Hisseler (Bilanço Durumları)")
        st.dataframe(df_temel[["Kod", "Bilanco_Durum", "ROE_0", "PDDD", "NetBorc_FAVOK", "Getiri_2a"]])

        # --- 5. TEKNİK KRİTERLER VE MA HESAPLAMA (Yahoo Finance üzerinden anlık) ---
        st.info("🔄 Temelden geçen tüm hisselerin hareketli ortalamaları (MA20, MA75, MA200) anlık çekiliyor...")
        
        teknik_asanadan_gecenler = []
        for idx, row in df_temel.iterrows():
            kod = str(row["Kod"]).strip() + ".IS"
            try:
                hist = yf.download(kod, period="1y", progress=False)
                if not hist.empty:
                    close = hist['Close']
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    if len(close) >= 200:
                        ma20 = float(close.rolling(20).mean().iloc[-1])
                        ma75 = float(close.rolling(75).mean().iloc[-1])
                        ma200 = float(close.rolling(200).mean().iloc[-1])

                        df_temel.loc[idx, 'MA20'] = round(ma20, 2)
                        df_temel.loc[idx, 'MA75'] = round(ma75, 2)
                        df_temel.loc[idx, 'MA200'] = round(ma200, 2)

                        # Teknik Kriter: MA75 > MA200 ve MA20 > MA75
                        if ma75 > ma200 and ma20 > ma75:
                            teknik_asanadan_gecenler.append(idx)
            except Exception as e:
                continue

        df_teknik = df_temel.loc[teknik_asanadan_gecenler].copy()
        
        # Manuel kontrol için MA Karşılaştırma Tablosu
        st.markdown("### 🔍 Hareketli Ortalama (MA) Karşılaştırma Tablosu (Manuel Kontrol İçin)")
        st.dataframe(df_temel[["Kod", "Kapanis", "MA20", "MA75", "MA200", "Bilanco_Durum"]])

        st.success(f"Teknik kriterleri sağlayan hisse sayısı: **{len(df_teknik)}**")

        if len(df_teknik) > 0:
            # --- 6. SKORLAMA VE CEZA HESAPLAMA ---
            roe = df_teknik["ROE_0"]
            m6 = df_teknik["Getiri_6a"]
            m2 = df_teknik["Getiri_2a"]
            m1 = df_teknik["Getiri_1a"]
            c_fiyat = df_teknik["Kapanis"]
            ma75_val = df_teknik["MA75"]
            eb = df_teknik["BrutEFKBuyume"]
            fb = df_teknik["FAVOKBuyume"]
            sb = df_teknik["NetSatisBuyume"]

            roeSkor = np.where(roe > 50, 100, roe * 2)
            momentumSkor = np.where(m6 > 100, 100, m6)
            m2Skor = np.where(m2 > 50, 100, m2 * 2)
            
            trendYuzdesi = ((c_fiyat - ma75_val) / ma75_val) * 100
            trendSkor = np.where(trendYuzdesi > 30, 30, np.where(trendYuzdesi < 0, 0, trendYuzdesi))
            
            pddd = df_teknik["PDDD"]
            pdddSkor = np.where(pddd < 1.5, 25, np.where(pddd < 3, 15, np.where(pddd < 5, 5, np.where(pddd > 6, -10, 0))))

            ebSkor = np.where(eb > 50, 100, np.where(eb > 0, eb * 2, 0))
            fbSkor = np.where(fb > 50, 100, np.where(fb > 0, fb * 2, 0))
            sbSkor = np.where(sb > 50, 100, np.where(sb > 0, sb * 2, 0))
            buySkor = (ebSkor + fbSkor + sbSkor) / 3

            negCeza = np.where(m1 < -10, -15, np.where(m1 < -5, -8, 0))
            ardisikCeza = np.where((m1 < 0) & (m2 < 0), -10, 0)

            df_teknik["SKOR"] = (
                (roeSkor * 0.30) + 
                (buySkor * 0.20) + 
                (momentumSkor * 0.15) + 
                (m2Skor * 0.05) + 
                (trendSkor * 0.18) + 
                (pdddSkor * 0.12) + 
                negCeza + 
                ardisikCeza
            )

            df_sonuc = df_teknik.sort_values(by="SKOR", ascending=False).reset_index(drop=True)
            df_sonuc.index += 1

            st.markdown("### 🏆 Nihai Portföy Adayları (İlk 5 Hisse)")
            
            def highlight_top5(s):
                return ['background-color: #d4edda; font-weight: bold;' if s.name <= 5 else '' for _ in s]

            gosterge_tablosu = df_sonuc[["Kod", "SKOR", "Bilanco_Durum", "ROE_0", "PDDD", "Getiri_1a", "Getiri_6a"]].head(15)
            st.dataframe(gosterge_tablosu.style.apply(highlight_top5, axis=1))

        else:
            st.warning("Teknik kriterleri sağlayan hisse bulunamadı.")
    else:
        st.warning("Temel kriterleri sağlayan hisse bulunamadı.")
else:
    st.info("👈 Lütfen sol menüden Fintables Excel dosyalarını yükleyin.")
