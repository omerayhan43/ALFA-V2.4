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

# --- 1. OTOMATİK XU100 ENDEKS & MAKRO GİRDİLERİ (ARKA PLAN) ---
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
        
        get_price = lambda d: float(df.loc[d]['Close'].iloc[0] if isinstance(df.loc[d]['Close'], pd.Series) else df.loc[d]['Close'])
        
        price_today = get_price(day_t)
        price_t2w = get_price(day_t2w)
        price_t2m = get_price(day_t2m)
        
        xu100_2h = ((price_today - price_t2w) / price_t2w) * 100
        xu100_2a = ((price_today - price_t2m) / price_t2m) * 100
        
        return 31.75, float(xu100_2h), float(xu100_2a)
    except:
        return 31.75, -1.98, 0.76

tufe_12, oto_2h, oto_2a = otomatik_makro_veriler()

# --- SESSION STATE (Veri ve Zaman Kalıcılığı) ---
if "df_merged" not in st.session_state: st.session_state.df_merged = None
if "upload_time" not in st.session_state: st.session_state.upload_time = None

# --- ÜST BAŞLIK & SAĞ ÜSTTE OTOMATİK MAKRO GÖSTERGESİ ---
header_col1, header_col2 = st.columns([3, 2])
with header_col1:
    st.title("📈 BIST Algoritmik Hisse Seçim Modeli (ALFA V2.4)")
    st.markdown("Temel + Teknik Filtreleme -> Ağırlıklı Skorlama -> İlk 5 Hisse Portföy Adayı")

with header_col2:
    zaman_bilgisi = f"<br><span>🕒 Yükleme: <b>{st.session_state.upload_time}</b></span>" if st.session_state.upload_time else ""
    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef; font-size: 13px; text-align: right;">
            <b>⚙️ Otomatik Makro Girdiler:</b><br>
            <span>TÜFE(12): <b>%{tufe_12:.2f}</b> | XU100 2A: <b>%{oto_2a:.2f}</b> | XU100 2H: <b>%{oto_2h:.2f}</b></span>
            {zaman_bilgisi}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# --- 2. SOL MENÜ (NAVİGASYON) ---
st.sidebar.markdown("### ⚡ ALFA Terminal")
st.sidebar.markdown("---")

menu_secim = st.sidebar.radio(
    "Navigasyon",
    [
        "📊 Radar & Taramalar", 
        "🔍 Hisse Teşhis Paneli", 
        "📈 Hareketli Ortalama İnceleme", 
        "📁 Veri Yönetimi (Excel Yükle)"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size: 12px; color: #6c757d;">
        <b>Sistem Bilgisi</b><br>
        • Sürüm: ALFA V2.4<br>
        • Veri Kaynağı: Fintables & yfinance<br>
        • Durum: Aktif
    </div>
    """,
    unsafe_allow_html=True
)

# --- 3. VERİ YÖNETİMİ VE DOSYA YÜKLEME EKRANI ---
if menu_secim == "📁 Veri Yönetimi (Excel Yükle)":
    st.subheader("📁 Fintables Veri Yönetimi ve Excel Yükleme")
    st.markdown("Modelin tarama yapabilmesi için güncel Fintables Excel dosyalarınızı aşağıya yükleyin. Yüklediğiniz dosyalar sekmeler arasında gezindiğinizde **asla silinmez**.")
    
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        file1 = st.file_uploader("1. Temel Analiz & Fiyat Dosyası", type=["xlsx", "xls"], key="f1_up")
    with col_up2:
        file2 = st.file_uploader("2. Eski Dönemler Dosyası (1)", type=["xlsx", "xls"], key="f2_up")

    if file1 and file2:
        try:
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

            st.session_state.df_merged = df
            st.session_state.upload_time = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            st.success(f"✅ Excel dosyaları başarıyla işlendi ve hafızaya kaydedildi! (Yükleme Zamanı: {st.session_state.upload_time}) Sol menüden **'📊 Radar & Taramalar'** sekmesine geçebilirsiniz.")
        except Exception as e:
            st.error(f"Dosyalar işlenirken hata oluştu: {e}")
    else:
        st.info("👈 Lütfen her iki Excel dosyasını da yükleyin.")

# Hafızada veri varsa diğer sekmeler aktif çalışır
if st.session_state.df_merged is not None:
    df = st.session_state.df_merged

    # --- TEMEL KRİTERLER (FİLTRELEME) ---
    df["pdddLimit"] = np.where(df["ROE_0"] > 90, 8 + (df["ROE_0"] - 90) * 0.07, 8)
    
    a = df["Getiri_2a"] > oto_2a
    b = df["ROE_0"] > tufe_12
    c = df["NetBorc_FAVOK"] < 4
    d = df["PDDD"] < df["pdddLimit"]
    ee = df["Getiri_2h"] > (oto_2h - 10)
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

    # --- 4. RADAR & TARAMALAR EKRANI ---
    if menu_secim == "📊 Radar & Taramalar":
        teknik_asanadan_gecenler = []
        if len(df_temel) > 0:
            with st.spinner("🔄 Temelden geçen hisselerin hareketli ortalamaları (MA20, MA75, MA200) anlık çekiliyor..."):
                for idx, row in df_temel.iterrows():
                    kod = str(row["Kod"]).strip() + ".IS"
                    try:
                        hist = yf.download(kod, period="1y", progress=False)
                        if not hist.empty:
                            hist = hist.sort_index().dropna()
                            close = hist['Close']
                            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
                            if len(close) >= 200:
                                ma20 = float(close.rolling(20).mean().iloc[-1])
                                ma75 = float(close.rolling(75).mean().iloc[-1])
                                ma200 = float(close.rolling(200).mean().iloc[-1])
                                df_temel.loc[idx, 'MA20'] = round(ma20, 2)
                                df_temel.loc[idx, 'MA75'] = round(ma75, 2)
                                df_temel.loc[idx, 'MA200'] = round(ma200, 2)
                                if ma75 > ma200 and ma20 > ma75:
                                    teknik_asanadan_gecenler.append(idx)
                    except:
                        continue

        df_teknik = df_temel.loc[teknik_asanadan_gecenler].copy()

        # Modern Metrik Kartları (KPIs)
        m1, m2, m3 = st.columns(3)
        m1.metric(label="📊 Toplam İncelenen", value=f"{len(df)} Hisse")
        m2.metric(label="🏛️ Temel Filtreyi Geçen", value=f"{len(df_temel)} Hisse")
        m3.metric(label="🏆 Teknik & Trendi Geçen (Adaylar)", value=f"{len(df_teknik)} Hisse")
        
        st.markdown("---")

        if len(df_temel) > 0:
            if len(df_teknik) > 0:
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
                    (roeSkor * 0.30) + (buySkor * 0.20) + (momentumSkor * 0.15) + 
                    (m2Skor * 0.05) + (trendSkor * 0.18) + (pdddSkor * 0.12) + negCeza + ardisikCeza
                )

                df_sonuc = df_teknik.sort_values(by="SKOR", ascending=False).reset_index(drop=True)
                df_sonuc.index += 1

                st.markdown("### 🏆 Nihai Portföy Adayları (En İyi Skorlar)")
                
                def highlight_top5(s):
                    return ['background-color: #d4edda; font-weight: bold;' if s.name <= 5 else '' for _ in s]
                
                st.dataframe(
                    df_sonuc[["Kod", "SKOR", "Bilanco_Durum", "ROE_0", "PDDD", "Getiri_1a", "Getiri_6a"]]
                    .head(15)
                    .style.format(precision=2, subset=["SKOR", "ROE_0", "PDDD", "Getiri_1a", "Getiri_6a"])
                    .apply(highlight_top5, axis=1),
                    use_container_width=True
                )
            else:
                st.warning("Teknik kriterleri (MA20 > MA75 > MA200) sağlayan hisse bulunamadı.")

            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("📋 Arka Plan Verileri: Temel Kriterleri Geçen Tüm Hisseler"):
                st.dataframe(
                    df_temel[["Kod", "Bilanco_Durum", "ROE_0", "PDDD", "NetBorc_FAVOK", "Getiri_2a"]]
                    .style.format(precision=2, subset=["ROE_0", "PDDD", "NetBorc_FAVOK", "Getiri_2a"]),
                    use_container_width=True
                )

            with st.expander("🔍 Arka Plan Verileri: MA Karşılaştırma Tablosu"):
                st.success("✅ MA verileri başarıyla çekildi.")
                st.dataframe(
                    df_temel[["Kod", "Kapanis", "MA20", "MA75", "MA200", "Bilanco_Durum"]]
                    .style.format(precision=2, subset=["Kapanis", "MA20", "MA75", "MA200"]),
                    use_container_width=True
                )
        else:
            st.warning("Temel kriterleri sağlayan hisse bulunamadı.")

    # --- 5. HİSSE TEŞHİS PANELİ ---
    elif menu_secim == "🔍 Hisse Teşhis Paneli":
        st.markdown("### 🔍 Hisseler İçin Derinlemesine Teşhis ve Puanlama Paneli")
        st.markdown("İstediğiniz hisse kodunu seçerek temel filtrelerden geçip geçmediğini, takıldığı kriterleri, teknik MA durumunu ve modelden aldığı skor detaylarını inceleyebilirsiniz.")
        
        hisse_listesi = [""] + sorted(df["Kod"].dropna().astype(str).str.upper().unique().tolist())
        secilen_hisse = st.selectbox("Hisse Seçin / Arayın (Örn: CRDFA, EGEGY, FORTE)", options=hisse_listesi)

        if secilen_hisse:
            tek_hisse_df = df[df["Kod"].str.upper() == secilen_hisse]
            if not tek_hisse_df.empty:
                th = tek_hisse_df.iloc[0]
                p_lim = float(8 + (th['ROE_0'] - 90) * 0.07 if th['ROE_0'] > 90 else 8)
                
                c_a = bool(th['Getiri_2a'] > oto_2a)
                c_b = bool(th['ROE_0'] > tufe_12)
                c_c = bool(th['NetBorc_FAVOK'] < 4)
                c_d = bool(th['PDDD'] < p_lim)
                c_ee = bool(th['Getiri_2h'] > (oto_2h - 10))
                c_f = bool(th['NetSatisBuyume'] > 0)
                c_g = bool(th['FAVOKBuyume'] > tufe_12)
                c_h = bool((th['BrutEFKBuyume'] > tufe_12) and (th['EFKBuyume'] > tufe_12))
                c_j = bool(th['Getiri_1a'] > -15)
                c_k = bool(th['HAOran'] < 60)
                c_teyit = bool((th['EFK_0'] >= th['ef_EFK_1']) or (th['EFK_0'] >= th['ef_EFK_4']))
                c_roe_t = bool((th['ROE_0'] >= th['ef_ROE_1']) or (th['ROE_0'] >= th['ef_ROE_4']))
                
                takilan_kriterler = []
                if not c_b: takilan_kriterler.append("ROE > TÜFE")
                if not c_a: takilan_kriterler.append("2A Getiri > XU100")
                if not c_ee: takilan_kriterler.append("2H Getiri Şartı")
                if not c_d: takilan_kriterler.append("PD/DD < Sınır")
                if not c_c: takilan_kriterler.append("Net Borç / FAVÖK < 4")
                if not c_f: takilan_kriterler.append("Net Satış Büyümesi > 0")
                if not (c_g or c_h or c_teyit): takilan_kriterler.append("Büyüme / Teyit Kriterleri")
                if not c_roe_t: takilan_kriterler.append("ROE Teyit Şartı")
                if not c_j: takilan_kriterler.append("1A Getiri > -15")
                if not c_k: takilan_kriterler.append("Halka Açıklık < 60")

                roe_val = float(th['ROE_0'])
                m6_val = float(th['Getiri_6a'])
                m2_val = float(th['Getiri_2a'])
                m1_val = float(th['Getiri_1a'])
                pddd_val = float(th['PDDD'])
                eb_val = float(th['BrutEFKBuyume'])
                fb_val = float(th['FAVOKBuyume'])
                sb_val = float(th['NetSatisBuyume'])

                r_skor = float(100 if roe_val > 50 else roe_val * 2)
                m_skor = float(100 if m6_val > 100 else m6_val)
                m2_skor = float(100 if m2_val > 50 else m2_val * 2)
                tr_ydz = float(((float(th['Kapanis']) - 10) / 10) * 100)
                tr_skor = float(30 if tr_ydz > 30 else (0 if tr_ydz < 0 else tr_ydz))
                p_skor = float(25 if pddd_val < 1.5 else (15 if pddd_val < 3 else (5 if pddd_val < 5 else (-10 if pddd_val > 6 else 0))))
                
                eb_s = float(100 if eb_val > 50 else (eb_val * 2 if eb_val > 0 else 0))
                fb_s = float(100 if fb_val > 50 else (fb_val * 2 if fb_val > 0 else 0))
                sb_s = float(100 if sb_val > 50 else (sb_val * 2 if sb_val > 0 else 0))
                b_skor = float((eb_s + fb_s + sb_s) / 3)
                
                n_ceza = float(-15 if m1_val < -10 else (-8 if m1_val < -5 else 0))
                ard_ceza = float(-10 if (m1_val < 0 and m2_val < 0) else 0)
                
                nihai_skor = float(
                    (r_skor * 0.30) + (b_skor * 0.20) + (m_skor * 0.15) + 
                    (m2_skor * 0.05) + (tr_skor * 0.18) + (p_skor * 0.12) + n_ceza + ard_ceza
                )

                # 3 Sütunlu Yan Yana Tasarım
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"#### 🏛️ Temel Analiz ({secilen_hisse})")
                    st.write(f"- ROE > TÜFE: {'✅ Geçti' if c_b else '❌ Kaldı'}")
                    st.write(f"- 2A Getiri > XU100: {'✅ Geçti' if c_a else '❌ Kaldı'}")
                    st.write(f"- 2H Getiri Şartı: {'✅ Geçti' if c_ee else '❌ Kaldı'}")
                    st.write(f"- PD/DD < Sınır: {'✅ Geçti' if c_d else '❌ Kaldı'}")
                    st.write(f"- Net Borç/FAVÖK: {'✅ Geçti' if c_c else '❌ Kaldı'}")
                    st.write(f"- Net Satış Büyümesi: {'✅ Geçti' if c_f else '❌ Kaldı'}")
                    st.write(f"- Büyüme/Teyit: {'✅ Geçti' if (c_g or c_h or c_teyit) else '❌ Kaldı'}")
                    st.write(f"- ROE Teyit: {'✅ Geçti' if c_roe_t else '❌ Kaldı'}")
                    st.write(f"- 1A Getiri > -15: {'✅ Geçti' if c_j else '❌ Kaldı'}")
                    st.write(f"- Halka Açıklık: {'✅ Geçti' if c_k else '❌ Kaldı'}")

                with col2:
                    st.markdown(f"#### 🎯 Skor: {nihai_skor:.2f}")
                    st.write(f"- ROE Katkısı (%30): {(r_skor * 0.30):.2f}")
                    st.write(f"- Büyüme Katkısı (%20): {(b_skor * 0.20):.2f}")
                    st.write(f"- Trend Katkısı (%18): {(tr_skor * 0.18):.2f}")
                    st.write(f"- Momentum Katkısı (%15): {(m_skor * 0.15):.2f}")
                    st.write(f"- PD/DD Katkısı (%12): {(p_skor * 0.12):.2f}")

                with col3:
                    st.markdown(f"#### 📈 Teknik Detaylar")
                    try:
                        h_yf = yf.download(secilen_hisse + ".IS", period="1y", progress=False)
                        if not h_yf.empty:
                            if isinstance(h_yf.columns, pd.MultiIndex): h_yf.columns = h_yf.columns.get_level_values(0)
                            h_yf = h_yf.sort_index().dropna()
                            cls = h_yf['Close']
                            if isinstance(cls, pd.DataFrame): cls = cls.iloc[:, 0]
                            if len(cls) >= 200:
                                m20 = float(cls.rolling(20).mean().iloc[-1])
                                m75 = float(cls.rolling(75).mean().iloc[-1])
                                m200 = float(cls.rolling(200).mean().iloc[-1])
                                teknik_gecti = bool((m75 > m200) and (m20 > m75))
                                st.write(f"- **Kapanış:** {float(th['Kapanis']):.2f}")
                                st.write(f"- **MA20:** {m20:.2f}")
                                st.write(f"- **MA75:** {m75:.2f}")
                                st.write(f"- **MA200:** {m200:.2f}")
                                st.write(f"- **Teknik Kural:** {'✅ Sağlıyor' if teknik_gecti else '❌ Sağlamıyor'}")
                            else:
                                st.warning("Yeterli fiyat geçmişi yok.")
                        else:
                            st.warning("Yahoo Finance verisi alınamadı.")
                    except Exception as e:
                        st.error(f"Teknik hata: {e}")

                # En Altta Hata Listesi
                if takilan_kriterler:
                    st.markdown("---")
                    st.warning("Hisse aşağıdaki kriterleri sağlamadığı için filtrelerden geçemedi;")
                    for kriter in takilan_kriterler:
                        st.write(f"• {kriter}")
                else:
                    st.success("🎉 Tebrikler! Hisse tüm temel ve teknik filtrelerden başarıyla geçti.")
            else:
                st.warning(f"'{secilen_hisse}' kodlu hisse Fintables dosyalarında bulunamadı.")
else:
    if menu_secim != "📁 Veri Yönetimi (Excel Yükle)":
        st.warning("👈 Lütfen sol menüden **'📁 Veri Yönetimi (Excel Yükle)'** seçeneğine tıklayarak dosyalarınızı yükleyin.")
    # --- YENİ: HAREKETLİ ORTALAMA İNCELEME PANELİ ---
    elif menu_secim == "📈 Hareketli Ortalama İnceleme":
        st.markdown("### 📈 Hareketli Ortalama ve Trend İnceleme Paneli")
        st.markdown("İstediğiniz hissenin fiyat hareketlerini ve MA20, MA75, MA200 ortalamalarını renkli, interaktif bir grafik üzerinde inceleyin.")
        
        # Anlık arama selectbox
        hisse_listesi = [""] + sorted(df["Kod"].dropna().astype(str).str.upper().unique().tolist())
        secilen_hisse = st.selectbox("İncelenecek Hisseyi Seçin:", options=hisse_listesi, key="ma_inceleme_select")

        if secilen_hisse:
            try:
                import plotly.graph_objects as go
                
                h_yf = yf.download(secilen_hisse + ".IS", period="1y", progress=False)
                if not h_yf.empty:
                    if isinstance(h_yf.columns, pd.MultiIndex):
                        h_yf.columns = h_yf.columns.get_level_values(0)
                    h_yf = h_yf.sort_index().dropna()
                    
                    cls = h_yf['Close']
                    if isinstance(cls, pd.DataFrame): cls = cls.iloc[:, 0]
                    
                    # Hareketli Ortalamalar
                    h_yf['MA20'] = cls.rolling(20).mean()
                    h_yf['MA75'] = cls.rolling(75).mean()
                    h_yf['MA200'] = cls.rolling(200).mean()
                    
                    # Plotly Grafik Yapısı (Renk Kodlu ve Net Çizgiler)
                    fig = go.Figure()
                    
                    # Fiyat Çizgisi (Koyu Mavi)
                    fig.add_trace(go.Scatter(x=h_yf.index, y=cls, mode='lines', name='Kapanış (Fiyat)', line=dict(color='#1f77b4', width=2)))
                    # MA20 (Camgöbeği / Açık Mavi)
                    fig.add_trace(go.Scatter(x=h_yf.index, y=h_yf['MA20'], mode='lines', name='MA20', line=dict(color='#17becf', width=1.5)))
                    # MA75 (Turuncu)
                    fig.add_trace(go.Scatter(x=h_yf.index, y=h_yf['MA75'], mode='lines', name='MA75', line=dict(color='#ff7f0e', width=1.5)))
                    # MA200 (Kırmızı)
                    fig.add_trace(go.Scatter(x=h_yf.index, y=h_yf['MA200'], mode='lines', name='MA200', line=dict(color='#d62728', width=2)))
                    
                    # Grafik Tasarım Düzenlemeleri
                    fig.update_layout(
                        title=f"{secilen_hisse} - Detaylı Fiyat ve MA Karşılaştırması",
                        xaxis_title="Tarih",
                        yaxis_title="Fiyat (TL)",
                        height=550,
                        hovermode="x unified", # Fareyi gezdirdiğinizde tüm değerleri tek kutuda gösterir
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Son Değerleri Eşleştirebilmek İçin Özet Metrik Paneli
                    son_fiyat = float(cls.iloc[-1])
                    son_ma20 = float(h_yf['MA20'].iloc[-1]) if not pd.isna(h_yf['MA20'].iloc[-1]) else 0
                    son_ma75 = float(h_yf['MA75'].iloc[-1]) if not pd.isna(h_yf['MA75'].iloc[-1]) else 0
                    son_ma200 = float(h_yf['MA200'].iloc[-1]) if not pd.isna(h_yf['MA200'].iloc[-1]) else 0
                    
                    st.markdown("#### 📊 Güncel Değer Eşleşme Tablosu")
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Son Fiyat", f"{son_fiyat:.2f} TL")
                    col_m2.metric("MA20 Değeri", f"{son_ma20:.2f} TL")
                    col_m3.metric("MA75 Değeri", f"{son_ma75:.2f} TL")
                    col_m4.metric("MA200 Değeri", f"{son_ma200:.2f} TL")
                    
                else:
                    st.warning("Yahoo Finance üzerinden bu hisse için veri bulunamadı.")
            except Exception as e:
                st.error(f"Grafik yüklenirken bir hata oluştu: {e}")
                st.info("İpucu: Plotly kütüphanesinin aktif olduğundan emin olun.")
