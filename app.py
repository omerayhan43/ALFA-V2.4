import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import json
from zoneinfo import ZoneInfo
import yfinance as yf
import plotly.graph_objects as go

# Sayfa Yapılandırması
st.set_page_config(page_title="ALFA Terminal - Borsa Algoritma Modeli", layout="wide")

# İSTANBUL SAAT DİLİMİ (UTC+3)
TURKEY_TZ = ZoneInfo("Europe/Istanbul")

# VERİ KALICILIĞI İÇİN DOSYA YOLLARI
DATA_CACHE_PATH = "alfa_data_cache.pkl"
META_CACHE_PATH = "alfa_meta_cache.json"

# --- ÖZGÜN CSS (Pointer İşareti ve Menü Tasarımı) ---
st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] div[role="radiogroup"] > label {
        cursor: pointer !important;
        padding: 5px 10px !important;
        border-radius: 6px !important;
        font-size: 13px !important;
        color: #334155 !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 1. OTOMATİK ENDEKS & MAKRO GİRDİLERİ (ARKA PLAN) ---
# TÜFE(12): yfinance'ta yer almadığı için manuel sabit (her ay güncellenmeli)
TUFE_12 = 31.75

@st.cache_data(ttl=3600)
def endeks_getirileri(ticker):
    """Verilen endeksin son 2 haftalık ve 2 aylık getirisini İŞ GÜNÜ (asof) mantığıyla hesaplar.
    Hedef gün (14/60 takvim günü öncesi) hafta sonu/tatile denk gelirse, asof() otomatik olarak
    bir ÖNCEKİ iş gününe snap eder (endeks index'inde yalnızca işlem günleri bulunur).
    Ayrıca hangi tarihlerin baz alındığını döndürür ki kullanıcı manuel doğrulayabilsin."""
    try:
        df = yf.download(ticker, period="3mo", progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        bugun = pd.Timestamp.now(tz=TURKEY_TZ).tz_localize(None)
        gun_t = df.index.asof(bugun)                                  # bugün / son iş günü
        gun_2h = df.index.asof(bugun - pd.Timedelta(days=14))         # ~2 hafta önce (iş günü)
        gun_2a = df.index.asof(bugun - pd.Timedelta(days=60))         # ~2 ay önce (iş günü)
        if pd.isna(gun_t) or pd.isna(gun_2h) or pd.isna(gun_2a):
            return None

        def fiyat(d):
            v = df.loc[d, 'Close']
            return float(v.iloc[0] if isinstance(v, pd.Series) else v)

        p_t, p_2h, p_2a = fiyat(gun_t), fiyat(gun_2h), fiyat(gun_2a)
        return {
            "ret_2h": (p_t - p_2h) / p_2h * 100,
            "ret_2a": (p_t - p_2a) / p_2a * 100,
            "tarih_bugun": pd.Timestamp(gun_t).strftime("%d.%m.%Y"),
            "tarih_2h": pd.Timestamp(gun_2h).strftime("%d.%m.%Y"),
            "tarih_2a": pd.Timestamp(gun_2a).strftime("%d.%m.%Y"),
            "yedek": False,
        }
    except Exception:
        return None

# V2.4 -> BIST100 (XU100) | V1.4 -> BIST TÜM (XUTUM)
_YEDEK = {"ret_2h": -1.98, "ret_2a": 0.76, "tarih_bugun": "N/A", "tarih_2h": "N/A", "tarih_2a": "N/A", "yedek": True}
_xu100 = endeks_getirileri("XU100.IS") or dict(_YEDEK)
_xutum = endeks_getirileri("XUTUM.IS")
if _xutum is None:
    # XUTUM sembolü Yahoo'da çözülemezse: XU100 çekildiyse onu yedek olarak işaretleyip kullan
    _xutum = {**_xu100, "yedek": True} if _xu100.get("tarih_bugun") != "N/A" else dict(_YEDEK)

tufe_12 = TUFE_12
oto_2h, oto_2a = _xu100["ret_2h"], _xu100["ret_2a"]                    # V2.4 girdileri (XU100)
oto_xutum_2h, oto_xutum_2a = _xutum["ret_2h"], _xutum["ret_2a"]        # V1.4 girdileri (XUTUM)


# --- EFEKTİF (BİLANÇO-AYARLI) KOLON HESAPLAYICI ---
def ekle_efektif_kolonlar(df):
    """GELDİ/GELMEDİ tespiti + tüm efektif dönem kolonlarını kurar.
    Bir hissenin son çeyreği (0) ile -1 dönemi birebir aynıysa (fintables aynı değeri gösterir),
    o hissenin YENİ bilançosu henüz GELMEMİŞ demektir; bu durumda karşılaştırma dönemi bir
    çeyrek geriye kaydırılır (-1->-2, -4->-5). Hem V2.4 hem V1.4 bu kolonları kullanır."""
    bilanco_gelmedi = (
        (df["BrutEFK_0"] == df["BrutEFK_1"]) &
        (df["EFK_0"] == df["EFK_1"]) &
        (df["FAVOK_0"] == df["FAVOK_1"]) &
        ((df["BrutEFK_0"] != 0) | (df["EFK_0"] != 0) | (df["FAVOK_0"] != 0))
    )
    df["Bilanco_Durum"] = np.where(bilanco_gelmedi, "GELMEDİ (Eski)", "GELDİ (Yeni)")

    def ef(taban_1, taban_2):
        # GELMEDİ ise -2 dönemi (varsa) yoksa -1; GELDİ ise -1 dönemi
        return np.where(bilanco_gelmedi,
                        np.where(df[taban_2] != 0, df[taban_2], df[taban_1]),
                        df[taban_1])

    # V2.4 + ortak
    df["ef_EFK_1"] = ef("EFK_1", "EFK_2")
    df["ef_EFK_4"] = ef("EFK_4", "EFK_5")
    df["ef_FAVOK_1"] = ef("FAVOK_1", "FAVOK_2")
    df["ef_BrutEFK_1"] = ef("BrutEFK_1", "BrutEFK_2")
    df["ef_ROE_1"] = ef("ROE_1", "ROE_2")
    df["ef_ROE_4"] = ef("ROE_4", "ROE_5")
    # V1.4 için ek: -4 finansallar (GELMEDİ fallback -5)
    df["ef_FAVOK_4"] = ef("FAVOK_4", "FAVOK_5")
    df["ef_BrutEFK_4"] = ef("BrutEFK_4", "BrutEFK_5")
    # V1.4 için ek: gecikmeli yıllık büyüme (ivme karşılaştırması)
    df["ef_EFKBuyume_1"] = ef("EFKBuyume_1", "EFKBuyume_2")
    df["ef_FAVOKBuyume_1"] = ef("FAVOKBuyume_1", "FAVOKBuyume_2")
    return df


# --- TEŞHİS PANELİ DEĞER BİÇİMLEYİCİ ---
def dg(x):
    """Teşhis panellerinde kriter değerini kısa/okunur formatlar (Mr=milyar, Mn=milyon)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(xf) >= 1e9:
        return f"{xf/1e9:.2f}Mr"
    if abs(xf) >= 1e6:
        return f"{xf/1e6:.2f}Mn"
    return f"{xf:.2f}"


# ============================================================================
# ORTAK PORTFÖY (KONSENSÜS) — 5 algoritmayı arka planda çalıştıran yardımcılar
# ============================================================================
def _pdddlimit(df):
    return np.where(df["ROE_0"] > 90, 8 + (df["ROE_0"] - 90) * 0.07, 8)

def _temel_v14(df):
    pl = _pdddlimit(df)
    a = df["ROE_0"] > tufe_12
    b = df["NetBorc_FAVOK"] < 4
    d = df["PDDD"] < pl
    ee = df["Getiri_2h"] > (oto_xutum_2h - 10)
    f = df["NetSatisBuyume"] > 0
    gx = (df["FAVOKBuyume"] > tufe_12) | ((df["FAVOK_0"] > df["ef_FAVOK_1"]) & (df["FAVOK_0"] > df["ef_FAVOK_4"]))
    hx = ((df["BrutEFKBuyume"] > tufe_12) | ((df["BrutEFK_0"] > df["ef_BrutEFK_1"]) & (df["BrutEFK_0"] > df["ef_BrutEFK_4"]))) & \
         ((df["EFKBuyume"] > tufe_12) | ((df["EFK_0"] > df["ef_EFK_1"]) & (df["EFK_0"] > df["ef_EFK_4"])))
    h = df["Getiri_2a"] > oto_xutum_2a
    j = df["Getiri_1a"] > -15
    efkTeyit = (df["EFK_0"] >= df["ef_EFK_1"]) | (df["EFK_0"] >= df["ef_EFK_4"])
    roeTeyit = (df["ROE_0"] >= df["ef_ROE_1"]) | (df["ROE_0"] >= df["ef_ROE_4"])
    k = df["HAOran"] < 60
    return a & b & d & ee & f & (hx | gx | efkTeyit) & h & j & roeTeyit & k

def _temel_v24(df):
    pl = _pdddlimit(df)
    a = df["Getiri_2a"] > oto_2a
    b = df["ROE_0"] > tufe_12
    c = df["NetBorc_FAVOK"] < 4
    d = df["PDDD"] < pl
    ee = df["Getiri_2h"] > (oto_2h - 10)
    f = df["NetSatisBuyume"] > 0
    g = df["FAVOKBuyume"] > tufe_12
    h = (df["BrutEFKBuyume"] > tufe_12) & (df["EFKBuyume"] > tufe_12)
    gx = df["FAVOK_0"] > df["ef_FAVOK_1"]
    hx = (df["BrutEFK_0"] > df["ef_BrutEFK_1"]) & (df["EFK_0"] > df["ef_EFK_1"])
    j = df["Getiri_1a"] > -15
    efkTeyit = (df["EFK_0"] >= df["ef_EFK_1"]) | (df["EFK_0"] >= df["ef_EFK_4"])
    roeTeyit = (df["ROE_0"] >= df["ef_ROE_1"]) | (df["ROE_0"] >= df["ef_ROE_4"])
    k = df["HAOran"] < 60
    return a & b & c & d & ee & f & ((h | g) | (hx | gx) | efkTeyit) & j & roeTeyit & k

def _temel_v34(df):
    pl = _pdddlimit(df)
    a = df["Getiri_2a"] > oto_2a
    b = df["ROE_0"] > tufe_12
    c = df["NetBorc_FAVOK"] < 4
    d = df["PDDD"] < pl
    e = df["Getiri_2h"] > (oto_2h - 10)
    f = df["Getiri_1a"] > -15
    ns = df["NetSatisBuyume"] > 0
    gx = (df["FAVOKBuyume"] > tufe_12) | (df["FAVOK_0"] > df["ef_FAVOK_1"])
    hx = ((df["BrutEFKBuyume"] > tufe_12) | (df["BrutEFK_0"] > df["ef_BrutEFK_1"])) & \
         ((df["EFKBuyume"] > tufe_12) | (df["EFK_0"] > df["ef_EFK_1"]))
    efkTeyit = (df["EFK_0"] >= df["ef_EFK_1"]) | (df["EFK_0"] >= df["ef_EFK_4"])
    roeTeyit = (df["ROE_0"] >= df["ef_ROE_1"]) | (df["ROE_0"] >= df["ef_ROE_4"])
    k = df["HAOran"] < 60
    return a & b & c & d & e & f & ns & (hx | gx | efkTeyit) & roeTeyit & k

def _temel_v113(df):
    pl = _pdddlimit(df)
    a = df["Getiri_2a"] > oto_2a
    b = df["ROE_0"] > tufe_12
    c = df["NetBorc_FAVOK"] < 4
    d = df["PDDD"] < pl
    e = df["Getiri_2h"] > (oto_2h - 10)
    f = df["NetSatisBuyume"] > 0
    g = df["FAVOKBuyume"] > tufe_12
    h = (df["BrutEFKBuyume"] > tufe_12) & (df["EFKBuyume"] > tufe_12)
    j = df["Getiri_1a"] > -15
    k = df["HAOran"] < 60
    return a & b & c & d & e & f & (h | g) & j & k

def _temel_v123(df):
    pl = _pdddlimit(df)
    a = df["Getiri_2a"] > oto_2a
    b = df["ROE_0"] > tufe_12
    c = df["NetBorc_FAVOK"] < 4
    d = df["PDDD"] < pl
    e = df["Getiri_2h"] > (oto_2h - 10)
    f = df["Getiri_1a"] > -15
    return a & b & c & d & e & f

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_prices(kodlar):
    """Verilen hisseler için 1y fiyattan MA20/60/75/200 + HHV252*0.77 + canlı kapanış hesaplar (tekil)."""
    cache = {}
    for kod in kodlar:
        try:
            hist = yf.download(kod + ".IS", period="1y", progress=False)
            if hist.empty:
                continue
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            hist = hist.sort_index().dropna()
            close = hist['Close']
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            if len(close) < 200:
                continue
            high = hist['High']
            if isinstance(high, pd.DataFrame): high = high.iloc[:, 0]
            cache[kod] = {
                'c': float(close.iloc[-1]),
                'ma20': float(close.rolling(20).mean().iloc[-1]),
                'ma60': float(close.rolling(60).mean().iloc[-1]),
                'ma75': float(close.rolling(75).mean().iloc[-1]),
                'ma200': float(close.rolling(200).mean().iloc[-1]),
                'hhv': float((high.tail(252) if len(high) >= 252 else high).max()) * 0.77,
            }
        except Exception:
            continue
    return cache

def _skor_std(r, ma75, penalties):
    """V2.4/V1.1.3 (penalties=True) ve V3.4/V1.2.3 (penalties=False) ortak skoru. Trend fintables Kapanış'ı kullanır."""
    roe = r['ROE_0']; m6 = r['Getiri_6a']; m2 = r['Getiri_2a']; m1 = r['Getiri_1a']
    c = r['Kapanis']; eb = r['EFKBuyume']; fb = r['FAVOKBuyume']; sb = r['NetSatisBuyume']; pddd = r['PDDD']
    roeSkor = 100 if roe > 50 else roe * 2
    momS = 100 if m6 > 100 else m6
    m2S = 100 if m2 > 50 else m2 * 2
    trendY = ((c - ma75) / ma75) * 100 if ma75 else 0
    trendS = 30 if trendY > 30 else (0 if trendY < 0 else trendY)
    pdddS = 25 if pddd < 1.5 else (15 if pddd < 3 else (5 if pddd < 5 else (-10 if pddd > 6 else 0)))
    ebS = 100 if eb > 50 else (eb * 2 if eb > 0 else 0)
    fbS = 100 if fb > 50 else (fb * 2 if fb > 0 else 0)
    sbS = 100 if sb > 50 else (sb * 2 if sb > 0 else 0)
    buyS = (ebS + fbS + sbS) / 3
    base = roeSkor * 0.30 + buyS * 0.20 + momS * 0.15 + m2S * 0.05 + trendS * 0.18 + pdddS * 0.12
    if penalties:
        negCeza = -15 if m1 < -10 else (-8 if m1 < -5 else 0)
        ardisik = -10 if (m1 < 0 and m2 < 0) else 0
        base += negCeza + ardisik
    return base

def _skor_v14(r, p):
    """V1.4 skoru: trend/maUzaklık canlı kapanışı kullanır; ivmeBonus/favokIvme/buyumeBonus dahil."""
    roe = r['ROE_0']; m6 = r['Getiri_6a']; m2 = r['Getiri_2a']
    c = p['c']; ma75 = p['ma75']; ma200 = p['ma200']
    eb = r['EFKBuyume']; fb = r['FAVOKBuyume']; sb = r['NetSatisBuyume']; beb = r['BrutEFKBuyume']; pddd = r['PDDD']
    ef_eb1 = r['ef_EFKBuyume_1']; ef_fb1 = r['ef_FAVOKBuyume_1']
    roeSkor = 100 if roe > 50 else roe * 2
    momS = 100 if m6 > 100 else m6
    m2S = 100 if m2 > 50 else m2 * 2
    trendY = ((c - ma75) / ma75) * 100 if ma75 else 0
    trendS = 30 if trendY > 30 else (0 if trendY < 0 else trendY)
    ebS = 100 if eb > 50 else (eb * 2 if eb > 0 else 0)
    fbS = 100 if fb > 50 else (fb * 2 if fb > 0 else 0)
    sbS = 100 if sb > 50 else (sb * 2 if sb > 0 else 0)
    buyS = (ebS + fbS + sbS) / 3
    ivme = 15 if (eb > ef_eb1 and eb > 0) else (5 if eb > ef_eb1 else 0)
    favokIvme = 10 if (fb > ef_fb1 and fb > 0) else 0
    maU = ((c - ma200) / ma200) * 100 if ma200 else 0
    maUceza = -15 if maU > 80 else (-8 if maU > 60 else 0)
    pdddS = 25 if pddd < 1.5 else (15 if pddd < 3 else (5 if pddd < 5 else (-10 if pddd > 6 else 0)))
    buyumeBonus = 15 if (sb > 0 and ((beb > tufe_12 and eb > tufe_12) or fb > tufe_12)) else \
                  (0 if (sb > 0 or beb > tufe_12 or eb > tufe_12 or fb > tufe_12) else -10)
    return roeSkor * 0.30 + buyS * 0.20 + momS * 0.15 + trendS * 0.18 + m2S * 0.05 + pdddS * 0.12 + \
           ivme * 0.90 + favokIvme + maUceza + buyumeBonus

# (ad, temel_fonksiyonu, teknik_kontrolü, skor_fonksiyonu) — öncelik/getiri sırasına göre
_ALGO_DEFS = [
    ("V1.4",   _temel_v14,  lambda r, p: p['c'] > p['ma200'] and p['c'] > p['ma75'] and p['ma20'] > p['ma60'] and p['c'] > p['hhv'], lambda r, p: _skor_v14(r, p)),
    ("V2.4",   _temel_v24,  lambda r, p: p['ma75'] > p['ma200'] and p['ma20'] > p['ma75'],                                          lambda r, p: _skor_std(r, p['ma75'], True)),
    ("V3.4",   _temel_v34,  lambda r, p: p['ma20'] > p['ma60'] and p['ma75'] > p['ma200'] and p['c'] > p['ma75'],                   lambda r, p: _skor_std(r, p['ma75'], False)),
    ("V1.1.3", _temel_v113, lambda r, p: p['ma75'] > p['ma200'],                                                                    lambda r, p: _skor_std(r, p['ma75'], True)),
    ("V1.2.3", _temel_v123, lambda r, p: p['ma20'] > p['ma60'] and p['ma75'] > p['ma200'],                                          lambda r, p: _skor_std(r, p['ma75'], False)),
]

def _sirali(df, mask, px, tek_fn, skor_fn):
    """Temel+teknik geçen hisseleri skora göre sıralar; DataFrame(Kod, SKOR, rank) döndürür."""
    sub = df[mask]
    rows = []
    for _, r in sub.iterrows():
        p = px.get(r['Kod'])
        if p is None:
            continue
        if not tek_fn(r, p):
            continue
        rows.append((r['Kod'], float(skor_fn(r, p))))
    res = pd.DataFrame(rows, columns=['Kod', 'SKOR']).sort_values('SKOR', ascending=False).reset_index(drop=True)
    res['rank'] = res.index + 1
    return res

# --- SESSION STATE & F5 VERİ KALICILIĞI YÖNETİMİ ---
if "df_merged" not in st.session_state: st.session_state.df_merged = None
if "upload_time" not in st.session_state: st.session_state.upload_time = None

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "v24_radar"
    st.session_state.radio_v113 = None
    st.session_state.radio_v14 = None
    st.session_state.radio_v24 = "📊 Radar & Taramalar"
    st.session_state.radio_v34 = None
    st.session_state.radio_v123 = None
    st.session_state.radio_ortak = None
    st.session_state.radio_ma = None
    st.session_state.radio_veri = None

# F5 Yenilemelerinde Veriyi Diskten Yükleme
if st.session_state.df_merged is None:
    if os.path.exists(DATA_CACHE_PATH) and os.path.exists(META_CACHE_PATH):
        try:
            st.session_state.df_merged = pd.read_pickle(DATA_CACHE_PATH)
            with open(META_CACHE_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                st.session_state.upload_time = meta.get("upload_time")
        except Exception:
            pass

# --- NAVİGASYON CALLBACK SİSTEMİ ---
RADIO_KEYS = ["radio_v113", "radio_v14", "radio_v24", "radio_v34", "radio_v123", "radio_ortak", "radio_ma", "radio_veri"]

def create_nav_callback(key, mapping):
    def callback():
        val = st.session_state.get(key)
        if val:
            st.session_state.nav_page = mapping[val]
            for rk in RADIO_KEYS:
                if rk != key:
                    st.session_state[rk] = None
    return callback

map_v113 = {"📊 Radar & Taramalar": "v113_radar", "🔍 Hisse Teşhis Paneli": "v113_diag"}
map_v14 = {"📊 Radar & Taramalar": "v14_radar", "🔍 Hisse Teşhis Paneli": "v14_diag"}
map_v24 = {"📊 Radar & Taramalar": "v24_radar", "🔍 Hisse Teşhis Paneli": "v24_diag"}
map_v34 = {"📊 Radar & Taramalar": "v34_radar", "🔍 Hisse Teşhis Paneli": "v34_diag"}
map_v123 = {"📊 Radar & Taramalar": "v123_radar", "🔍 Hisse Teşhis Paneli": "v123_diag"}
map_ortak = {"🎯 Ortak Portföy Tablosu": "ortak_portfoy"}
map_ma = {"📈 Hareketli Ortalama İnceleme": "ma_review"}
map_veri = {"📁 Veri Yönetimi (Excel Yükle)": "data_mgmt"}

# --- ÜST DÜZEY AYAK (LEG) GEÇİŞİ ---
def _leg_cb():
    v = st.session_state.get("leg_radio", "")
    if "Portföy" in v:
        if not str(st.session_state.nav_page).startswith("portfoy"):
            st.session_state.nav_page = "portfoy_home"
    else:
        if str(st.session_state.nav_page).startswith("portfoy"):
            st.session_state.nav_page = "v24_radar"

# --- ÜST BAŞLIK & OTOMATİK MAKRO GÖSTERGESİ ---
header_col1, header_col2 = st.columns([3, 2])
with header_col1:
    st.title("📈 ALFA Terminal Algoritmik Hisse Seçim Modelleri")
    st.markdown("Çoklu Algoritma Taraması -> Ağırlıklı Skorlama -> Portföy Adayları")

with header_col2:
    zaman_bilgisi = f"<br><span>🕒 Yükleme: <b>{st.session_state.upload_time}</b></span>" if st.session_state.upload_time else ""
    xutum_not = " ⚠️(yedek: XU100)" if _xutum.get("yedek") else ""
    xu100_not = " ⚠️(yedek)" if _xu100.get("yedek") else ""
    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef; font-size: 12px; text-align: right;">
            <b>⚙️ Otomatik Makro Girdiler</b> &nbsp;|&nbsp; TÜFE(12): <b>%{tufe_12:.2f}</b><br>
            <span>📈 <b>XU100</b> (V2.4) → 2H: <b>%{oto_2h:.2f}</b> · 2A: <b>%{oto_2a:.2f}</b>{xu100_not}</span><br>
            <span style="font-size:10.5px;color:#6c757d;">2H: {_xu100['tarih_2h']} → {_xu100['tarih_bugun']} &nbsp;·&nbsp; 2A: {_xu100['tarih_2a']} → {_xu100['tarih_bugun']}</span><br>
            <span>📊 <b>XUTUM</b> (V1.4) → 2H: <b>%{oto_xutum_2h:.2f}</b> · 2A: <b>%{oto_xutum_2a:.2f}</b>{xutum_not}</span><br>
            <span style="font-size:10.5px;color:#6c757d;">2H: {_xutum['tarih_2h']} → {_xutum['tarih_bugun']} &nbsp;·&nbsp; 2A: {_xutum['tarih_2a']} → {_xutum['tarih_bugun']}</span>
            {zaman_bilgisi}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# --- SOL MENÜ NAVİGASYONU ---
st.sidebar.markdown("### ⚡ ALFA Terminal")
st.sidebar.markdown("---")

current_page = st.session_state.nav_page
aktif_leg = "portfoy" if str(current_page).startswith("portfoy") else "algoritma"

# Üst düzey: iki ana ayak
if "leg_radio" not in st.session_state:
    st.session_state.leg_radio = "🔍 Alfa Algoritmik Filtreleme"
st.sidebar.radio(
    "Uygulama Ayağı",
    options=["🔍 Alfa Algoritmik Filtreleme", "💼 Portföy Yönetimi"],
    key="leg_radio",
    on_change=_leg_cb,
    label_visibility="collapsed",
)
st.sidebar.markdown("---")

if aktif_leg == "algoritma":
    st.sidebar.markdown("#### 🔍 Alfa Algoritmik Filtreleme")
    is_v113_active = current_page in ["v113_radar", "v113_diag"]
    is_v14_active = current_page in ["v14_radar", "v14_diag"]
    is_v24_active = current_page in ["v24_radar", "v24_diag"]
    is_v34_active = current_page in ["v34_radar", "v34_diag"]
    is_v123_active = current_page in ["v123_radar", "v123_diag"]
    is_ortak_active = current_page == "ortak_portfoy"
    is_ma_active = current_page == "ma_review"
    is_veri_active = current_page == "data_mgmt"

    with st.sidebar.expander("🤖 ALFA V1.1.3", expanded=is_v113_active):
        st.radio("v113_opt", options=["📊 Radar & Taramalar", "🔍 Hisse Teşhis Paneli"], key="radio_v113", on_change=create_nav_callback("radio_v113", map_v113), label_visibility="collapsed")

    with st.sidebar.expander("🤖 ALFA V1.2.3", expanded=is_v123_active):
        st.radio("v123_opt", options=["📊 Radar & Taramalar", "🔍 Hisse Teşhis Paneli"], key="radio_v123", on_change=create_nav_callback("radio_v123", map_v123), label_visibility="collapsed")

    with st.sidebar.expander("🤖 ALFA V1.4", expanded=is_v14_active):
        st.radio("v14_opt", options=["📊 Radar & Taramalar", "🔍 Hisse Teşhis Paneli"], key="radio_v14", on_change=create_nav_callback("radio_v14", map_v14), label_visibility="collapsed")

    with st.sidebar.expander("🤖 ALFA V2.4", expanded=is_v24_active):
        st.radio("v24_opt", options=["📊 Radar & Taramalar", "🔍 Hisse Teşhis Paneli"], key="radio_v24", on_change=create_nav_callback("radio_v24", map_v24), label_visibility="collapsed")

    with st.sidebar.expander("🤖 ALFA V3.4", expanded=is_v34_active):
        st.radio("v34_opt", options=["📊 Radar & Taramalar", "🔍 Hisse Teşhis Paneli"], key="radio_v34", on_change=create_nav_callback("radio_v34", map_v34), label_visibility="collapsed")

    with st.sidebar.expander("🎯 Ortak Portföy", expanded=is_ortak_active):
        st.radio("ortak_opt", options=["🎯 Ortak Portföy Tablosu"], key="radio_ortak", on_change=create_nav_callback("radio_ortak", map_ortak), label_visibility="collapsed")

    with st.sidebar.expander("📈 Trend & MA İnceleme", expanded=is_ma_active):
        st.radio("ma_opt", options=["📈 Hareketli Ortalama İnceleme"], key="radio_ma", on_change=create_nav_callback("radio_ma", map_ma), label_visibility="collapsed")

    with st.sidebar.expander("📁 Veri Yönetimi", expanded=is_veri_active):
        st.radio("veri_opt", options=["📁 Veri Yönetimi (Excel Yükle)"], key="radio_veri", on_change=create_nav_callback("radio_veri", map_veri), label_visibility="collapsed")

else:  # aktif_leg == "portfoy"
    st.sidebar.markdown("#### 💼 Portföy Yönetimi")
    st.sidebar.info("İskelet hazır. Alt başlıkları belirlediğinde buraya ekleyeceğiz.")

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size: 12px; color: #6c757d;'><b>Sistem Bilgisi</b><br>• Sürüm: ALFA Multi-Core V2.4<br>• Veri Kaynağı: Fintables & yfinance<br>• Durum: Aktif</div>", unsafe_allow_html=True)

# ==============================================================================
# SAYFA MANTIKLARI
# ==============================================================================

# --- VERİ YÖNETİMİ ---
if current_page == "data_mgmt":
    st.subheader("📁 Fintables Veri Yönetimi ve Excel Yükleme")
    st.markdown("Modelin tarama yapabilmesi için güncel Fintables Excel dosyalarınızı aşağıya yükleyin. Yüklediğiniz dosyalar sayfa yenilemelerinde (F5) **asla silinmez**.")
    
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        file1 = st.file_uploader("1. Temel Analiz & Fiyat Dosyası", type=["xlsx", "xls"], key="f1_up")
    with col_up2:
        file2 = st.file_uploader("2. Eski Dönemler Dosyası (1)", type=["xlsx", "xls"], key="f2_up")

    if file1 and file2:
        try:
            ham1 = pd.read_excel(file1)
            ham2 = pd.read_excel(file2)

            # --- SÜTUN SAYISI GÜVENLİK KONTROLÜ ---
            if ham1.shape[1] < 23:
                st.error(f"❌ 1. dosyada en az 23 sütun bekleniyordu, {ham1.shape[1]} bulundu. Fintables şablonunuzu kontrol edin.")
                st.stop()
            if ham2.shape[1] < 15:
                st.error(f"❌ 2. dosyada (1) en az 15 sütun bekleniyordu, {ham2.shape[1]} bulundu. Yeni büyüme/−4/−5 sütunlarını eklediğinizden emin olun.")
                st.stop()

            df1 = ham1.iloc[:, :23]
            df1.columns = [
                "Kod", "ROE_0", "ROE_1", "ROE_4", "BrutEFK_0", "BrutEFK_1", 
                "EFK_0", "EFK_1", "EFK_4", "FAVOK_0", "FAVOK_1", 
                "NetSatisBuyume", "FAVOKBuyume", "BrutEFKBuyume", "EFKBuyume", 
                "PDDD", "NetBorc_FAVOK", "HAOran", "Getiri_2h", "Getiri_1a", 
                "Getiri_2a", "Getiri_6a", "Kapanis"
            ]

            df2 = ham2.iloc[:, :15]
            df2.columns = [
                "Kod", "ROE_2", "BrutEFK_2", "EFK_2", "FAVOK_2", "ROE_5", "EFK_5",
                "BrutEFK_4", "FAVOK_4", "BrutEFK_5", "FAVOK_5",
                "EFKBuyume_1", "FAVOKBuyume_1", "EFKBuyume_2", "FAVOKBuyume_2"
            ]

            df = pd.merge(df1, df2, on="Kod", how="left").fillna(0)

            # GELDİ/GELMEDİ tespiti + tüm efektif dönem kolonları (V2.4 + V1.4 ortak)
            df = ekle_efektif_kolonlar(df)

            st.session_state.df_merged = df
            now_tr = datetime.datetime.now(TURKEY_TZ)
            st.session_state.upload_time = now_tr.strftime("%d.%m.%Y %H:%M:%S")

            df.to_pickle(DATA_CACHE_PATH)
            with open(META_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump({"upload_time": st.session_state.upload_time}, f)

            st.success(f"✅ Excel dosyaları başarıyla işlendi ve hafızaya kaydedildi! (Yükleme Zamanı: {st.session_state.upload_time}) Sol menüden algoritmalarınıza geçebilirsiniz.")
        except Exception as e:
            st.error(f"Dosyalar işlenirken hata oluştu: {e}")
    else:
        if st.session_state.df_merged is not None:
            st.success(f"📌 Sistemde en son yüklenen veri aktif (Yükleme Zamanı: {st.session_state.upload_time}). Yeni dosya yüklemedikçe bu veri korunacaktır.")
        else:
            st.info("👈 Lütfen her iki Excel dosyasını da yükleyin.")

# --- 💼 PORTFÖY YÖNETİMİ (İSKELET) ---
elif current_page == "portfoy_home":
    st.markdown("### 💼 Portföy Yönetimi")
    st.info("Bu bölüm portföy takibi için hazırlanıyor. Alt başlıkları belirlediğinde (ör. pozisyonlar, "
            "getiri takibi, işlem geçmişi) buraya ekleyeceğiz.")
    st.markdown("**Planlanan iskelet:**")
    st.markdown("- 📌 Açık pozisyonlar ve maliyetler\n- 📈 Portföy getirisi / performans\n- 🧾 İşlem geçmişi\n- ⚖️ Ağırlık / dağılım")

# --- 🎯 ORTAK PORTFÖY (KONSENSÜS) ---
elif current_page == "ortak_portfoy":
    st.markdown("### 🎯 Ortak Portföy (Konsensüs Tablosu)")
    if st.session_state.df_merged is None:
        st.info("👈 Önce 'Veri Yönetimi' sekmesinden Excel dosyalarını yükleyin.")
    else:
        df = st.session_state.df_merged.copy()
        priority = ["V1.4", "V2.4", "V3.4", "V1.1.3", "V1.2.3"]

        # 1) Her algoritmanın temelden geçenleri (yfinance'siz)
        masks = {ad: temel_fn(df) for ad, temel_fn, _, _ in _ALGO_DEFS}
        union_kods = sorted(set().union(*[set(df.loc[m, "Kod"]) for m in masks.values()]))

        if len(union_kods) == 0:
            st.warning("Hiçbir algoritma temel filtresinden hisse geçmedi.")
        else:
            with st.spinner(f"🔄 5 algoritma çalıştırılıyor · {len(union_kods)} hisse için fiyat verisi çekiliyor..."):
                px = _fetch_prices(tuple(union_kods))
                ranked = {}
                for ad, temel_fn, tek_fn, skor_fn in _ALGO_DEFS:
                    ranked[ad] = _sirali(df, masks[ad], px, tek_fn, skor_fn)

            # Yardımcı yapılar
            top5 = {ad: list(ranked[ad]["Kod"].head(5)) for ad in priority}
            top5set = {ad: set(top5[ad]) for ad in priority}
            fullset = {ad: set(ranked[ad]["Kod"]) for ad in priority}
            skormap = {ad: dict(zip(ranked[ad]["Kod"], ranked[ad]["SKOR"])) for ad in priority}

            def yildiz(k):
                return sum(1 for ad in priority if k in top5set[ad])

            def skor_ve_kaynak(k):
                for ad in priority:  # en yüksek öncelikli algoritmanın skoru
                    if k in skormap[ad]:
                        return skormap[ad][k], ad
                return None, None

            # 2) YEŞİL PORTFÖY KURULUŞU
            green = []
            def ekle(k):
                if k not in green:
                    green.append(k)

            # A) 5 algoritmada da ilk-5 ortak → skora göre
            all5 = list(set.intersection(*[top5set[ad] for ad in priority])) if all(len(top5set[ad]) > 0 for ad in priority) else []
            all5.sort(key=lambda k: skor_ve_kaynak(k)[0] or -1e18, reverse=True)
            for k in all5:
                ekle(k)
            # B) V1.4 ilk-5 (sıra), C) V2.4 ilk-5, D) V3.4 ilk-5  → çekirdek
            for ad in ["V1.4", "V2.4", "V3.4"]:
                for k in top5[ad]:
                    ekle(k)
            core_count = len(green)
            # Çekirdek 5'ten azsa yedeklerle (V1.1.3, V1.2.3) 5'e tamamla; 5+ ise dokunma
            if core_count < 5:
                for ad in ["V1.1.3", "V1.2.3"]:
                    for k in top5[ad]:
                        if len(green) >= 5:
                            break
                        ekle(k)
                    if len(green) >= 5:
                        break
                green = green[:5]
            green_set = set(green)

            # 3) Tablo satır sırası: yeşiller (seçim sırası) + griler (yıldız, öncelik ağırlığı, sonra skor)
            W = {ad: (len(priority) - i) for i, ad in enumerate(priority)}  # V1.4=5 ... V1.2.3=1
            def oncelik_agirligi(k):
                return sum(W[ad] for ad in priority if k in top5set[ad])
            all_stocks = set().union(*[fullset[ad] for ad in priority])
            gray = [k for k in all_stocks if k not in green_set]
            gray.sort(key=lambda k: (yildiz(k), oncelik_agirligi(k), skor_ve_kaynak(k)[0] or -1e18), reverse=True)
            ordered = green + gray

            # 4) Görüntü tablosu
            satirlar = []
            for i, k in enumerate(ordered):
                sc, _ = skor_ve_kaynak(k)
                satir = {
                    "#": (i + 1) if k in green_set else "",
                    "Hisse": k,
                    "Skor": round(sc, 1) if sc is not None else None,
                    "⭐": "⭐" * yildiz(k),
                }
                for ad in priority:
                    if k in top5set[ad]:
                        satir[ad] = "✔"       # ilk 5
                    elif k in fullset[ad]:
                        satir[ad] = "✓"       # geçti ama gri (>5)
                    else:
                        satir[ad] = "✕"       # geçmedi
                satirlar.append(satir)
            disp = pd.DataFrame(satirlar, columns=["#", "Hisse", "Skor", "⭐"] + priority)

            # 5) Stil: yeşil satır canlı; matris hücreleri soluk
            def _stil(dfd):
                s = pd.DataFrame("", index=dfd.index, columns=dfd.columns)
                for ridx, k in enumerate(ordered):
                    yesil = k in green_set
                    satir_css = "background-color:#7ed9a6; font-weight:bold; color:#0b3d24;" if yesil \
                                else "background-color:#f2f3f4; color:#95a5a6;"
                    for col in ["#", "Hisse", "Skor", "⭐"]:
                        s.loc[ridx, col] = satir_css
                    for ad in priority:
                        if k in top5set[ad]:
                            s.loc[ridx, ad] = "background-color:#c8e6c9; color:#1b5e20; font-weight:bold; font-size:16px; text-align:center;"   # ilk 5
                        elif k in fullset[ad]:
                            s.loc[ridx, ad] = "background-color:#eafaf1; color:#2e8b57; font-weight:bold; font-size:16px; text-align:center;"   # gri geçen
                        else:
                            s.loc[ridx, ad] = "background-color:#fdecea; color:#c0392b; font-weight:bold; font-size:16px; text-align:center;"   # geçmedi
                return s

            # Özet
            yildizli5 = [k for k in green if yildiz(k) == 5]
            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 Portföy (yeşil)", f"{len(green)} hisse")
            c2.metric("⭐ 5 algoritmada ortak", f"{len(all5)} hisse")
            c3.metric("📋 Tablodaki toplam", f"{len(ordered)} hisse")
            if core_count >= 5:
                st.caption(f"ℹ️ Çekirdek (V1.4+V2.4+V3.4) {core_count} hisse verdi (≥5) → ıraksama senaryosu: üst sınır yok, hepsi yeşil.")
            else:
                st.caption(f"ℹ️ Çekirdek (V1.4+V2.4+V3.4) {core_count} hisse → yedeklerle (V1.1.3, V1.2.3) 5'e tamamlandı.")

            st.dataframe(
                disp.style.apply(_stil, axis=None).format({"Skor": "{:.1f}"}, na_rep="-"),
                use_container_width=True, hide_index=True
            )
            st.markdown(
                """
                <div style="font-size:14px; margin-top:6px; padding:10px; background-color:#f8f9fa; border:1px solid #e9ecef; border-radius:8px;">
                    <b>Simge açıklaması:</b>&nbsp;
                    <span style="color:#1b5e20; font-weight:bold;">✔ İlk 5'te</span> &nbsp;&nbsp;
                    <span style="color:#2e8b57; font-weight:bold;">✓ Geçti (sıra&nbsp;&gt;5)</span> &nbsp;&nbsp;
                    <span style="color:#c0392b; font-weight:bold;">✕ Geçmedi</span>
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <span style="color:#0b3d24;">🟢 Yeşil satır = bu ayki portföy</span><br>
                    <span style="color:#555;">Sütun sırası getiri önceliğine göre: <b>V1.4 &gt; V2.4 &gt; V3.4 &gt; V1.1.3 &gt; V1.2.3</b></span>
                </div>
                """,
                unsafe_allow_html=True
            )

# --- HAREKETLİ ORTALAMA İNCELEME PANELİ ---
elif current_page == "ma_review":
    st.markdown("### 📈 Hareketli Ortalama ve Trend İnceleme Paneli")
    hisse_havuzu = sorted(st.session_state.df_merged["Kod"].dropna().astype(str).str.upper().unique().tolist()) if st.session_state.df_merged is not None else ["A1CAP", "AKBNK", "THYAO", "TUPRS"]
    secilen_hisse = st.selectbox("Hisse Seçin / Arayın:", options=[""] + hisse_havuzu, key="ma_inceleme_select")

    if secilen_hisse:
        try:
            with st.spinner(f"🔄 '{secilen_hisse}' Grafik verileri hazırlanıyor..."):
                h_yf = yf.download(secilen_hisse + ".IS", period="3y", progress=False)
                
            if not h_yf.empty:
                if isinstance(h_yf.columns, pd.MultiIndex): h_yf.columns = h_yf.columns.get_level_values(0)
                h_yf = h_yf.sort_index().dropna()
                
                h_yf['MA20'] = h_yf['Close'].rolling(20).mean()
                h_yf['MA75'] = h_yf['Close'].rolling(75).mean()
                h_yf['MA200'] = h_yf['Close'].rolling(200).mean()
                
                hi_252 = h_yf['High'].tail(252) if len(h_yf) >= 252 else h_yf['High']
                hhv_limit = float(hi_252.max()) * 0.77

                op, hi, lo, cls = h_yf['Open'], h_yf['High'], h_yf['Low'], h_yf['Close']
                if isinstance(op, pd.DataFrame): op = op.iloc[:, 0]
                if isinstance(hi, pd.DataFrame): hi = hi.iloc[:, 0]
                if isinstance(lo, pd.DataFrame): lo = lo.iloc[:, 0]
                if isinstance(cls, pd.DataFrame): cls = cls.iloc[:, 0]

                son_fiyat = float(cls.iloc[-1])
                son_ma20 = float(h_yf['MA20'].iloc[-1]) if not pd.isna(h_yf['MA20'].iloc[-1]) else 0
                son_ma75 = float(h_yf['MA75'].iloc[-1]) if not pd.isna(h_yf['MA75'].iloc[-1]) else 0
                son_ma200 = float(h_yf['MA200'].iloc[-1]) if not pd.isna(h_yf['MA200'].iloc[-1]) else 0

                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=h_yf.index, open=op, high=hi, low=lo, close=cls, name='Fiyat', increasing_line_color='#089981', decreasing_line_color='#F23645'))
                fig.add_trace(go.Scatter(x=h_yf.index, y=h_yf['MA20'], mode='lines', name='MA 20', line=dict(color='#2962FF', width=2)))
                fig.add_trace(go.Scatter(x=h_yf.index, y=h_yf['MA75'], mode='lines', name='MA 75', line=dict(color='#089981', width=2)))
                fig.add_trace(go.Scatter(x=h_yf.index, y=h_yf['MA200'], mode='lines', name='MA 200', line=dict(color='#2A2E39', width=2)))
                
                last_date = h_yf.index[-1]
                def add_badge(f_obj, y_val, text, bg):
                    f_obj.add_annotation(x=last_date, y=y_val, text=f" <b>{text}</b> ", showarrow=False, xref="x", yref="y", xanchor="left", yanchor="middle", bgcolor=bg, font=dict(color="white", size=11), borderpad=3)

                add_badge(fig, son_fiyat, f"{son_fiyat:.2f}", "#089981" if cls.iloc[-1] >= op.iloc[-1] else "#F23645")
                if son_ma20: add_badge(fig, son_ma20, f"{son_ma20:.2f}", "#2962FF")
                if son_ma75: add_badge(fig, son_ma75, f"{son_ma75:.2f}", "#089981")
                if son_ma200: add_badge(fig, son_ma200, f"{son_ma200:.2f}", "#2A2E39")

                max_dt, min_dt = h_yf.index[-1], h_yf.index[0]
                dt_1m, dt_3m, dt_6m, dt_1y, dt_3y = [(max_dt - pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in [30, 90, 180, 365, 1095]]

                fig.update_layout(
                    template="plotly_white", height=650, margin=dict(l=10, r=85, t=60, b=10), xaxis_rangeslider_visible=False, showlegend=False, hovermode="x unified",
                    updatemenus=[dict(type="buttons", direction="right", active=3, x=0.01, y=1.08, xanchor="left", yanchor="bottom", bgcolor="#F8FAFC", bordercolor="#CBD5E1", buttons=[
                        dict(label="1A", method="relayout", args=[{"xaxis.range": [dt_1m, max_dt.strftime("%Y-%m-%d")], "yaxis.autorange": True}]),
                        dict(label="3A", method="relayout", args=[{"xaxis.range": [dt_3m, max_dt.strftime("%Y-%m-%d")], "yaxis.autorange": True}]),
                        dict(label="6A", method="relayout", args=[{"xaxis.range": [dt_6m, max_dt.strftime("%Y-%m-%d")], "yaxis.autorange": True}]),
                        dict(label="1Y", method="relayout", args=[{"xaxis.range": [dt_1y, max_dt.strftime("%Y-%m-%d")], "yaxis.autorange": True}]),
                        dict(label="3Y", method="relayout", args=[{"xaxis.range": [dt_3y, max_dt.strftime("%Y-%m-%d")], "yaxis.autorange": True}]),
                        dict(label="Tümü", method="relayout", args=[{"xaxis.range": [min_dt.strftime("%Y-%m-%d"), max_dt.strftime("%Y-%m-%d")], "yaxis.autorange": True}]),
                    ])]
                )
                fig.update_xaxes(gridcolor="#F0F0F0", range=[dt_1y, max_dt.strftime("%Y-%m-%d")])
                fig.update_yaxes(side="right", tickformat=".2f", gridcolor="#F0F0F0", autorange=True, fixedrange=False)
                st.plotly_chart(fig, use_container_width=True)
            else: st.warning("Veri çekilemedi.")
        except Exception as e: st.error(f"Hata: {e}")

# --- 🤖 ALFA V1.1.3 SİSTEMİ ---
elif current_page in ["v113_radar", "v113_diag"]:
    if st.session_state.df_merged is not None:
        df = st.session_state.df_merged.copy()

        # V1.1.3 TEMEL FİLTRELERİ  (en sade varyant; efektif kolon KULLANMAZ; endeks = XU100)
        # NOT: Spec'te pddLimit "IF(b>90, 8+(b-90)/0.07, 8)" yazıyordu (b boolean + /0.07).
        # Diğer 4 algoritmayla tutarlı olması için standart forma çevirdim: IF(ROE>90, 8+(ROE-90)*0.07, 8).
        df["pdddLimit"] = np.where(df["ROE_0"] > 90, 8 + (df["ROE_0"] - 90) * 0.07, 8)

        a = df["Getiri_2a"] > oto_2a                                  # XU100
        b = df["ROE_0"] > tufe_12
        c = df["NetBorc_FAVOK"] < 4
        d = df["PDDD"] < df["pdddLimit"]
        e = df["Getiri_2h"] > (oto_2h - 10)                           # XU100
        f = df["NetSatisBuyume"] > 0
        g = df["FAVOKBuyume"] > tufe_12
        h = (df["BrutEFKBuyume"] > tufe_12) & (df["EFKBuyume"] > tufe_12)
        j = df["Getiri_1a"] > -15
        k = df["HAOran"] < 60

        temel_filtreliler_v113 = a & b & c & d & e & f & (h | g) & j & k
        df_temel = df[temel_filtreliler_v113].copy()

        # RADAR VE TARAMALAR
        if current_page == "v113_radar":
            st.markdown("### 🤖 ALFA V1.1.3 - Radar & Taramalar")
            teknik_asanadan_gecenler = []
            analiz_edilen = 0

            if len(df_temel) > 0:
                with st.spinner("🔄 ALFA V1.1.3 için MA75, MA200 verileri anlık çekiliyor..."):
                    for idx, row in df_temel.iterrows():
                        kod = str(row["Kod"]).strip() + ".IS"
                        try:
                            hist = yf.download(kod, period="1y", progress=False)
                            if not hist.empty:
                                if isinstance(hist.columns, pd.MultiIndex):
                                    hist.columns = hist.columns.get_level_values(0)
                                hist = hist.sort_index().dropna()
                                close = hist['Close']
                                if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
                                if len(close) >= 200:
                                    analiz_edilen += 1
                                    ma75 = float(close.rolling(75).mean().iloc[-1])
                                    ma200 = float(close.rolling(200).mean().iloc[-1])
                                    df_temel.loc[idx, 'MA75'] = round(ma75, 2)
                                    df_temel.loc[idx, 'MA200'] = round(ma200, 2)

                                    # V1.1.3 TEKNİK ŞARTI: MA75 > MA200
                                    if ma75 > ma200:
                                        teknik_asanadan_gecenler.append(idx)
                        except:
                            continue

            df_teknik = df_temel.loc[teknik_asanadan_gecenler].copy()

            m1, m2, m3 = st.columns(3)
            m1.metric(label="📊 Toplam İncelenen", value=f"{len(df)} Hisse")
            m2.metric(label="🏛️ Temel Filtreyi Geçen", value=f"{len(df_temel)} Hisse")
            m3.metric(label="🏆 Teknik & Trendi Geçen (Adaylar)", value=f"{len(df_teknik)} Hisse")
            if len(df_temel) > 0:
                st.caption(f"ℹ️ Temeli geçen {len(df_temel)} adaydan **{analiz_edilen}** tanesi için ≥200 günlük fiyat verisi çekilebildi. "
                           f"{'Hepsi çekildi.' if analiz_edilen == len(df_temel) else 'Çekilemeyen adaylar yfinance verisi/geçmişi eksik olabilir.'}")
            st.markdown("---")

            if len(df_temel) > 0:
                if len(df_teknik) > 0:
                    roe = df_teknik["ROE_0"]
                    m6 = df_teknik["Getiri_6a"]
                    m2 = df_teknik["Getiri_2a"]
                    m1 = df_teknik["Getiri_1a"]
                    c_fiyat = df_teknik["Kapanis"]
                    ma75_val = df_teknik["MA75"]
                    eb = df_teknik["EFKBuyume"]
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

                    # V1.1.3 FORMÜLÜ (V2.4 ile aynı: cezalı)
                    df_teknik["SKOR"] = (
                        (roeSkor * 0.30) + (buySkor * 0.20) + (momentumSkor * 0.15) +
                        (m2Skor * 0.05) + (trendSkor * 0.18) + (pdddSkor * 0.12) + negCeza + ardisikCeza
                    )

                    df_sonuc = df_teknik.sort_values(by="SKOR", ascending=False).reset_index(drop=True)
                    df_sonuc.index += 1

                    st.markdown("### 🏆 ALFA V1.1.3 Portföy Adayları (En İyi Skorlar)")
                    def highlight_top5(s):
                        return ['background-color: #d4edda; font-weight: bold;' if s.name <= 5 else '' for _ in s]

                    st.dataframe(
                        df_sonuc[["Kod", "SKOR", "Bilanco_Durum", "ROE_0", "PDDD", "Getiri_1a", "Getiri_6a"]]
                        .head(15)
                        .style.format(precision=2, subset=["SKOR", "ROE_0", "PDDD", "Getiri_1a", "Getiri_6a"])
                        .apply(highlight_top5, axis=1),
                        use_container_width=True
                    )
                else: st.warning("ALFA V1.1.3 teknik kriterlerini sağlayan hisse bulunamadı.")
            else: st.warning("ALFA V1.1.3 temel kriterlerini sağlayan hisse bulunamadı.")

        # TEŞHİS PANELİ
        elif current_page == "v113_diag":
            st.markdown("### 🔍 ALFA V1.1.3 - Hisse Teşhis Paneli")
            hisse_listesi = [""] + sorted(df["Kod"].dropna().astype(str).str.upper().unique().tolist())
            secilen_hisse = st.selectbox("Hisse Seçin / Arayın:", options=hisse_listesi)

            if secilen_hisse:
                tek_hisse_df = df[df["Kod"].str.upper() == secilen_hisse]
                if not tek_hisse_df.empty:
                    th = tek_hisse_df.iloc[0]
                    p_lim = float(8 + (th['ROE_0'] - 90) * 0.07 if th['ROE_0'] > 90 else 8)

                    c_a = bool(th['Getiri_2a'] > oto_2a)
                    c_b = bool(th['ROE_0'] > tufe_12)
                    c_c = bool(th['NetBorc_FAVOK'] < 4)
                    c_d = bool(th['PDDD'] < p_lim)
                    c_e = bool(th['Getiri_2h'] > (oto_2h - 10))
                    c_f = bool(th['NetSatisBuyume'] > 0)
                    c_g = bool(th['FAVOKBuyume'] > tufe_12)
                    c_h = bool((th['BrutEFKBuyume'] > tufe_12) and (th['EFKBuyume'] > tufe_12))
                    c_j = bool(th['Getiri_1a'] > -15)
                    c_k = bool(th['HAOran'] < 60)

                    takilan = []
                    krit = [
                        ("2A Getiri > XU100", c_a, f"%{th['Getiri_2a']:.2f} vs XU100 %{oto_2a:.2f}"),
                        ("ROE > TÜFE", c_b, f"ROE %{th['ROE_0']:.2f} vs TÜFE %{tufe_12:.2f}"),
                        ("Net Borç/FAVÖK < 4", c_c, f"{dg(th['NetBorc_FAVOK'])}"),
                        ("PD/DD < Sınır", c_d, f"PDDD {dg(th['PDDD'])} vs sınır {dg(p_lim)}"),
                        ("2H Getiri Şartı", c_e, f"%{th['Getiri_2h']:.2f} vs eşik %{(oto_2h - 10):.2f}"),
                        ("Net Satış Büyümesi > 0", c_f, f"%{th['NetSatisBuyume']:.2f}"),
                        ("Büyüme (FAVÖK↑ veya BrütEFK↑&EFK↑)", (c_h or c_g), f"FAVÖKBüy %{th['FAVOKBuyume']:.1f} · EFKBüy %{th['EFKBuyume']:.1f} · BrütBüy %{th['BrutEFKBuyume']:.1f}"),
                        ("1A Getiri > -15", c_j, f"%{th['Getiri_1a']:.2f}"),
                        ("Halka Açıklık < 60", c_k, f"{dg(th['HAOran'])}"),
                    ]
                    takilan = [l for l, ok, v in krit if not ok]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"#### 🏛️ ALFA V1.1.3 Temel Analiz ({secilen_hisse})")
                        for l, ok, v in krit:
                            st.write(f"- {l}: {'✅ Geçti' if ok else '❌ Kaldı'}")

                    with col2:
                        st.markdown(f"#### 📈 ALFA V1.1.3 Teknik Şartlar")
                        try:
                            h_yf = yf.download(secilen_hisse + ".IS", period="1y", progress=False)
                            if not h_yf.empty:
                                if isinstance(h_yf.columns, pd.MultiIndex): h_yf.columns = h_yf.columns.get_level_values(0)
                                h_yf = h_yf.sort_index().dropna()
                                cls = h_yf['Close']
                                if isinstance(cls, pd.DataFrame): cls = cls.iloc[:, 0]
                                if len(cls) >= 200:
                                    m75 = float(cls.rolling(75).mean().iloc[-1])
                                    m200 = float(cls.rolling(200).mean().iloc[-1])
                                    t1 = m75 > m200
                                    st.write(f"- **MA75 > MA200:** {'✅' if t1 else '❌'} ({m75:.2f} > {m200:.2f})")
                                    if not t1:
                                        takilan.append("ALFA V1.1.3 Teknik Şartı (MA75 > MA200)")
                        except Exception as ex: st.error(f"Teknik hata: {ex}")

                    if takilan:
                        st.markdown("---")
                        st.warning("Hisse aşağıdaki kriterleri sağlamadığı için filtrelerden geçemedi:")
                        for kr in takilan: st.write(f"• {kr}")
                    else: st.success("🎉 Tebrikler! Hisse V1.1.3 filtrelerinin hepsinden başarıyla geçti.")
    else:
        st.info("👈 Önce 'Veri Yönetimi' sekmesinden Excel dosyalarını yükleyin.")

# --- 🤖 ALFA V1.4 SİSTEMİ (YENİ EKLENEN MODEL) ---
elif current_page in ["v14_radar", "v14_diag"]:
    if st.session_state.df_merged is not None:
        df = st.session_state.df_merged.copy()

        # --- YENİ KOLON GÜVENCESİ (bayat cache'e karşı) ---
        # V1.4 yeni efektif kolonlara ihtiyaç duyar. Bunlar yoksa ama ham -4/-5/büyüme
        # sütunları varsa yeniden hesapla; ham veri de yoksa yeniden yükleme iste.
        _gerekli = ["ef_FAVOK_4", "ef_BrutEFK_4", "ef_EFKBuyume_1", "ef_FAVOKBuyume_1"]
        if not all(c in df.columns for c in _gerekli):
            _baz = ["FAVOK_4", "FAVOK_5", "BrutEFK_4", "BrutEFK_5",
                    "EFKBuyume_1", "EFKBuyume_2", "FAVOKBuyume_1", "FAVOKBuyume_2"]
            if all(c in df.columns for c in _baz):
                df = ekle_efektif_kolonlar(df)
            else:
                st.warning("⚠️ ALFA V1.4 için güncel **15 sütunlu file(1)** gerekiyor. "
                           "Lütfen 'Veri Yönetimi' sekmesinden yeni Fintables dosyalarını yükleyin.")
                st.stop()

        # V1.4 TEMEL FİLTRELERİ  (tüm -1/-4 karşılaştırmaları EFEKTİF kolonlarla; endeks = XUTUM)
        df["pdddLimit"] = np.where(df["ROE_0"] > 90, 8 + (df["ROE_0"] - 90) * 0.07, 8)

        a = df["ROE_0"] > tufe_12
        b = df["NetBorc_FAVOK"] < 4
        d = df["PDDD"] < df["pdddLimit"]
        ee = df["Getiri_2h"] > (oto_xutum_2h - 10)                     # XUTUM
        f = df["NetSatisBuyume"] > 0
        gx = (df["FAVOKBuyume"] > tufe_12) | \
             ((df["FAVOK_0"] > df["ef_FAVOK_1"]) & (df["FAVOK_0"] > df["ef_FAVOK_4"]))
        hx = ((df["BrutEFKBuyume"] > tufe_12) |
              ((df["BrutEFK_0"] > df["ef_BrutEFK_1"]) & (df["BrutEFK_0"] > df["ef_BrutEFK_4"]))) & \
             ((df["EFKBuyume"] > tufe_12) |
              ((df["EFK_0"] > df["ef_EFK_1"]) & (df["EFK_0"] > df["ef_EFK_4"])))
        h = df["Getiri_2a"] > oto_xutum_2a                            # XUTUM
        j = df["Getiri_1a"] > -15
        efkTeyit = (df["EFK_0"] >= df["ef_EFK_1"]) | (df["EFK_0"] >= df["ef_EFK_4"])
        roeTeyit = (df["ROE_0"] >= df["ef_ROE_1"]) | (df["ROE_0"] >= df["ef_ROE_4"])
        k = df["HAOran"] < 60

        temel_filtreliler_v14 = a & b & d & ee & f & (hx | gx | efkTeyit) & h & j & roeTeyit & k
        df_temel = df[temel_filtreliler_v14].copy()

        # RADAR VE TARAMALAR
        if current_page == "v14_radar":
            st.markdown("### 🤖 ALFA V1.4 - Radar & Taramalar")
            teknik_asanadan_gecenler = []
            analiz_edilen = 0
            
            if len(df_temel) > 0:
                with st.spinner("🔄 ALFA V1.4 için MA20, MA60, MA75, MA200 ve HHV252 verileri kontrol ediliyor..."):
                    for idx, row in df_temel.iterrows():
                        kod = str(row["Kod"]).strip() + ".IS"
                        try:
                            hist = yf.download(kod, period="1y", progress=False)
                            if not hist.empty:
                                if isinstance(hist.columns, pd.MultiIndex):
                                    hist.columns = hist.columns.get_level_values(0)
                                hist = hist.sort_index().dropna()
                                close = hist['Close']
                                if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
                                if len(close) >= 200:
                                    analiz_edilen += 1
                                    c_val = float(close.iloc[-1])
                                    ma20 = float(close.rolling(20).mean().iloc[-1])
                                    ma60 = float(close.rolling(60).mean().iloc[-1])
                                    ma75 = float(close.rolling(75).mean().iloc[-1])
                                    ma200 = float(close.rolling(200).mean().iloc[-1])
                                    
                                    high_s = hist['High']
                                    if isinstance(high_s, pd.DataFrame): high_s = high_s.iloc[:, 0]
                                    hi_252 = high_s.tail(252) if len(high_s) >= 252 else high_s
                                    hhv_limit = float(hi_252.max()) * 0.77

                                    df_temel.loc[idx, 'Kapanis_Anlik'] = round(c_val, 2)
                                    df_temel.loc[idx, 'MA20'] = round(ma20, 2)
                                    df_temel.loc[idx, 'MA60'] = round(ma60, 2)
                                    df_temel.loc[idx, 'MA75'] = round(ma75, 2)
                                    df_temel.loc[idx, 'MA200'] = round(ma200, 2)
                                    df_temel.loc[idx, 'HHV_Limit'] = round(hhv_limit, 2)

                                    # V1.4 TEKNİK ŞARTI: C > MA200 & C > MA75 & MA20 > MA60 & C > HHV(252)*0.77
                                    if (c_val > ma200) and (c_val > ma75) and (ma20 > ma60) and (c_val > hhv_limit):
                                        teknik_asanadan_gecenler.append(idx)
                        except:
                            continue

            df_teknik = df_temel.loc[teknik_asanadan_gecenler].copy()

            m1, m2, m3 = st.columns(3)
            m1.metric(label="📊 Toplam İncelenen", value=f"{len(df)} Hisse")
            m2.metric(label="🏛️ Temel Filtreyi Geçen", value=f"{len(df_temel)} Hisse")
            m3.metric(label="🏆 Teknik & Trendi Geçen (Adaylar)", value=f"{len(df_teknik)} Hisse")
            if len(df_temel) > 0:
                st.caption(f"ℹ️ Temeli geçen {len(df_temel)} adaydan **{analiz_edilen}** tanesi için ≥200 günlük fiyat verisi çekilebildi (teknik değerlendirmeye giren). "
                           f"{'Hepsi çekildi.' if analiz_edilen == len(df_temel) else 'Çekilemeyen adaylar yfinance verisi/geçmişi eksik olabilir.'}")
            st.markdown("---")

            if len(df_temel) > 0:
                if len(df_teknik) > 0:
                    roe = df_teknik["ROE_0"]
                    m6 = df_teknik["Getiri_6a"]
                    m2 = df_teknik["Getiri_2a"]
                    c_fiyat = df_teknik["Kapanis_Anlik"]
                    ma75_val = df_teknik["MA75"]
                    ma200_val = df_teknik["MA200"]
                    eb = df_teknik["EFKBuyume"]
                    fb = df_teknik["FAVOKBuyume"]
                    sb = df_teknik["NetSatisBuyume"]
                    beb = df_teknik["BrutEFKBuyume"]
                    pddd = df_teknik["PDDD"]
                    ef_eb1 = df_teknik["ef_EFKBuyume_1"]   # efektif -1 (GELMEDİ ise -2) EFK büyümesi
                    ef_fb1 = df_teknik["ef_FAVOKBuyume_1"] # efektif -1 (GELMEDİ ise -2) FAVÖK büyümesi

                    roeSkor = np.where(roe > 50, 100, roe * 2)
                    momentumSkor = np.where(m6 > 100, 100, m6)
                    m2Skor = np.where(m2 > 50, 100, m2 * 2)
                    trendYuzdesi = ((c_fiyat - ma75_val) / ma75_val) * 100
                    trendSkor = np.where(trendYuzdesi > 30, 30, np.where(trendYuzdesi < 0, 0, trendYuzdesi))

                    ebSkor = np.where(eb > 50, 100, np.where(eb > 0, eb * 2, 0))
                    fbSkor = np.where(fb > 50, 100, np.where(fb > 0, fb * 2, 0))
                    sbSkor = np.where(sb > 50, 100, np.where(sb > 0, sb * 2, 0))
                    buySkor = (ebSkor + fbSkor + sbSkor) / 3

                    # V1.4 ÖZEL BONUS & CEZALAR  (spec: yıllık büyüme ivmesi, efektif -1 ile)
                    # ivmeBonus = IF(EFKBüy(0)>EFKBüy(-1) ve EFKBüy(0)>0, 15, IF(EFKBüy(0)>EFKBüy(-1), 5, 0))
                    ivmeBonus = np.where((eb > ef_eb1) & (eb > 0), 15,
                                         np.where(eb > ef_eb1, 5, 0))
                    # favokIvme = IF(FAVOKBüy(0)>FAVOKBüy(-1) ve FAVOKBüy(0)>0, 10, 0)
                    favokIvme = np.where((fb > ef_fb1) & (fb > 0), 10, 0)
                    maUzaklik = ((c_fiyat - ma200_val) / ma200_val) * 100
                    maUzakCeza = np.where(maUzaklik > 80, -15, np.where(maUzaklik > 60, -8, 0))
                    pdddSkor = np.where(pddd < 1.5, 25, np.where(pddd < 3, 15, np.where(pddd < 5, 5, np.where(pddd > 6, -10, 0))))

                    buyumeBonus = np.where(
                        (sb > 0) & (((beb > tufe_12) & (eb > tufe_12)) | (fb > tufe_12)), 15,
                        np.where((sb > 0) | (beb > tufe_12) | (eb > tufe_12) | (fb > tufe_12), 0, -10)
                    )

                    # V1.4 FORMÜLÜ
                    df_teknik["SKOR"] = (
                        (roeSkor * 0.30) + (buySkor * 0.20) + (momentumSkor * 0.15) + 
                        (trendSkor * 0.18) + (m2Skor * 0.05) + (pdddSkor * 0.12) + 
                        (ivmeBonus * 0.90) + favokIvme + maUzakCeza + buyumeBonus
                    )

                    df_sonuc = df_teknik.sort_values(by="SKOR", ascending=False).reset_index(drop=True)
                    df_sonuc.index += 1

                    st.markdown("### 🏆 ALFA V1.4 Portföy Adayları (En İyi Skorlar)")
                    def highlight_top5(s):
                        return ['background-color: #d4edda; font-weight: bold;' if s.name <= 5 else '' for _ in s]

                    st.dataframe(
                        df_sonuc[["Kod", "SKOR", "Bilanco_Durum", "ROE_0", "PDDD", "Getiri_1a", "Getiri_6a"]]
                        .head(15)
                        .style.format(precision=2, subset=["SKOR", "ROE_0", "PDDD", "Getiri_1a", "Getiri_6a"])
                        .apply(highlight_top5, axis=1),
                        use_container_width=True
                    )
                else: st.warning("ALFA V1.4 teknik kriterlerini sağlayan hisse bulunamadı.")
            else: st.warning("ALFA V1.4 temel kriterlerini sağlayan hisse bulunamadı.")

        # TEŞHİS PANELİ
        elif current_page == "v14_diag":
            st.markdown("### 🔍 ALFA V1.4 - Hisse Teşhis Paneli")
            hisse_listesi = [""] + sorted(df["Kod"].dropna().astype(str).str.upper().unique().tolist())
            secilen_hisse = st.selectbox("Hisse Seçin / Arayın:", options=hisse_listesi)

            if secilen_hisse:
                tek_hisse_df = df[df["Kod"].str.upper() == secilen_hisse]
                if not tek_hisse_df.empty:
                    th = tek_hisse_df.iloc[0]
                    p_lim = float(8 + (th['ROE_0'] - 90) * 0.07 if th['ROE_0'] > 90 else 8)
                    
                    c_a = bool(th['ROE_0'] > tufe_12)
                    c_b = bool(th['NetBorc_FAVOK'] < 4)
                    c_d = bool(th['PDDD'] < p_lim)
                    c_ee = bool(th['Getiri_2h'] > (oto_xutum_2h - 10))
                    c_f = bool(th['NetSatisBuyume'] > 0)
                    c_gx = bool((th['FAVOKBuyume'] > tufe_12) or ((th['FAVOK_0'] > th['ef_FAVOK_1']) and (th['FAVOK_0'] > th['ef_FAVOK_4'])))
                    c_hx = bool(((th['BrutEFKBuyume'] > tufe_12) or ((th['BrutEFK_0'] > th['ef_BrutEFK_1']) and (th['BrutEFK_0'] > th['ef_BrutEFK_4']))) and ((th['EFKBuyume'] > tufe_12) or ((th['EFK_0'] > th['ef_EFK_1']) and (th['EFK_0'] > th['ef_EFK_4']))))
                    c_h = bool(th['Getiri_2a'] > oto_xutum_2a)
                    c_j = bool(th['Getiri_1a'] > -15)
                    c_teyit = bool((th['EFK_0'] >= th['ef_EFK_1']) or (th['EFK_0'] >= th['ef_EFK_4']))
                    c_roe_t = bool((th['ROE_0'] >= th['ef_ROE_1']) or (th['ROE_0'] >= th['ef_ROE_4']))
                    c_k = bool(th['HAOran'] < 60)

                    takilan = []
                    krit = [
                        ("ROE > TÜFE", c_a, f"ROE %{th['ROE_0']:.2f} vs TÜFE %{tufe_12:.2f}"),
                        ("Net Borç/FAVÖK < 4", c_b, f"{dg(th['NetBorc_FAVOK'])}"),
                        ("PD/DD < Sınır", c_d, f"PDDD {dg(th['PDDD'])} vs sınır {dg(p_lim)}"),
                        ("2H Getiri Şartı", c_ee, f"%{th['Getiri_2h']:.2f} vs eşik %{(oto_xutum_2h - 10):.2f}"),
                        ("Net Satış Büyümesi > 0", c_f, f"%{th['NetSatisBuyume']:.2f}"),
                        ("Büyüme/Teyit", (c_hx or c_gx or c_teyit), f"FAVÖKBüy %{th['FAVOKBuyume']:.1f} · EFKBüy %{th['EFKBuyume']:.1f} · BrütBüy %{th['BrutEFKBuyume']:.1f}"),
                        ("2A Getiri > XUTUM", c_h, f"%{th['Getiri_2a']:.2f} vs XUTUM %{oto_xutum_2a:.2f}"),
                        ("1A Getiri > -15", c_j, f"%{th['Getiri_1a']:.2f}"),
                        ("ROE Teyit", c_roe_t, f"ROE %{th['ROE_0']:.2f} vs efROE-1 %{th['ef_ROE_1']:.2f} / efROE-4 %{th['ef_ROE_4']:.2f}"),
                        ("Halka Açıklık < 60", c_k, f"{dg(th['HAOran'])}"),
                    ]
                    takilan = [l for l, ok, v in krit if not ok]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"#### 🏛️ ALFA V1.4 Temel Analiz ({secilen_hisse})")
                        for l, ok, v in krit:
                            st.write(f"- {l}: {'✅ Geçti' if ok else '❌ Kaldı'}")

                    with col2:
                        st.markdown(f"#### 📈 ALFA V1.4 Teknik Şartlar")
                        try:
                            h_yf = yf.download(secilen_hisse + ".IS", period="1y", progress=False)
                            if not h_yf.empty:
                                if isinstance(h_yf.columns, pd.MultiIndex): h_yf.columns = h_yf.columns.get_level_values(0)
                                h_yf = h_yf.sort_index().dropna()
                                cls = h_yf['Close']
                                if isinstance(cls, pd.DataFrame): cls = cls.iloc[:, 0]
                                if len(cls) >= 200:
                                    c_val = float(cls.iloc[-1])
                                    m20 = float(cls.rolling(20).mean().iloc[-1])
                                    m60 = float(cls.rolling(60).mean().iloc[-1])
                                    m75 = float(cls.rolling(75).mean().iloc[-1])
                                    m200 = float(cls.rolling(200).mean().iloc[-1])
                                    hi_252 = h_yf['High'].tail(252) if len(h_yf) >= 252 else h_yf['High']
                                    hhv_limit = float(hi_252.max()) * 0.77

                                    t1 = c_val > m200
                                    t2 = c_val > m75
                                    t3 = m20 > m60
                                    t4 = c_val > hhv_limit

                                    st.write(f"- **Kapanış > MA200:** {'✅' if t1 else '❌'} ({c_val:.2f} > {m200:.2f})")
                                    st.write(f"- **Kapanış > MA75:** {'✅' if t2 else '❌'} ({c_val:.2f} > {m75:.2f})")
                                    st.write(f"- **MA20 > MA60:** {'✅' if t3 else '❌'} ({m20:.2f} > {m60:.2f})")
                                    st.write(f"- **Fiyat > HHV(252)*0.77:** {'✅' if t4 else '❌'} ({c_val:.2f} > {hhv_limit:.2f})")
                                    
                                    if not (t1 and t2 and t3 and t4):
                                        takilan.append("ALFA V1.4 Teknik Şartı (C>MA200, C>MA75, MA20>MA60, C>HHV*0.77)")
                        except Exception as e: st.error(f"Teknik hata: {e}")

                    if takilan:
                        st.markdown("---")
                        st.warning("Hisse aşağıdaki kriterleri sağlamadığı için filtrelerden geçemedi:")
                        for kr in takilan: st.write(f"• {kr}")
                    else: st.success("🎉 Tebrikler! Hisse V1.4 filtrelerinin hepsinden başarıyla geçti.")

# --- 🤖 ALFA V2.4 SİSTEMİ (MEVCUT MODEL) ---
elif current_page in ["v24_radar", "v24_diag"]:
    if st.session_state.df_merged is not None:
        df = st.session_state.df_merged

        df["pdddLimit"] = np.where(df["ROE_0"] > 90, 8 + (df["ROE_0"] - 90) * 0.07, 8)
        
        a = df["Getiri_2a"] > oto_2a
        b = df["ROE_0"] > tufe_12
        c = df["NetBorc_FAVOK"] < 4
        d = df["PDDD"] < df["pdddLimit"]
        ee = df["Getiri_2h"] > (oto_2h - 10)
        f = df["NetSatisBuyume"] > 0
        g = df["FAVOKBuyume"] > tufe_12
        h = (df["BrutEFKBuyume"] > tufe_12) & (df["EFKBuyume"] > tufe_12)
        # DÜZELTME: Excel'e sadık -> efektif (bilanço-ayarlı) dönemleri kullan
        gx = df["FAVOK_0"] > df["ef_FAVOK_1"]
        hx = (df["BrutEFK_0"] > df["ef_BrutEFK_1"]) & (df["EFK_0"] > df["ef_EFK_1"])
        j = df["Getiri_1a"] > -15
        efkTeyit = (df["EFK_0"] >= df["ef_EFK_1"]) | (df["EFK_0"] >= df["ef_EFK_4"])
        roeTeyit = (df["ROE_0"] >= df["ef_ROE_1"]) | (df["ROE_0"] >= df["ef_ROE_4"])
        k = df["HAOran"] < 60

        temel_filtreliler = a & b & c & d & ee & f & ((h | g) | (hx | gx) | efkTeyit) & j & roeTeyit & k
        df_temel = df[temel_filtreliler].copy()

        if current_page == "v24_radar":
            st.markdown("### 🤖 ALFA V2.4 - Radar & Taramalar")
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
                                    if ma75 > ma200 and ma20 > ma75: teknik_asanadan_gecenler.append(idx)
                        except: continue

            df_teknik = df_temel.loc[teknik_asanadan_gecenler].copy()

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
                    eb = df_teknik["EFKBuyume"]   # DÜZELTME: Excel ebSkor EFK büyümesini kullanır (BrutEFK değil)
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

                    st.markdown("### 🏆 ALFA V2.4 Portföy Adayları (En İyi Skorlar)")
                    def highlight_top5(s):
                        return ['background-color: #d4edda; font-weight: bold;' if s.name <= 5 else '' for _ in s]

                    st.dataframe(
                        df_sonuc[["Kod", "SKOR", "Bilanco_Durum", "ROE_0", "PDDD", "Getiri_1a", "Getiri_6a"]]
                        .head(15)
                        .style.format(precision=2, subset=["SKOR", "ROE_0", "PDDD", "Getiri_1a", "Getiri_6a"])
                        .apply(highlight_top5, axis=1),
                        use_container_width=True
                    )
                else: st.warning("Teknik kriterleri sağlayan hisse bulunamadı.")
            else: st.warning("Temel kriterleri sağlayan hisse bulunamadı.")

        elif current_page == "v24_diag":
            st.markdown("### 🔍 ALFA V2.4 - Hisse Teşhis Paneli")
            hisse_listesi = [""] + sorted(df["Kod"].dropna().astype(str).str.upper().unique().tolist())
            secilen_hisse = st.selectbox("Hisse Seçin / Arayın:", options=hisse_listesi)

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
                    
                    takilan = []
                    krit = [
                        ("ROE > TÜFE", c_b, f"ROE %{th['ROE_0']:.2f} vs TÜFE %{tufe_12:.2f}"),
                        ("2A Getiri > XU100", c_a, f"%{th['Getiri_2a']:.2f} vs XU100 %{oto_2a:.2f}"),
                        ("2H Getiri Şartı", c_ee, f"%{th['Getiri_2h']:.2f} vs eşik %{(oto_2h - 10):.2f}"),
                        ("PD/DD < Sınır", c_d, f"PDDD {dg(th['PDDD'])} vs sınır {dg(p_lim)}"),
                        ("Net Borç/FAVÖK < 4", c_c, f"{dg(th['NetBorc_FAVOK'])}"),
                        ("Net Satış Büyümesi > 0", c_f, f"%{th['NetSatisBuyume']:.2f}"),
                        ("Büyüme/Teyit", (c_g or c_h or c_teyit), f"FAVÖKBüy %{th['FAVOKBuyume']:.1f} · EFKBüy %{th['EFKBuyume']:.1f} · BrütBüy %{th['BrutEFKBuyume']:.1f}"),
                        ("ROE Teyit", c_roe_t, f"ROE %{th['ROE_0']:.2f} vs efROE-1 %{th['ef_ROE_1']:.2f} / efROE-4 %{th['ef_ROE_4']:.2f}"),
                        ("1A Getiri > -15", c_j, f"%{th['Getiri_1a']:.2f}"),
                        ("Halka Açıklık < 60", c_k, f"{dg(th['HAOran'])}"),
                    ]
                    takilan = [l for l, ok, v in krit if not ok]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"#### 🏛️ Temel Analiz ({secilen_hisse})")
                        for l, ok, v in krit:
                            st.write(f"- {l}: {'✅ Geçti' if ok else '❌ Kaldı'}")

                    with col2:
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
                                    if not teknik_gecti: takilan.append("Teknik MA Kuralı (MA75 > MA200 ve MA20 > MA75)")

                                    st.write(f"- **Kapanış:** {float(th['Kapanis']):.2f}")
                                    st.write(f"- **MA20:** {m20:.2f}")
                                    st.write(f"- **MA75:** {m75:.2f}")
                                    st.write(f"- **MA200:** {m200:.2f}")
                                    st.write(f"- **Teknik Kural:** {'✅ Sağlıyor' if teknik_gecti else '❌ Sağlamıyor'}")
                        except Exception as e: st.error(f"Teknik hata: {e}")

                    if takilan:
                        st.markdown("---")
                        st.warning("Hisse aşağıdaki kriterleri sağlamadığı için filtrelerden geçemedi:")
                        for kr in takilan: st.write(f"• {kr}")
                    else: st.success("🎉 Tebrikler! Hisse tüm filtrelerden başarıyla geçti.")

# --- 🤖 ALFA V3.4 SİSTEMİ ---
elif current_page in ["v34_radar", "v34_diag"]:
    if st.session_state.df_merged is not None:
        df = st.session_state.df_merged.copy()

        # V3.4 TEMEL FİLTRELERİ  (-1 karşılaştırmaları EFEKTİF; endeks = XU100; gx/hx OR-yapılı)
        df["pdddLimit"] = np.where(df["ROE_0"] > 90, 8 + (df["ROE_0"] - 90) * 0.07, 8)

        a = df["Getiri_2a"] > oto_2a                                  # XU100
        b = df["ROE_0"] > tufe_12
        c = df["NetBorc_FAVOK"] < 4
        d = df["PDDD"] < df["pdddLimit"]
        e = df["Getiri_2h"] > (oto_2h - 10)                           # XU100
        f = df["Getiri_1a"] > -15
        ns = df["NetSatisBuyume"] > 0
        gx = (df["FAVOKBuyume"] > tufe_12) | (df["FAVOK_0"] > df["ef_FAVOK_1"])
        hx = ((df["BrutEFKBuyume"] > tufe_12) | (df["BrutEFK_0"] > df["ef_BrutEFK_1"])) & \
             ((df["EFKBuyume"] > tufe_12) | (df["EFK_0"] > df["ef_EFK_1"]))
        efkTeyit = (df["EFK_0"] >= df["ef_EFK_1"]) | (df["EFK_0"] >= df["ef_EFK_4"])
        roeTeyit = (df["ROE_0"] >= df["ef_ROE_1"]) | (df["ROE_0"] >= df["ef_ROE_4"])
        k = df["HAOran"] < 60

        temel_filtreliler_v34 = a & b & c & d & e & f & ns & (hx | gx | efkTeyit) & roeTeyit & k
        df_temel = df[temel_filtreliler_v34].copy()

        # RADAR VE TARAMALAR
        if current_page == "v34_radar":
            st.markdown("### 🤖 ALFA V3.4 - Radar & Taramalar")
            teknik_asanadan_gecenler = []
            analiz_edilen = 0

            if len(df_temel) > 0:
                with st.spinner("🔄 ALFA V3.4 için MA20, MA60, MA75, MA200 verileri anlık çekiliyor..."):
                    for idx, row in df_temel.iterrows():
                        kod = str(row["Kod"]).strip() + ".IS"
                        try:
                            hist = yf.download(kod, period="1y", progress=False)
                            if not hist.empty:
                                if isinstance(hist.columns, pd.MultiIndex):
                                    hist.columns = hist.columns.get_level_values(0)
                                hist = hist.sort_index().dropna()
                                close = hist['Close']
                                if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
                                if len(close) >= 200:
                                    analiz_edilen += 1
                                    c_val = float(close.iloc[-1])
                                    ma20 = float(close.rolling(20).mean().iloc[-1])
                                    ma60 = float(close.rolling(60).mean().iloc[-1])
                                    ma75 = float(close.rolling(75).mean().iloc[-1])
                                    ma200 = float(close.rolling(200).mean().iloc[-1])

                                    df_temel.loc[idx, 'Kapanis_Anlik'] = round(c_val, 2)
                                    df_temel.loc[idx, 'MA20'] = round(ma20, 2)
                                    df_temel.loc[idx, 'MA60'] = round(ma60, 2)
                                    df_temel.loc[idx, 'MA75'] = round(ma75, 2)
                                    df_temel.loc[idx, 'MA200'] = round(ma200, 2)

                                    # V3.4 TEKNİK ŞARTI: MA20>MA60 & MA75>MA200 & C>MA75
                                    if (ma20 > ma60) and (ma75 > ma200) and (c_val > ma75):
                                        teknik_asanadan_gecenler.append(idx)
                        except:
                            continue

            df_teknik = df_temel.loc[teknik_asanadan_gecenler].copy()

            m1, m2, m3 = st.columns(3)
            m1.metric(label="📊 Toplam İncelenen", value=f"{len(df)} Hisse")
            m2.metric(label="🏛️ Temel Filtreyi Geçen", value=f"{len(df_temel)} Hisse")
            m3.metric(label="🏆 Teknik & Trendi Geçen (Adaylar)", value=f"{len(df_teknik)} Hisse")
            if len(df_temel) > 0:
                st.caption(f"ℹ️ Temeli geçen {len(df_temel)} adaydan **{analiz_edilen}** tanesi için ≥200 günlük fiyat verisi çekilebildi. "
                           f"{'Hepsi çekildi.' if analiz_edilen == len(df_temel) else 'Çekilemeyen adaylar yfinance verisi/geçmişi eksik olabilir.'}")
            st.markdown("---")

            if len(df_temel) > 0:
                if len(df_teknik) > 0:
                    roe = df_teknik["ROE_0"]
                    m6 = df_teknik["Getiri_6a"]
                    m2 = df_teknik["Getiri_2a"]
                    c_fiyat = df_teknik["Kapanis"]          # trend skoru: fintables Kapanış (spec: c=Kapanis())
                    ma75_val = df_teknik["MA75"]
                    eb = df_teknik["EFKBuyume"]
                    fb = df_teknik["FAVOKBuyume"]
                    sb = df_teknik["NetSatisBuyume"]
                    pddd = df_teknik["PDDD"]

                    roeSkor = np.where(roe > 50, 100, roe * 2)
                    momentumSkor = np.where(m6 > 100, 100, m6)
                    m2Skor = np.where(m2 > 50, 100, m2 * 2)
                    trendYuzdesi = ((c_fiyat - ma75_val) / ma75_val) * 100
                    trendSkor = np.where(trendYuzdesi > 30, 30, np.where(trendYuzdesi < 0, 0, trendYuzdesi))
                    pdddSkor = np.where(pddd < 1.5, 25, np.where(pddd < 3, 15, np.where(pddd < 5, 5, np.where(pddd > 6, -10, 0))))
                    ebSkor = np.where(eb > 50, 100, np.where(eb > 0, eb * 2, 0))
                    fbSkor = np.where(fb > 50, 100, np.where(fb > 0, fb * 2, 0))
                    sbSkor = np.where(sb > 50, 100, np.where(sb > 0, sb * 2, 0))
                    buySkor = (ebSkor + fbSkor + sbSkor) / 3

                    # V3.4 FORMÜLÜ (V2.4'ün cezasız hali)
                    df_teknik["SKOR"] = (
                        (roeSkor * 0.30) + (buySkor * 0.20) + (momentumSkor * 0.15) +
                        (m2Skor * 0.05) + (trendSkor * 0.18) + (pdddSkor * 0.12)
                    )

                    df_sonuc = df_teknik.sort_values(by="SKOR", ascending=False).reset_index(drop=True)
                    df_sonuc.index += 1

                    st.markdown("### 🏆 ALFA V3.4 Portföy Adayları (En İyi Skorlar)")
                    def highlight_top5(s):
                        return ['background-color: #d4edda; font-weight: bold;' if s.name <= 5 else '' for _ in s]

                    st.dataframe(
                        df_sonuc[["Kod", "SKOR", "Bilanco_Durum", "ROE_0", "PDDD", "Getiri_1a", "Getiri_6a"]]
                        .head(15)
                        .style.format(precision=2, subset=["SKOR", "ROE_0", "PDDD", "Getiri_1a", "Getiri_6a"])
                        .apply(highlight_top5, axis=1),
                        use_container_width=True
                    )
                else: st.warning("ALFA V3.4 teknik kriterlerini sağlayan hisse bulunamadı.")
            else: st.warning("ALFA V3.4 temel kriterlerini sağlayan hisse bulunamadı.")

        # TEŞHİS PANELİ
        elif current_page == "v34_diag":
            st.markdown("### 🔍 ALFA V3.4 - Hisse Teşhis Paneli")
            hisse_listesi = [""] + sorted(df["Kod"].dropna().astype(str).str.upper().unique().tolist())
            secilen_hisse = st.selectbox("Hisse Seçin / Arayın:", options=hisse_listesi)

            if secilen_hisse:
                tek_hisse_df = df[df["Kod"].str.upper() == secilen_hisse]
                if not tek_hisse_df.empty:
                    th = tek_hisse_df.iloc[0]
                    p_lim = float(8 + (th['ROE_0'] - 90) * 0.07 if th['ROE_0'] > 90 else 8)

                    c_a = bool(th['Getiri_2a'] > oto_2a)
                    c_b = bool(th['ROE_0'] > tufe_12)
                    c_c = bool(th['NetBorc_FAVOK'] < 4)
                    c_d = bool(th['PDDD'] < p_lim)
                    c_e = bool(th['Getiri_2h'] > (oto_2h - 10))
                    c_f = bool(th['Getiri_1a'] > -15)
                    c_ns = bool(th['NetSatisBuyume'] > 0)
                    c_gx = bool((th['FAVOKBuyume'] > tufe_12) or (th['FAVOK_0'] > th['ef_FAVOK_1']))
                    c_hx = bool(((th['BrutEFKBuyume'] > tufe_12) or (th['BrutEFK_0'] > th['ef_BrutEFK_1'])) and ((th['EFKBuyume'] > tufe_12) or (th['EFK_0'] > th['ef_EFK_1'])))
                    c_teyit = bool((th['EFK_0'] >= th['ef_EFK_1']) or (th['EFK_0'] >= th['ef_EFK_4']))
                    c_roe_t = bool((th['ROE_0'] >= th['ef_ROE_1']) or (th['ROE_0'] >= th['ef_ROE_4']))
                    c_k = bool(th['HAOran'] < 60)

                    takilan = []
                    krit = [
                        ("2A Getiri > XU100", c_a, f"%{th['Getiri_2a']:.2f} vs XU100 %{oto_2a:.2f}"),
                        ("ROE > TÜFE", c_b, f"ROE %{th['ROE_0']:.2f} vs TÜFE %{tufe_12:.2f}"),
                        ("Net Borç/FAVÖK < 4", c_c, f"{dg(th['NetBorc_FAVOK'])}"),
                        ("PD/DD < Sınır", c_d, f"PDDD {dg(th['PDDD'])} vs sınır {dg(p_lim)}"),
                        ("2H Getiri Şartı", c_e, f"%{th['Getiri_2h']:.2f} vs eşik %{(oto_2h - 10):.2f}"),
                        ("1A Getiri > -15", c_f, f"%{th['Getiri_1a']:.2f}"),
                        ("Net Satış Büyümesi > 0", c_ns, f"%{th['NetSatisBuyume']:.2f}"),
                        ("Büyüme/Teyit", (c_hx or c_gx or c_teyit), f"FAVÖKBüy %{th['FAVOKBuyume']:.1f} · EFKBüy %{th['EFKBuyume']:.1f} · BrütBüy %{th['BrutEFKBuyume']:.1f}"),
                        ("ROE Teyit", c_roe_t, f"ROE %{th['ROE_0']:.2f} vs efROE-1 %{th['ef_ROE_1']:.2f} / efROE-4 %{th['ef_ROE_4']:.2f}"),
                        ("Halka Açıklık < 60", c_k, f"{dg(th['HAOran'])}"),
                    ]
                    takilan = [l for l, ok, v in krit if not ok]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"#### 🏛️ ALFA V3.4 Temel Analiz ({secilen_hisse})")
                        for l, ok, v in krit:
                            st.write(f"- {l}: {'✅ Geçti' if ok else '❌ Kaldı'}")

                    with col2:
                        st.markdown(f"#### 📈 ALFA V3.4 Teknik Şartlar")
                        try:
                            h_yf = yf.download(secilen_hisse + ".IS", period="1y", progress=False)
                            if not h_yf.empty:
                                if isinstance(h_yf.columns, pd.MultiIndex): h_yf.columns = h_yf.columns.get_level_values(0)
                                h_yf = h_yf.sort_index().dropna()
                                cls = h_yf['Close']
                                if isinstance(cls, pd.DataFrame): cls = cls.iloc[:, 0]
                                if len(cls) >= 200:
                                    c_val = float(cls.iloc[-1])
                                    m20 = float(cls.rolling(20).mean().iloc[-1])
                                    m60 = float(cls.rolling(60).mean().iloc[-1])
                                    m75 = float(cls.rolling(75).mean().iloc[-1])
                                    m200 = float(cls.rolling(200).mean().iloc[-1])

                                    t1 = m20 > m60
                                    t2 = m75 > m200
                                    t3 = c_val > m75

                                    st.write(f"- **MA20 > MA60:** {'✅' if t1 else '❌'} ({m20:.2f} > {m60:.2f})")
                                    st.write(f"- **MA75 > MA200:** {'✅' if t2 else '❌'} ({m75:.2f} > {m200:.2f})")
                                    st.write(f"- **Kapanış > MA75:** {'✅' if t3 else '❌'} ({c_val:.2f} > {m75:.2f})")

                                    if not (t1 and t2 and t3):
                                        takilan.append("ALFA V3.4 Teknik Şartı (MA20>MA60, MA75>MA200, C>MA75)")
                        except Exception as ex: st.error(f"Teknik hata: {ex}")

                    if takilan:
                        st.markdown("---")
                        st.warning("Hisse aşağıdaki kriterleri sağlamadığı için filtrelerden geçemedi:")
                        for kr in takilan: st.write(f"• {kr}")
                    else: st.success("🎉 Tebrikler! Hisse V3.4 filtrelerinin hepsinden başarıyla geçti.")
    else:
        st.info("👈 Önce 'Veri Yönetimi' sekmesinden Excel dosyalarını yükleyin.")

# --- 🤖 ALFA V1.2.3 SİSTEMİ ---
elif current_page in ["v123_radar", "v123_diag"]:
    if st.session_state.df_merged is not None:
        df = st.session_state.df_merged.copy()

        # V1.2.3 TEMEL FİLTRELERİ  (büyüme kapısı YOK; HAOran filtresi YOK; efektif kolon kullanmaz; endeks = XU100)
        # NOT: Spec'te pddLimit "IF(b>90,...)" (b boolean) yazıyordu; diğer algoritmalarla tutarlı olması için
        # standart forma aldım: pddLimit = IF(ROE>90, 8+(ROE-90)*0.07, 8). (ROE>90 olan hisselerde etkili, ör. TRHOL.)
        df["pdddLimit"] = np.where(df["ROE_0"] > 90, 8 + (df["ROE_0"] - 90) * 0.07, 8)

        a = df["Getiri_2a"] > oto_2a                                  # XU100
        b = df["ROE_0"] > tufe_12
        c = df["NetBorc_FAVOK"] < 4
        d = df["PDDD"] < df["pdddLimit"]
        e = df["Getiri_2h"] > (oto_2h - 10)                           # XU100
        f = df["Getiri_1a"] > -15

        # NOT: Spec'te k=HAOran() karşılaştırmasızdı (işlevsiz). Kullanıcı onayıyla V1.2.3'te
        # HAOran (halka açıklık) filtresi BULUNMAZ.
        temel_filtreliler_v123 = a & b & c & d & e & f
        df_temel = df[temel_filtreliler_v123].copy()

        # RADAR VE TARAMALAR
        if current_page == "v123_radar":
            st.markdown("### 🤖 ALFA V1.2.3 - Radar & Taramalar")
            teknik_asanadan_gecenler = []
            analiz_edilen = 0

            if len(df_temel) > 0:
                with st.spinner("🔄 ALFA V1.2.3 için MA20, MA60, MA75, MA200 verileri anlık çekiliyor..."):
                    for idx, row in df_temel.iterrows():
                        kod = str(row["Kod"]).strip() + ".IS"
                        try:
                            hist = yf.download(kod, period="1y", progress=False)
                            if not hist.empty:
                                if isinstance(hist.columns, pd.MultiIndex):
                                    hist.columns = hist.columns.get_level_values(0)
                                hist = hist.sort_index().dropna()
                                close = hist['Close']
                                if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
                                if len(close) >= 200:
                                    analiz_edilen += 1
                                    ma20 = float(close.rolling(20).mean().iloc[-1])
                                    ma60 = float(close.rolling(60).mean().iloc[-1])
                                    ma75 = float(close.rolling(75).mean().iloc[-1])
                                    ma200 = float(close.rolling(200).mean().iloc[-1])
                                    df_temel.loc[idx, 'MA20'] = round(ma20, 2)
                                    df_temel.loc[idx, 'MA60'] = round(ma60, 2)
                                    df_temel.loc[idx, 'MA75'] = round(ma75, 2)
                                    df_temel.loc[idx, 'MA200'] = round(ma200, 2)

                                    # V1.2.3 TEKNİK ŞARTI: MA20>MA60 & MA75>MA200
                                    if (ma20 > ma60) and (ma75 > ma200):
                                        teknik_asanadan_gecenler.append(idx)
                        except:
                            continue

            df_teknik = df_temel.loc[teknik_asanadan_gecenler].copy()

            m1, m2, m3 = st.columns(3)
            m1.metric(label="📊 Toplam İncelenen", value=f"{len(df)} Hisse")
            m2.metric(label="🏛️ Temel Filtreyi Geçen", value=f"{len(df_temel)} Hisse")
            m3.metric(label="🏆 Teknik & Trendi Geçen (Adaylar)", value=f"{len(df_teknik)} Hisse")
            if len(df_temel) > 0:
                st.caption(f"ℹ️ Temeli geçen {len(df_temel)} adaydan **{analiz_edilen}** tanesi için ≥200 günlük fiyat verisi çekilebildi. "
                           f"{'Hepsi çekildi.' if analiz_edilen == len(df_temel) else 'Çekilemeyen adaylar yfinance verisi/geçmişi eksik olabilir.'}")
            st.markdown("---")

            if len(df_temel) > 0:
                if len(df_teknik) > 0:
                    roe = df_teknik["ROE_0"]
                    m6 = df_teknik["Getiri_6a"]
                    m2 = df_teknik["Getiri_2a"]
                    c_fiyat = df_teknik["Kapanis"]
                    ma75_val = df_teknik["MA75"]
                    eb = df_teknik["EFKBuyume"]
                    fb = df_teknik["FAVOKBuyume"]
                    sb = df_teknik["NetSatisBuyume"]
                    pddd = df_teknik["PDDD"]

                    roeSkor = np.where(roe > 50, 100, roe * 2)
                    momentumSkor = np.where(m6 > 100, 100, m6)
                    m2Skor = np.where(m2 > 50, 100, m2 * 2)
                    trendYuzdesi = ((c_fiyat - ma75_val) / ma75_val) * 100
                    trendSkor = np.where(trendYuzdesi > 30, 30, np.where(trendYuzdesi < 0, 0, trendYuzdesi))
                    pdddSkor = np.where(pddd < 1.5, 25, np.where(pddd < 3, 15, np.where(pddd < 5, 5, np.where(pddd > 6, -10, 0))))
                    ebSkor = np.where(eb > 50, 100, np.where(eb > 0, eb * 2, 0))
                    fbSkor = np.where(fb > 50, 100, np.where(fb > 0, fb * 2, 0))
                    sbSkor = np.where(sb > 50, 100, np.where(sb > 0, sb * 2, 0))
                    buySkor = (ebSkor + fbSkor + sbSkor) / 3

                    # V1.2.3 FORMÜLÜ (V3.4 ile aynı: cezasız)
                    df_teknik["SKOR"] = (
                        (roeSkor * 0.30) + (buySkor * 0.20) + (momentumSkor * 0.15) +
                        (m2Skor * 0.05) + (trendSkor * 0.18) + (pdddSkor * 0.12)
                    )

                    df_sonuc = df_teknik.sort_values(by="SKOR", ascending=False).reset_index(drop=True)
                    df_sonuc.index += 1

                    st.markdown("### 🏆 ALFA V1.2.3 Portföy Adayları (En İyi Skorlar)")
                    def highlight_top5(s):
                        return ['background-color: #d4edda; font-weight: bold;' if s.name <= 5 else '' for _ in s]

                    st.dataframe(
                        df_sonuc[["Kod", "SKOR", "Bilanco_Durum", "ROE_0", "PDDD", "Getiri_1a", "Getiri_6a"]]
                        .head(15)
                        .style.format(precision=2, subset=["SKOR", "ROE_0", "PDDD", "Getiri_1a", "Getiri_6a"])
                        .apply(highlight_top5, axis=1),
                        use_container_width=True
                    )
                else: st.warning("ALFA V1.2.3 teknik kriterlerini sağlayan hisse bulunamadı.")
            else: st.warning("ALFA V1.2.3 temel kriterlerini sağlayan hisse bulunamadı.")

        # TEŞHİS PANELİ
        elif current_page == "v123_diag":
            st.markdown("### 🔍 ALFA V1.2.3 - Hisse Teşhis Paneli")
            hisse_listesi = [""] + sorted(df["Kod"].dropna().astype(str).str.upper().unique().tolist())
            secilen_hisse = st.selectbox("Hisse Seçin / Arayın:", options=hisse_listesi)

            if secilen_hisse:
                tek_hisse_df = df[df["Kod"].str.upper() == secilen_hisse]
                if not tek_hisse_df.empty:
                    th = tek_hisse_df.iloc[0]
                    p_lim = float(8 + (th['ROE_0'] - 90) * 0.07 if th['ROE_0'] > 90 else 8)

                    c_a = bool(th['Getiri_2a'] > oto_2a)
                    c_b = bool(th['ROE_0'] > tufe_12)
                    c_c = bool(th['NetBorc_FAVOK'] < 4)
                    c_d = bool(th['PDDD'] < p_lim)
                    c_e = bool(th['Getiri_2h'] > (oto_2h - 10))
                    c_f = bool(th['Getiri_1a'] > -15)

                    takilan = []
                    krit = [
                        ("2A Getiri > XU100", c_a, f"%{th['Getiri_2a']:.2f} vs XU100 %{oto_2a:.2f}"),
                        ("ROE > TÜFE", c_b, f"ROE %{th['ROE_0']:.2f} vs TÜFE %{tufe_12:.2f}"),
                        ("Net Borç/FAVÖK < 4", c_c, f"{dg(th['NetBorc_FAVOK'])}"),
                        ("PD/DD < Sınır", c_d, f"PDDD {dg(th['PDDD'])} vs sınır {dg(p_lim)}"),
                        ("2H Getiri Şartı", c_e, f"%{th['Getiri_2h']:.2f} vs eşik %{(oto_2h - 10):.2f}"),
                        ("1A Getiri > -15", c_f, f"%{th['Getiri_1a']:.2f}"),
                    ]
                    takilan = [l for l, ok, v in krit if not ok]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"#### 🏛️ ALFA V1.2.3 Temel Analiz ({secilen_hisse})")
                        for l, ok, v in krit:
                            st.write(f"- {l}: {'✅ Geçti' if ok else '❌ Kaldı'}")
                        st.caption("ℹ️ V1.2.3'te büyüme (NetSatış/FAVÖK/EFK) ve halka açıklık (HAOran) temel filtrede yer almaz; büyüme yalnızca skorlamada kullanılır.")

                    with col2:
                        st.markdown(f"#### 📈 ALFA V1.2.3 Teknik Şartlar")
                        try:
                            h_yf = yf.download(secilen_hisse + ".IS", period="1y", progress=False)
                            if not h_yf.empty:
                                if isinstance(h_yf.columns, pd.MultiIndex): h_yf.columns = h_yf.columns.get_level_values(0)
                                h_yf = h_yf.sort_index().dropna()
                                cls = h_yf['Close']
                                if isinstance(cls, pd.DataFrame): cls = cls.iloc[:, 0]
                                if len(cls) >= 200:
                                    m20 = float(cls.rolling(20).mean().iloc[-1])
                                    m60 = float(cls.rolling(60).mean().iloc[-1])
                                    m75 = float(cls.rolling(75).mean().iloc[-1])
                                    m200 = float(cls.rolling(200).mean().iloc[-1])
                                    t1 = m20 > m60
                                    t2 = m75 > m200
                                    st.write(f"- **MA20 > MA60:** {'✅' if t1 else '❌'} ({m20:.2f} > {m60:.2f})")
                                    st.write(f"- **MA75 > MA200:** {'✅' if t2 else '❌'} ({m75:.2f} > {m200:.2f})")
                                    if not (t1 and t2):
                                        takilan.append("ALFA V1.2.3 Teknik Şartı (MA20>MA60, MA75>MA200)")
                        except Exception as ex: st.error(f"Teknik hata: {ex}")

                    if takilan:
                        st.markdown("---")
                        st.warning("Hisse aşağıdaki kriterleri sağlamadığı için filtrelerden geçemedi:")
                        for kr in takilan: st.write(f"• {kr}")
                    else: st.success("🎉 Tebrikler! Hisse V1.2.3 filtrelerinin hepsinden başarıyla geçti.")
    else:
        st.info("👈 Önce 'Veri Yönetimi' sekmesinden Excel dosyalarını yükleyin.")
