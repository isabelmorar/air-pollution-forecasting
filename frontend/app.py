import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import base64
import os
import requests

img_path = os.path.join(os.path.dirname(__file__), "snoopy_image.png")
with open(img_path, "rb") as f:
    SNOOPY_B64 = base64.b64encode(f.read()).decode()

st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500;600&display=swap');

:root {
    --bg:         #F0F6FC;
    --surface:    #FFFFFF;
    --blue-dark:  #1A4A80;
    --blue-mid:   #2B6CB0;
    --blue-light: #BEE3F8;
    --blue-pale:  #EBF4FF;
    --ink:        #111827;
    --ink-mid:    #1F2937;
    --muted:      #374151;
    --subtle:     #6B7280;
    --border:     #C3D9EF;
    --green-bg:   #F0FFF4;
    --green-txt:  #22543D;
    --amber-bg:   #FFFBEB;
    --amber-txt:  #92400E;
    --red-bg:     #FFF5F5;
    --red-txt:    #9B1C1C;
    --radius:     14px;
    --shadow:     0 2px 16px rgba(27,79,138,0.09);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--ink);
}
.stApp { background-color: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.4rem !important; max-width: 1300px; }

/* ── HERO ── */
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    color: var(--blue-dark);
    letter-spacing: -1.5px;
    line-height: 1.05;
    text-align: center;
    margin-bottom: 6px;
}
.hero-sub {
    font-size: 0.92rem;
    color: var(--subtle);
    text-align: center;
    font-weight: 400;
    margin-bottom: 1.8rem;
    letter-spacing: 0.1px;
}

/* ── SNOOPY CARD ── */
.snoopy-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 20px 22px;
    box-shadow: var(--shadow);
}
.snoopy-card img {
    width: 200px;
    height: 200px;
    object-fit: cover;
    border-radius: 50%;
    border: 3px solid var(--blue-light);
    margin-bottom: 16px;
    display: block;
    margin-left: auto;
    margin-right: auto;
}
.snoopy-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 8px;
    letter-spacing: -0.3px;
}
.snoopy-desc {
    font-size: 0.83rem;
    color: var(--muted);
    line-height: 1.6;
    margin-bottom: 16px;
    text-align: left;
}
.snoopy-reading {
    background: var(--blue-pale);
    border: 1px solid var(--blue-light);
    border-left: 3px solid var(--blue-mid);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 0.82rem;
    color: var(--ink-mid);
    line-height: 1.7;
    text-align: left;
}
.snoopy-reading strong { color: var(--ink); }

/* ── AQI PILL ── */
.aqi-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    margin-top: 6px;
}

/* ── SECTION TITLE ── */
.section-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--blue-mid);
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin: 1.2rem 0 0.7rem;
    border-left: 3px solid var(--blue-mid);
    padding-left: 10px;
}

/* ── METRIC CARD ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 14px 14px;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--blue-dark), var(--blue-light));
    border-radius: var(--radius) var(--radius) 0 0;
}
.mc-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--subtle);
    text-transform: uppercase;
    letter-spacing: 0.9px;
    margin-bottom: 8px;
}
.mc-value {
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1;
    font-family: 'DM Mono', monospace;
    letter-spacing: -1px;
}
.mc-unit {
    font-size: 0.73rem;
    color: var(--subtle);
    font-weight: 500;
    margin-top: 4px;
}

/* ── INPUT SECTION ── */
.input-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 24px 18px;
    box-shadow: var(--shadow);
}
.input-label {
    font-size: 0.70rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 2px;
}

/* ── PREDICT RESULT ── */
.pred-card {
    background: var(--blue-pale);
    border: 1px solid var(--blue-light);
    border-radius: var(--radius);
    padding: 30px 26px;
    text-align: center;
    box-shadow: var(--shadow);
}
.pred-label {
    font-size: 0.70rem;
    font-weight: 700;
    color: var(--subtle);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 12px;
}
.pred-num {
    font-size: 4.5rem;
    font-weight: 600;
    color: var(--blue-dark);
    line-height: 1;
    font-family: 'DM Mono', monospace;
    letter-spacing: -3px;
}
.pred-unit {
    font-size: 0.88rem;
    color: var(--subtle);
    font-weight: 500;
    margin-top: 6px;
}
.pred-quote {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--blue-mid);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.84rem;
    color: var(--ink-mid);
    margin-top: 16px;
    line-height: 1.55;
    text-align: left;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 5px;
    box-shadow: var(--shadow);
    margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    border: none !important;
    color: var(--subtle) !important;
    padding: 7px 22px !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: var(--blue-pale) !important;
    color: var(--blue-dark) !important;
    font-weight: 700 !important;
}

/* ── BUTTON ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    background: var(--blue-dark) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 32px !important;
    box-shadow: 0 2px 10px rgba(27,79,138,0.28) !important;
    transition: background 0.15s !important;
    letter-spacing: 0.1px !important;
}
.stButton > button:hover { background: var(--blue-mid) !important; }

.stSelectbox label, .stSlider label { display: none; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def aqi_level(pm25):
    if pm25 <= 12:
        return "Good",       "#276749", "white", "El aire está limpio. Buena jornada para actividades al aire libre."
    if pm25 <= 35:
        return "Moderate",   "#92400E", "white", "Calidad aceptable. Grupos sensibles deben tomar precauciones."
    if pm25 <= 55:
        return "Unhealthy*", "#C05621", "white", "Grupos sensibles deberían limitar el tiempo al aire libre."
    if pm25 <= 150:
        return "Unhealthy",  "#9B1C1C", "white", "Evita la exposición prolongada al exterior."
    return   "Hazardous",   "#702459", "white", "Permanece en interiores. Condiciones extremadamente peligrosas."

def fake_forecast(base, hours=24):
    np.random.seed(7)
    vals  = np.clip(base + np.linspace(0, random.uniform(-8, 8), hours)
                    + np.random.normal(0, base * 0.07, hours), 0, 500)
    return pd.DataFrame({"time": [datetime.now() + timedelta(hours=i) for i in range(hours)],
                         "pm25": vals})

def call_api(pollution, dew, temp, press, wnd_spd, snow, rain, wnd_dir):
    payload = {
    "pm25": pollution, "dew": dew, "temp": temp,
    "press": press, "wnd_spd": wnd_spd, "snow": snow, 
    "rain": rain, "wnd_dir": wnd_dir
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
        response.raise_for_status()
        return response.json()["prediction"]
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar al backend. Verifica que el servidor esté corriendo.")
        return None
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return None

PLOT_CFG = dict(
    paper_bgcolor="white", plot_bgcolor="#F7FAFD",
    font_family="DM Sans", font_color="#374151",
    margin=dict(t=40, b=24, l=16, r=16), height=275,
    xaxis=dict(showgrid=False, linecolor="#C3D9EF", tickfont=dict(color="#6B7280", size=11)),
    yaxis=dict(gridcolor="#E2EDF7", linecolor="#C3D9EF", tickfont=dict(color="#6B7280", size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12, color="#374151")),
    title_font=dict(size=13, color="#1A4A80", family="DM Sans"),
)

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "LSTM-Multivariate_pollution.csv"))
    df["time"] = pd.to_datetime(df["date"])
    return df.tail(72)
df     = load_data()
latest = df.iloc[-1]
lvl, pill_color, pill_text, quote = aqi_level(latest["pollution"])


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">Air Quality Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">PM2.5 Forecasting · Dashboard & Prediction</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 3.8], gap="large")

with left:
    st.markdown(f"""
    <div class="snoopy-card">
        <img src="data:image/png;base64,{SNOOPY_B64}" alt="Snoopy"/>
        <div class="snoopy-name">Air Quality Predictor</div>
        <div class="snoopy-desc">
            Real-time monitoring of PM2.5<br><br>
        </div>
        <div class="snoopy-reading">
            <strong>Latest Reading</strong><br>
            PM2.5 &nbsp; <strong>{latest['pollution']:.1f} μg/m³</strong><br>
            Temp &nbsp;&nbsp;&nbsp; {latest['temp']:.1f} °C<br>
            Wind &nbsp; {latest['wnd_spd']:.1f} km/h<br><br>
            Current Status<br>
            <span class="aqi-pill" style="background:{pill_color}; color:{pill_text};">{lvl}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    tab1, tab2 = st.tabs(["  Dashboard  ", "  Predict  "])

    # ── DASHBOARD ─────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Current Conditions</div>', unsafe_allow_html=True)

        metrics = [
            ("PM2.5",     f"{latest['pollution']:.1f}", "μg/m³"),
            ("Temp",      f"{latest['temp']:.1f}",      "°C"),
            ("Wind Speed", f"{latest['wnd_spd']:.1f}",   "km/h"),
            ("Dew Point", f"{latest['dew']:.1f}",       "°C"),
            ("Pressure",   f"{latest['press']:.0f}",     "hPa"),
            ("Snow",     "0.0",                        "cm"),
            ("Rain",    "0.0",                        "mm"),
        ]
        cols = st.columns(7)
        for col, (label, val, unit) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="mc-label">{label}</div>
                    <div class="mc-value">{val}</div>
                    <div class="mc-unit">{unit}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Historical 72 Hours</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 2])

        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["time"], y=df["pollution"],
                mode="lines", fill="tozeroy",
                line=dict(color="#2B6CB0", width=2.5),
                fillcolor="rgba(43,108,176,0.09)",
            ))
            for thr, lbl, clr in [(12,"Good","#276749"),(35,"Moderate","#92400E"),(55,"Bad","#9B1C1C")]:
                fig.add_hline(y=thr, line_dash="dot", line_color=clr, line_width=1.2,
                              annotation_text=lbl, annotation_font_size=10,
                              annotation_font_color=clr, annotation_position="right")
            fig.update_layout(**{**PLOT_CFG, "title": "PM2.5 (μg/m³)"})
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df["wnd_spd"], y=df["pollution"], mode="markers",
                marker=dict(color=df["temp"], colorscale="Blues", size=7, opacity=0.75,
                            colorbar=dict(title="°C", thickness=10, len=0.8,
                                         tickfont=dict(color="#6B7280"),
                                         title_font=dict(color="#6B7280"))),
            ))
            fig2.update_layout(**{**PLOT_CFG, "title": "Viento vs PM2.5",
                                  "xaxis_title": "Wind (km/h)", "yaxis_title": "PM2.5 (μg/m³)"})
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">Forecast 24 Hours</div>', unsafe_allow_html=True)
        fdf = fake_forecast(latest["pollution"])
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=fdf["time"], y=fdf["pm25"], mode="lines",
            line=dict(color="#1A4A80", width=2, dash="dot"),
            fill="tozeroy", fillcolor="rgba(26,74,128,0.07)",
        ))
        fig3.update_layout(**{**PLOT_CFG, "height": 215,
                               "title": "PM2.5 estimated — next 24h"})
        st.plotly_chart(fig3, use_container_width=True)

    # ── PREDICCION ────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-title">Input Variables</div>', unsafe_allow_html=True)

        r1 = st.columns(4)
        fields_r1 = [
            ("PM2.5 actual (μg/m³)", "pm25",  0,   500, 94),
            ("Dew Point (°C)",        "dew",  -40,  28,   2),
            ("Temperature (°C)",      "temp", -19,  42,  12),
            ("Pressure (hPa)",         "press", 991, 1046, 1016),
        ]
        sliders = {}
        for col, (label, key, mn, mx, default) in zip(r1, fields_r1):
            with col:
                st.markdown(f'<div class="input-label">{label}</div>', unsafe_allow_html=True)
                sliders[key] = st.slider(key, mn, mx, default, label_visibility="collapsed")

        r2 = st.columns(4)
        with r2[0]:
            st.markdown('<div class="input-label">Wind Speed (km/h)</div>', unsafe_allow_html=True)
            sliders["wnd_spd"] = st.slider("wnd_spd", 0, 200, 24, label_visibility="collapsed")
        with r2[1]:
            st.markdown('<div class="input-label">Rain (mm)</div>', unsafe_allow_html=True)
            sliders["rain"] = st.slider("rain", 0, 36, 0, label_visibility="collapsed")
        with r2[2]:
            st.markdown('<div class="input-label">Snow (cm)</div>', unsafe_allow_html=True)
            sliders["snow"] = st.slider("snow", 0, 27, 0, label_visibility="collapsed")
        with r2[3]:
            st.markdown('<div class="input-label">Wind Direction</div>', unsafe_allow_html=True)
            sliders["wnd_dir"] = st.selectbox("wnd_dir", ["NE", "NW", "SE", "cv"], label_visibility="collapsed")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")

        btn_col, _ = st.columns([1, 3])
        with btn_col:
            predict_btn = st.button("Predict")

        if predict_btn:
            pred = call_api(sliders["pm25"], sliders["dew"], sliders["temp"],
                    sliders["press"], sliders["wnd_spd"], sliders["snow"], sliders["rain"], sliders["wnd_dir"])
            p_lvl, p_color, p_txt, p_quote = aqi_level(pred)
            st.markdown("")
            res_col, _ = st.columns([1, 2])
            with res_col:
                st.markdown(f"""
                <div class="pred-card">
                    <div class="pred-label">PM2.5 Predicho — h+1</div>
                    <div class="pred-num">{pred}</div>
                    <div class="pred-unit">μg/m³</div>
                    <div style="margin-top:14px;">
                        <span class="aqi-pill" style="background:{p_color}; color:{p_txt};">{p_lvl}</span>
                    </div>
                    <div class="pred-quote">{p_quote}</div>
                </div>
                """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; font-size:0.74rem; color:#9CA3AF; margin-top:2.5rem;
            padding-top:1rem; border-top:1px solid #C3D9EF; letter-spacing:0.2px;">
    Snoopy Air Watch &nbsp;·&nbsp; PM2.5 Forecasting
</div>
""", unsafe_allow_html=True)