import base64
from pathlib import Path
from config import LOGO_PATH, UNIVERSITY

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
  background-color: #000000 !important;
  color: #ffffff !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: #000000 !important;}

/* Global Background */
.stApp {
  background-color: #000000 !important;
  color: #ffffff !important;
}

/* KARE Banner - Strict 2-color: Black + Light Lavender */
.kare-banner {
  background: #000000;
  color: #ffffff;
  border: 1.5px solid #D8B4FE;
  border-radius: 12px;
  padding: 18px 24px;
  margin-bottom: 20px;
  box-shadow: 0 0 15px rgba(216, 180, 254, 0.15);
  display: flex;
  align-items: center;
  gap: 20px;
}
.kare-banner h1 {
  font-size: 1.6rem;
  margin: 0 0 4px 0;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #D8B4FE !important;
}
.kare-banner p { margin: 0; opacity: 0.9; font-size: 0.95rem; color: #ffffff !important; }
.kare-kicker {
  color: #D8B4FE !important;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 4px;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
  color: #D8B4FE !important;
}

/* Cards & Containers */
.kare-card, .viva-container {
  background: #000000 !important;
  border: 1.5px solid #D8B4FE !important;
  border-radius: 12px !important;
  padding: 18px 20px !important;
  margin-bottom: 14px !important;
  color: #ffffff !important;
}

.kare-chip {
  display: inline-block;
  background: #000000;
  color: #D8B4FE;
  border: 1px solid #D8B4FE;
  border-radius: 999px;
  padding: 4px 12px;
  font-size: 0.78rem;
  font-weight: 600;
  margin-right: 6px;
}

/* Metrics */
div[data-testid="stMetric"] {
  background: #000000 !important;
  border: 1.5px solid #D8B4FE !important;
  border-radius: 10px !important;
  padding: 12px 16px !important;
}
div[data-testid="stMetric"] label {
  color: #ffffff !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
  color: #D8B4FE !important;
  font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
  background-color: #000000 !important;
  color: #D8B4FE !important;
  border: 1.5px solid #D8B4FE !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  transition: all 0.2s ease-in-out !important;
}
.stButton > button:hover {
  background-color: #D8B4FE !important;
  color: #000000 !important;
  border-color: #D8B4FE !important;
  box-shadow: 0 0 10px rgba(216, 180, 254, 0.4) !important;
}
.stButton > button[kind="primary"] {
  background-color: #D8B4FE !important;
  color: #000000 !important;
  border: 1.5px solid #D8B4FE !important;
}
.stButton > button[kind="primary"]:hover {
  background-color: #000000 !important;
  color: #D8B4FE !important;
  box-shadow: 0 0 12px rgba(216, 180, 254, 0.5) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background-color: #000000 !important;
  border-right: 1.5px solid #D8B4FE !important;
}
[data-testid="stSidebar"] * {
  color: #ffffff !important;
}

/* Inputs, Textareas, Selectboxes */
input, textarea, select, div[data-baseweb="select"] {
  background-color: #000000 !important;
  color: #ffffff !important;
  border-color: #D8B4FE !important;
}
div[data-baseweb="input"] {
  background-color: #000000 !important;
  border: 1px solid #D8B4FE !important;
  border-radius: 8px !important;
}
div[data-baseweb="select"] > div {
  background-color: #000000 !important;
  border: 1px solid #D8B4FE !important;
}

/* Tabs */
button[data-baseweb="tab"] {
  color: #ffffff !important;
  border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: #D8B4FE !important;
  border-bottom: 2px solid #D8B4FE !important;
  font-weight: 700 !important;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div > div {
  background-color: #D8B4FE !important;
}
div[data-testid="stProgress"] > div {
  background-color: #000000 !important;
  border: 1px solid #D8B4FE !important;
}

/* Alert boxes / Notifications */
div[data-testid="stAlert"] {
  background-color: #000000 !important;
  color: #ffffff !important;
  border: 1.5px solid #D8B4FE !important;
  border-radius: 8px !important;
}
div[data-testid="stAlert"] * {
  color: #ffffff !important;
}

/* Expanders */
div[data-testid="stExpander"] {
  background-color: #000000 !important;
  border: 1px solid #D8B4FE !important;
  border-radius: 8px !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
  border: 1px solid #D8B4FE !important;
  border-radius: 8px !important;
  background-color: #000000 !important;
}

hr {
  border-color: #D8B4FE !important;
  opacity: 0.3 !important;
}

.status-line {
  font-weight: 700;
  color: #D8B4FE !important;
  letter-spacing: 0.04em;
}

/* Audio Player */
audio {
  filter: invert(1) hue-rotate(240deg);
}
</style>
"""


def _get_logo_base64() -> str:
    if LOGO_PATH.exists():
        try:
            return base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        except Exception:
            return ""
    return ""


def inject_theme():
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def banner(title: str, subtitle: str, kicker: str = UNIVERSITY):
    import streamlit as st
    logo_b64 = _get_logo_base64()
    logo_html = ""
    if logo_b64:
        logo_html = f"""
        <div style="background: #000000; border: 1px solid #D8B4FE; padding: 6px 12px; border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
          <img src="data:image/png;base64,{logo_b64}" style="height: 55px; width: auto; max-width: 220px; object-fit: contain;" alt="KARE Logo" />
        </div>
        """
    st.markdown(
        f"""
        <div class="kare-banner">
          {logo_html}
          <div style="flex-grow: 1;">
            <div class="kare-kicker">{kicker}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
