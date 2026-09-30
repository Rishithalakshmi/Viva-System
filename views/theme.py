import base64
from pathlib import Path
from config import LOGO_PATH, UNIVERSITY

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
  font-family: "Source Sans 3", "Segoe UI", sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 400px at 10% -10%, rgba(201,162,39,0.10), transparent 50%),
    linear-gradient(180deg, #f4efe4 0%, #ece6d8 100%);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent !important;}

.kare-banner {
  background: linear-gradient(90deg, #0b1f3a 0%, #16365e 55%, #7a1f2b 100%);
  color: #f7f1e3;
  border: 1px solid rgba(201,162,39,0.35);
  border-radius: 18px;
  padding: 18px 24px;
  margin-bottom: 18px;
  box-shadow: 0 10px 30px rgba(11,31,58,0.18);
  display: flex;
  align-items: center;
  gap: 20px;
}
.kare-banner h1 {
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 1.65rem;
  margin: 0 0 4px 0;
  letter-spacing: 0.2px;
  color: #ffffff;
}
.kare-banner p { margin: 0; opacity: 0.92; font-size: 0.95rem; }
.kare-kicker {
  color: #e7c56a;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.kare-card {
  background: rgba(255,252,246,0.92);
  border: 1px solid #d9d0bc;
  border-radius: 16px;
  padding: 16px 18px;
  margin-bottom: 12px;
}
.kare-chip {
  display: inline-block;
  background: #0b1f3a;
  color: #f7f1e3;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.78rem;
  margin-right: 6px;
}
.stButton>button {
  border-radius: 10px;
  font-weight: 600;
}
div[data-testid="stMetric"] {
  background: #fffaf1;
  border: 1px solid #d9d0bc;
  border-radius: 14px;
  padding: 8px 12px;
}
[data-testid="stSidebar"] {
  background: #0b1f3a;
}
[data-testid="stSidebar"] * {
  color: #f4efe4 !important;
}
[data-testid="stSidebar"] .stSelectbox div, [data-testid="stSidebar"] input {
  color: #0b1f3a !important;
}
.status-line {
  font-weight: 700;
  color: #7a1f2b;
  letter-spacing: 0.04em;
}
.viva-container {
  background: #ffffff;
  border-radius: 16px;
  padding: 24px;
  border: 1px solid #d9d0bc;
  box-shadow: 0 8px 24px rgba(0,0,0,0.06);
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
        <div style="background: #ffffff; padding: 6px 12px; border-radius: 12px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
          <img src="data:image/png;base64,{logo_b64}" style="height: 60px; width: auto; max-width: 240px; object-fit: contain;" alt="KARE Logo" />
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

