import streamlit as st

from auth import get_user_by_id
from config import APP_NAME, DEPARTMENT, UNIVERSITY
from database import init_db
from views.admin import render_admin
from views.faculty import render_faculty
from views.hod import render_hod
from views.login import render_login
from views.student import render_student
from views.theme import inject_theme

from config import APP_NAME, DEPARTMENT, LOGO_PATH, UNIVERSITY

st.set_page_config(page_title=APP_NAME, page_icon="🎓", layout="wide")
inject_theme()
init_db()

if "user_id" not in st.session_state:
    render_login()
    st.stop()

user = get_user_by_id(st.session_state.user_id)
if user is None or not user.is_active:
    st.session_state.clear()
    st.warning("Your session is no longer valid. Sign in again.")
    render_login()
    st.stop()

# Distraction-free viva mode CSS flag when viva is active
if st.session_state.get("viva_active"):
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        .stAppHeader { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Top-Right Small Logout Icon
top_c1, top_c2 = st.columns([22, 1])
with top_c2:
    if st.button("🚪", help="Logout", key="top_right_logout_icon"):
        if st.session_state.get("viva_active"):
            st.session_state.confirm_logout_viva = True
        else:
            st.session_state.clear()
            st.rerun()

if st.session_state.get("confirm_logout_viva"):
    st.warning("⚠️ **Active Examination Warning**: You have an active oral viva examination in progress. Logging out will leave the exam session.")
    dlg_c1, dlg_c2 = st.columns(2)
    if dlg_c1.button("Yes, Log Out", type="primary", key="confirm_logout_yes"):
        st.session_state.clear()
        st.rerun()
    if dlg_c2.button("Cancel & Resume Viva", key="confirm_logout_cancel"):
        st.session_state.pop("confirm_logout_viva", None)
        st.rerun()
else:
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
        st.markdown(f"**{UNIVERSITY}**")
        st.caption(DEPARTMENT)
        st.divider()
        st.write(user.full_name)
        st.caption(user.email)
        st.caption(user.role.title())
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

if user.role == "ADMIN":
    render_admin(user)
elif user.role == "FACULTY":
    render_faculty(user)
elif user.role == "HOD":
    render_hod(user)
elif user.role == "STUDENT":
    render_student(user)
else:
    st.error("This account does not have a recognised role.")
