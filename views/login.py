import streamlit as st

from auth import authenticate, bootstrap_admin, register_student, user_count
from views.theme import banner


from config import UNIVERSITY

def render_login() -> None:
    banner(
        "Virtual Laboratory Viva Assessment",
        "Secure sign-in for students, faculty, Heads of Department and administrators.",
        f"{UNIVERSITY} · Laboratory Assessment Cell",
    )

    has_users = user_count() > 0
    tabs = ["Sign in"]
    if has_users:
        tabs.append("Student registration")
    else:
        tabs.append("Create first administrator")

    tab_map = st.tabs(tabs)

    with tab_map[0]:
        st.subheader("University sign-in")
        st.caption("Email must be a numeric university ID, for example 99240040986@kalasalingam.ac.in.")
        with st.form("login_form"):
            email = st.text_input("University email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            user, error = authenticate(email, password)
            if error:
                st.error(error)
            else:
                st.session_state.user_id = user.id
                st.session_state.role = user.role
                st.rerun()

    with tab_map[1]:
        if not has_users:
            st.subheader("Initial administrator setup")
            st.info("No accounts exist yet. Create the first administrator using a valid university email.")
            with st.form("bootstrap_form"):
                full_name = st.text_input("Full name")
                email = st.text_input("Administrator email")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create administrator", type="primary", use_container_width=True)
            if submitted:
                if password != confirm:
                    st.error("Passwords do not match.")
                else:
                    user, error = bootstrap_admin(email, password, full_name)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.user_id = user.id
                        st.session_state.role = user.role
                        st.rerun()
        else:
            st.subheader("Student registration")
            st.caption("Register your student account with your university email and sign in immediately.")
            with st.form("register_form"):
                full_name = st.text_input("Full name")
                email = st.text_input("University email")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Register and Sign In", type="primary", use_container_width=True)
            if submitted:
                if password != confirm:
                    st.error("Passwords do not match.")
                else:
                    _user, error = register_student(email, password, full_name)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.user_id = _user.id
                        st.session_state.role = _user.role
                        st.success("Registration successful! Logging in...")
                        st.rerun()
