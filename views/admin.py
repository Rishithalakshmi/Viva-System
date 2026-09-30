from __future__ import annotations

import pandas as pd
import streamlit as st

from auth import create_user, reset_user_password, set_user_active, set_user_role
from config import ROLES
from models import User
from services.viva import list_users
from views.theme import banner


def render_admin(user: User) -> None:
    banner(
        "Administrator",
        "Create university accounts, assign roles, and activate or deactivate access.",
        user.email,
    )

    create_tab, manage_tab = st.tabs(["Create account", "Manage users"])

    with create_tab:
        st.caption("Every account must use a numeric university email such as 99240040986@kalasalingam.ac.in.")
        with st.form("admin_create_user"):
            full_name = st.text_input("Full name")
            email = st.text_input("University email")
            role = st.selectbox("Role", list(ROLES))
            password = st.text_input("Temporary password", type="password")
            active = st.checkbox("Activate immediately", value=True)
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            created, error = create_user(
                email,
                password,
                full_name,
                role,
                is_active=active,
                created_by_id=user.id,
            )
            if error:
                st.error(error)
            else:
                st.success(f"Created {created.role} account for {created.email}.")

    with manage_tab:
        users = list_users()
        if not users:
            st.info("No users found.")
            return
        frame = pd.DataFrame(
            [
                {
                    "ID": item.id,
                    "Name": item.full_name,
                    "Email": item.email,
                    "Role": item.role,
                    "Active": "Yes" if item.is_active else "No",
                    "Created": item.created_at.strftime("%Y-%m-%d %H:%M"),
                }
                for item in users
            ]
        )
        st.dataframe(frame, use_container_width=True, hide_index=True)

        options = {f"{item.full_name} · {item.email}": item for item in users}
        selected_label = st.selectbox("Select account", list(options))
        selected = options[selected_label]
        c1, c2, c3 = st.columns(3)
        new_role = c1.selectbox("Role", list(ROLES), index=list(ROLES).index(selected.role))
        if c2.button("Save role", use_container_width=True):
            error = set_user_role(selected.id, new_role)
            st.error(error) if error else st.success("Role updated.")
            if not error:
                st.rerun()
        if selected.is_active:
            if c3.button("Deactivate", use_container_width=True):
                if selected.id == user.id:
                    st.error("You cannot deactivate your own administrator account.")
                else:
                    set_user_active(selected.id, False)
                    st.rerun()
        else:
            if c3.button("Activate", type="primary", use_container_width=True):
                set_user_active(selected.id, True)
                st.rerun()

        st.markdown("#### Reset password")
        with st.form("reset_password"):
            new_password = st.text_input("New password", type="password")
            if st.form_submit_button("Reset password"):
                error = reset_user_password(selected.id, new_password)
                st.error(error) if error else st.success("Password updated.")
