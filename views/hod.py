from __future__ import annotations

import pandas as pd
import streamlit as st

from config import MAX_MARKS_TOTAL
from models import User
from services.viva import attempt_rows, hod_stats
from views.theme import banner


def render_hod(user: User) -> None:
    banner(
        f"Head of Department · {user.full_name}",
        "Live assessment statistics, student performance and experiment-wise reports drawn from the database.",
        user.email,
    )

    stats = hod_stats()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Students", stats["students"])
    c2.metric("Faculty", stats["faculty"])
    c3.metric("Manuals", stats["manuals"])
    c4.metric("Experiments", stats["experiments"])
    c5.metric("Completed vivas", stats["completed_vivas"])
    c6.metric("Avg session", f"{stats['average_session_marks']:.1f}/{MAX_MARKS_TOTAL}")

    rows = attempt_rows()
    if not rows:
        st.info("No viva results have been recorded yet.")
        return

    frame = pd.DataFrame(rows)
    frame["created_at"] = pd.to_datetime(frame["created_at"])

    st.markdown("#### Filters")
    f1, f2, f3 = st.columns(3)
    students = ["All"] + sorted(frame["email"].unique().tolist())
    experiments = ["All"] + sorted(frame["experiment"].unique().tolist())
    manuals = ["All"] + sorted(frame["manual"].unique().tolist())
    student_filter = f1.selectbox("Student", students)
    experiment_filter = f2.selectbox("Experiment", experiments)
    manual_filter = f3.selectbox("Manual", manuals)

    filtered = frame.copy()
    if student_filter != "All":
        filtered = filtered[filtered["email"] == student_filter]
    if experiment_filter != "All":
        filtered = filtered[filtered["experiment"] == experiment_filter]
    if manual_filter != "All":
        filtered = filtered[filtered["manual"] == manual_filter]

    st.markdown("#### Student performance")
    student_perf = (
        filtered.groupby(["student", "email"])
        .agg(Answers=("score", "count"), Average=("score", "mean"), Highest=("score", "max"), Lowest=("score", "min"))
        .reset_index()
        .sort_values("Average", ascending=False)
    )
    st.dataframe(student_perf, use_container_width=True, hide_index=True)

    st.markdown("#### Experiment performance")
    exp_perf = (
        filtered.groupby(["manual", "experiment"])
        .agg(Answers=("score", "count"), Average=("score", "mean"), Highest=("score", "max"), Lowest=("score", "min"))
        .reset_index()
        .sort_values("Average")
    )
    st.dataframe(exp_perf, use_container_width=True, hide_index=True)

    if not filtered.empty:
        st.bar_chart(exp_perf.set_index("experiment")["Average"], use_container_width=True)

    st.markdown("#### Detailed report")
    st.dataframe(
        filtered[
            ["created_at", "student", "email", "manual", "experiment", "question", "score", "feedback", "session_status"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download filtered report",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="hod_viva_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
