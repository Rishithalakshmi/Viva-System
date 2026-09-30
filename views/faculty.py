from __future__ import annotations

import pandas as pd
import streamlit as st

from models import User
from services.viva import (
    attempt_rows,
    create_manual_from_upload,
    delete_experiment,
    get_manual,
    list_manuals,
    set_manual_published,
    update_experiment,
    update_manual_settings,
)
from views.theme import banner


def render_faculty(user: User) -> None:
    banner(
        f"Faculty console · {user.full_name}",
        "Upload laboratory manuals, configure viva parameters (fullscreen exit limits, question counts), and monitor student performance.",
        user.email,
    )

    upload_tab, review_tab, results_tab = st.tabs(
        ["Upload manual", "Review & Configure", "Student attempts & Violations"]
    )

    # =========================================================
    # UPLOAD MANUAL
    # =========================================================
    with upload_tab:
        st.subheader("Laboratory manual")

        st.caption(
            "Upload one complete laboratory manual. "
            "Experiments are detected automatically from the document."
        )

        title = st.text_input(
            "Manual title",
            placeholder="Data Science Laboratory Manual",
        )

        uploaded = st.file_uploader(
            "Manual file",
            type=["pdf", "docx", "txt"],
        )

        if st.button(
            "Extract experiments",
            type="primary",
            use_container_width=True,
        ):
            if uploaded is None:
                st.error("Choose a manual file first.")
            else:
                status = st.empty()
                progress = st.progress(10)

                try:
                    status.info("Reading the complete laboratory manual...")
                    progress.progress(30)

                    data = uploaded.getvalue()

                    progress.progress(50)

                    manual = create_manual_from_upload(
                        user.id,
                        title.strip() or uploaded.name,
                        uploaded.name,
                        data,
                    )

                    progress.progress(100)

                    status.success(
                        "Manual processed successfully. "
                        "Review the detected experiments."
                    )

                    st.session_state["faculty_manual_id"] = manual.id

                    st.rerun()

                except Exception as exc:
                    progress.progress(100)
                    status.error("Manual extraction failed.")
                    st.exception(exc)

    # =========================================================
    # REVIEW AND CONFIGURE
    # =========================================================
    with review_tab:
        manuals = list_manuals(faculty_id=user.id)

        if not manuals:
            st.info("Upload a laboratory manual to begin.")

        else:
            options = {
                f"{item.title} "
                f"({'Published' if item.is_published else 'Draft'})": item.id
                for item in manuals
            }

            selected = st.selectbox(
                "Select manual",
                list(options.keys()),
            )

            manual = get_manual(options[selected])

            if manual is None:
                st.error("Unable to load the selected manual.")

            else:
                c1, c2, c3 = st.columns(3)

                with c1:
                    if st.button(
                        "Publish",
                        type="primary",
                        use_container_width=True,
                    ):
                        try:
                            set_manual_published(manual.id, True)
                            st.success(
                                "Manual published. "
                                "Students can now attempt these experiments."
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

                with c2:
                    if st.button(
                        "Unpublish",
                        use_container_width=True,
                    ):
                        try:
                            set_manual_published(manual.id, False)
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

                with c3:
                    st.caption("Published manuals are visible to students.")

                st.markdown("---")
                st.markdown("### Viva assessment settings")
                st.caption("Configure anti-cheating rule constraints and question count for this manual.")

                s_col1, s_col2, s_col3 = st.columns([2, 2, 1])
                with s_col1:
                    allowed_exits = st.number_input(
                        "Allowed fullscreen exits",
                        min_value=0,
                        max_value=10,
                        value=int(getattr(manual, "allowed_exits", 3) or 3),
                        help="Number of times a student can temporarily leave fullscreen before the exam automatically terminates.",
                        key=f"exits_{manual.id}",
                    )
                with s_col2:
                    q_count = st.number_input(
                        "Viva question count",
                        min_value=1,
                        max_value=10,
                        value=int(getattr(manual, "question_count", 5) or 5),
                        help="Number of viva questions generated per attempt.",
                        key=f"qcount_{manual.id}",
                    )
                with s_col3:
                    st.write("")
                    st.write("")
                    if st.button("Save settings", key=f"save_sett_{manual.id}", use_container_width=True):
                        try:
                            update_manual_settings(manual.id, allowed_exits, q_count)
                            st.success("Viva settings updated.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

                st.markdown("---")
                st.markdown("### Detected experiments")

                experiments = getattr(manual, "experiments", [])

                if not experiments:
                    st.warning("No experiments were detected in this manual.")

                for experiment in experiments:
                    with st.expander(
                        f"Experiment {experiment.number}: {experiment.title}"
                    ):
                        new_number = st.number_input(
                            "Experiment number",
                            min_value=1,
                            value=int(experiment.number),
                            key=f"number_{experiment.id}",
                        )

                        new_title = st.text_input(
                            "Experiment title",
                            value=experiment.title,
                            key=f"title_{experiment.id}",
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button(
                                "Save changes",
                                key=f"save_{experiment.id}",
                                use_container_width=True,
                            ):
                                try:
                                    update_experiment(
                                        experiment.id,
                                        number=new_number,
                                        title=new_title,
                                        content=experiment.content,
                                    )
                                    st.success("Experiment updated.")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(str(exc))

                        with col2:
                            if st.button(
                                "Delete experiment",
                                key=f"delete_{experiment.id}",
                                use_container_width=True,
                            ):
                                try:
                                    delete_experiment(experiment.id)
                                    st.success("Experiment deleted.")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(str(exc))

    # =========================================================
    # STUDENT ATTEMPTS & VIOLATIONS
    # =========================================================
    with results_tab:
        rows = attempt_rows(faculty_id=user.id)

        if not rows:
            st.info("No student attempts recorded yet.")
        else:
            frame = pd.DataFrame(rows)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Recorded answers", len(frame))

            if "email" in frame.columns:
                c2.metric("Students", frame["email"].nunique())
            else:
                c2.metric("Students", 0)

            if "score" in frame.columns and not frame.empty:
                c3.metric("Average mark", f"{frame['score'].mean():.2f}/10")
            else:
                c3.metric("Average mark", "0.00/10")

            if "session_status" in frame.columns:
                terminated_count = frame[frame["session_status"] == "TERMINATED"]["university_id"].nunique()
                c4.metric("Terminated vivas", terminated_count)

            st.markdown("### Experiment performance")

            if "experiment" in frame.columns and "score" in frame.columns:
                performance = (
                    frame.groupby("experiment")
                    .agg(
                        Attempts=("score", "count"),
                        Average=("score", "mean"),
                        Highest=("score", "max"),
                        Lowest=("score", "min"),
                    )
                    .reset_index()
                )

                st.dataframe(
                    performance,
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("### Attempts, violations & evaluation feedback")

            display_columns = [
                "created_at",
                "student",
                "university_id",
                "experiment",
                "question",
                "score",
                "session_status",
                "exit_count",
                "allowed_exits",
                "terminated_reason",
                "feedback",
            ]

            available_columns = [
                column for column in display_columns if column in frame.columns
            ]

            st.dataframe(
                frame[available_columns] if available_columns else frame,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Export results CSV",
                frame.to_csv(index=False).encode("utf-8"),
                file_name="viva_results.csv",
                mime="text/csv",
                use_container_width=True,
            )