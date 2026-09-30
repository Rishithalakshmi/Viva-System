from __future__ import annotations

import json
import pathlib
import tempfile
import time
import pandas as pd
import pyttsx3
import streamlit as st
import streamlit.components.v1 as components

from config import MAX_MARKS_PER_QUESTION, MAX_MARKS_TOTAL, QUESTIONS_PER_VIVA, UNIVERSITY
from models import User
from services.asr import DEFAULT_AUDIO_DEVICE, record_audio_dshow, transcribe_audio
from services.llm import clean_question_text
from services.viva import (
    audio_file_path,
    get_open_session,
    get_session,
    published_experiments,
    record_fullscreen_exit,
    start_viva_session,
    student_sessions,
    submit_answer,
    terminate_viva_session,
)
from utils.text import format_experiment_title, load_json_list
from views.theme import banner


def speak_question_bytes(question_text: str) -> bytes:
    if not question_text:
        return b""
    try:
        temp_file = pathlib.Path(tempfile.gettempdir()) / f"viva_q_{abs(hash(question_text))}.wav"
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.save_to_file(question_text, str(temp_file))
        engine.runAndWait()
        if temp_file.exists():
            data = temp_file.read_bytes()
            try:
                temp_file.unlink()
            except Exception:
                pass
            return data
    except Exception:
        pass
    return b""


def _reset_answer_state() -> None:
    for key in ("transcript", "audio_path", "last_audio_hash", "eval_payload", "status_note"):
        st.session_state.pop(key, None)


def _current_open_question(session):
    for question in session.questions:
        if question.answer is None:
            return question
    return None


def _inject_viva_security_and_fullscreen_js(session_id: int, allowed_exits: int, current_exits: int, is_completed: bool = False):
    """
    Inject browser-side JavaScript for:
    1. Anti-cheating: Disable copy/paste/cut/contextmenu on answer textarea
    2. Fullscreen enforcement and 10-second exit countdown system
    3. Automatic fullscreen exit on examination completion
    """
    is_comp_js = "true" if is_completed else "false"
    js_code = f"""
    <script>
    (function() {{
        const parentDoc = window.parent.document;
        const sessionId = {session_id};
        const allowedExits = {allowed_exits};
        const isCompleted = {is_comp_js};
        
        // Automatic Exit Fullscreen when exam legitimately completes
        if (isCompleted) {{
            if (parentDoc.fullscreenElement) {{
                try {{ parentDoc.exitFullscreen().catch(() => {{}}); }} catch(e) {{}}
            }}
            const modal = parentDoc.getElementById('viva-exit-warning-modal');
            if (modal) modal.style.display = 'none';
            parentDoc.onfullscreenchange = null;
            parentDoc.onvisibilitychange = null;
            return;
        }}

        // 1. Anti-cheating Copy-Paste Prevention
        function disableCopyPaste() {{
            const textareas = parentDoc.querySelectorAll('textarea');
            textareas.forEach(el => {{
                if (!el.dataset.vivaProtected) {{
                    el.dataset.vivaProtected = "true";
                    el.addEventListener('paste', function(e) {{
                        e.preventDefault();
                        alert("Clipboard paste is disabled during the live viva examination. Please type or speak your response directly.");
                    }});
                    el.addEventListener('copy', function(e) {{ e.preventDefault(); }});
                    el.addEventListener('cut', function(e) {{ e.preventDefault(); }});
                    el.addEventListener('contextmenu', function(e) {{ e.preventDefault(); }});
                }}
            }});
        }}
        
        parentDoc.addEventListener('keydown', function(e) {{
            if ((e.ctrlKey || e.metaKey) && (e.key === 'v' || e.key === 'V' || e.key === 'c' || e.key === 'C' || e.key === 'x' || e.key === 'X')) {{
                const activeEl = parentDoc.activeElement;
                if (activeEl && activeEl.tagName === 'TEXTAREA') {{
                    e.preventDefault();
                    alert("Clipboard shortcuts (Ctrl+C, Ctrl+V, Ctrl+X) are disabled during the viva exam.");
                }}
            }}
        }}, true);

        setInterval(disableCopyPaste, 800);

        // 2. Fullscreen Exit Detection & 10-Second Countdown Warning
        let warningModal = parentDoc.getElementById('viva-exit-warning-modal');
        if (!warningModal) {{
            warningModal = parentDoc.createElement('div');
            warningModal.id = 'viva-exit-warning-modal';
            warningModal.style.cssText = 'display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(11,31,58,0.92); z-index:999999; color:#fff; text-align:center; padding-top:15vh; font-family:sans-serif;';
            warningModal.innerHTML = `
                <div style="background:#fff; color:#0b1f3a; max-width:550px; margin:0 auto; padding:32px; border-radius:18px; box-shadow:0 20px 50px rgba(0,0,0,0.5);">
                    <h2 style="color:#7a1f2b; margin-top:0;">⚠️ FULLSCREEN VIOLATION DETECTED</h2>
                    <p style="font-size:1.1rem;">You have exited distraction-free fullscreen mode.</p>
                    <p style="font-weight:bold; font-size:1.2rem; color:#7a1f2b;">Return to fullscreen within <span id="viva-countdown-num" style="font-size:2rem; color:#0b1f3a;">10</span> seconds or your examination will be automatically terminated.</p>
                    <button id="viva-return-fs-btn" style="background:#0b1f3a; color:#fff; font-size:1.1rem; font-weight:bold; padding:12px 28px; border:none; border-radius:10px; cursor:pointer; margin-top:15px;">Return to Fullscreen</button>
                </div>
            `;
            parentDoc.body.appendChild(warningModal);
        }}

        const countdownSpan = parentDoc.getElementById('viva-countdown-num');
        const returnBtn = parentDoc.getElementById('viva-return-fs-btn');
        let countdownTimer = null;
        let secondsLeft = 10;
        let isHandlingExit = false;

        function triggerExitViolation() {{
            if (isHandlingExit || isCompleted) return;
            isHandlingExit = true;
            secondsLeft = 10;
            if (countdownSpan) countdownSpan.innerText = secondsLeft;
            if (warningModal) warningModal.style.display = 'block';

            if (countdownTimer) clearInterval(countdownTimer);
            countdownTimer = setInterval(() => {{
                secondsLeft -= 1;
                if (countdownSpan) countdownSpan.innerText = secondsLeft;
                if (secondsLeft <= 0) {{
                    clearInterval(countdownTimer);
                    // Notify Streamlit server of termination request
                    const params = new URLSearchParams(window.parent.location.search);
                    params.set('viva_exit_session', sessionId);
                    params.set('viva_action', 'terminate');
                    window.parent.location.search = params.toString();
                }}
            }}, 1000);
        }}

        if (returnBtn) {{
            returnBtn.onclick = function() {{
                if (countdownTimer) clearInterval(countdownTimer);
                if (warningModal) warningModal.style.display = 'none';
                isHandlingExit = false;
                try {{
                    parentDoc.documentElement.requestFullscreen().catch(() => {{}});
                }} catch(e) {{}}
                // Log violation count on backend
                const params = new URLSearchParams(window.parent.location.search);
                params.set('viva_exit_session', sessionId);
                params.set('viva_action', 'record_exit');
                window.parent.location.search = params.toString();
            }};
        }}

        parentDoc.onfullscreenchange = function() {{
            if (!parentDoc.fullscreenElement && !isCompleted) {{
                triggerExitViolation();
            }}
        }};
        
        parentDoc.onvisibilitychange = function() {{
            if (parentDoc.hidden && !isCompleted) {{
                triggerExitViolation();
            }}
        }};
    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)


def render_student(user: User) -> None:
    # Process incoming JS fullscreen violation events via query params if present
    query_params = st.query_params
    if "viva_exit_session" in query_params:
        session_id_val = int(query_params["viva_exit_session"])
        action = query_params.get("viva_action", "record_exit")
        st.query_params.clear()

        if action == "terminate":
            terminate_viva_session(session_id_val, "Exceeded 10-second fullscreen return countdown limit.")
            st.warning("Viva session terminated due to fullscreen exit countdown timeout.")
        else:
            res = record_fullscreen_exit(session_id_val)
            if res.get("status") == "TERMINATED":
                st.error(f"Viva session terminated: {res.get('reason')}")
        st.rerun()

    active_session_id = st.session_state.get("viva_session_id")
    active_session = get_session(active_session_id) if active_session_id else None

    # Check if student is in an active viva session
    is_in_viva = (
        active_session is not None
        and active_session.status == "IN_PROGRESS"
        and st.session_state.get("viva_mode_started", False)
    )

    st.session_state.viva_active = is_in_viva

    # =========================================================================
    # DISTRACTION-FREE VIVA MODE (Requirements 1, 2, 3, 5, 6, 7, 8)
    # =========================================================================
    if is_in_viva and active_session:
        _render_distraction_free_viva(user, active_session)
        return

    # Standard Student Workspace View
    banner(
        f"Student workspace · {user.full_name}",
        "Select a published experiment, complete an oral viva assessment with AI voice, and review your results.",
        f"ID {user.university_id} · {user.email}",
    )

    profile, viva, history = st.tabs(["Profile", "Oral viva", "Previous attempts"])

    with profile:
        c1, c2, c3, c4 = st.columns(4)
        sessions = student_sessions(user.id)
        completed = [item for item in sessions if item.status == "COMPLETED"]
        in_progress = [item for item in sessions if item.status == "IN_PROGRESS"]
        avg = (
            sum(item.total_score for item in completed) / len(completed)
            if completed
            else 0.0
        )
        c1.metric("University ID", user.university_id)
        c2.metric("Completed vivas", len(completed))
        c3.metric("In progress", len(in_progress))
        c4.metric("Average total", f"{avg:.1f}/{MAX_MARKS_TOTAL}")
        st.markdown("#### Available laboratory manuals")
        experiments = published_experiments()
        if not experiments:
            st.info("No published laboratory manuals are available yet. Please wait for faculty to publish.")
        else:
            rows = [
                {
                    "Manual": item.manual.title,
                    "Experiment": f"Experiment {item.number}",
                    "Topic": format_experiment_title(item.title, item.number),
                    "Faculty": item.manual.faculty.full_name if item.manual.faculty else "",
                }
                for item in experiments
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with viva:
        experiments = published_experiments()
        if not experiments:
            st.info("Faculty have not published a laboratory manual yet.")
        else:
            labels = {
                f"Experiment {item.number} — {format_experiment_title(item.title, item.number)}": item.id
                for item in experiments
            }
            selected_label = st.selectbox("Select experiment", list(labels.keys()))
            experiment_id = labels[selected_label]
            if st.session_state.get("selected_experiment_id") != experiment_id:
                st.session_state.selected_experiment_id = experiment_id
                st.session_state.pop("viva_session_id", None)
                st.session_state.viva_mode_started = False
                _reset_answer_state()

            open_session = get_open_session(user.id, experiment_id)
            if open_session and open_session.questions:
                st.session_state.viva_session_id = open_session.id

            col_a, col_b = st.columns([1, 2])
            with col_a:
                start = st.button("🚀 Start / Resume Viva", type="primary", use_container_width=True)
            with col_b:
                st.caption("AI Voice speaks each question. Dedicated distraction-free fullscreen viva mode will launch.")

            start_failed = False
            if start:
                progress = st.progress(15)
                note = st.empty()
                note.info("Initializing viva session & generating questions with AI voice...")
                try:
                    progress.progress(50)
                    session = start_viva_session(user.id, experiment_id)
                    progress.progress(100)
                    st.session_state.viva_session_id = session.id
                    st.session_state.viva_mode_started = True
                    _reset_answer_state()
                    note.success("Viva session ready! Entering fullscreen mode...")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as exc:
                    progress.progress(100)
                    note.error(f"Viva initialization failed: {exc}")
                    st.error(str(exc))
                    start_failed = True

            session_id = st.session_state.get("viva_session_id")
            if not start_failed and session_id:
                session = get_session(session_id)
                if session and session.status == "TERMINATED":
                    st.error(f"❌ This viva session was terminated: {session.terminated_reason or 'Fullscreen violation'}")
                    st.write(f"**Fullscreen exits recorded:** {session.exit_count} / {session.allowed_exits}")

    with history:
        sessions = student_sessions(user.id)
        if not sessions:
            st.caption("No viva attempts recorded yet.")
        else:
            rows = []
            for item in sessions:
                rows.append(
                    {
                        "Started": item.started_at.strftime("%Y-%m-%d %H:%M"),
                        "Manual": item.experiment.manual.title,
                        "Experiment": f"Experiment {item.experiment.number} — {format_experiment_title(item.experiment.title, item.experiment.number)}",
                        "Status": item.status.replace("_", " ").title(),
                        "Total Score": f"{item.total_score:.1f}/{item.max_score:.0f}",
                        "Exits / Allowed": f"{item.exit_count} / {item.allowed_exits}",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            for item in sessions:
                with st.expander(f"Experiment {item.experiment.number} — {format_experiment_title(item.experiment.title, item.experiment.number)} · {item.status}"):
                    if item.status == "TERMINATED":
                        st.error(f"Session Terminated: {item.terminated_reason}")
                    for question in item.questions:
                        clean_q = clean_question_text(question.question_text)
                        st.markdown(f"**Q{question.order_index}.** {clean_q}")
                        if question.answer:
                            st.write(f"**Transcript:** {question.answer.transcript}")
                            st.write(f"**Score:** {question.answer.score}/{MAX_MARKS_PER_QUESTION}")
                            st.write(f"**Feedback:** {question.answer.feedback}")
                        else:
                            st.caption("Not answered.")
                        st.divider()


def _render_distraction_free_viva(user: User, session) -> None:
    """Renders distraction-free viva mode with immediate per-question evaluation & final summary."""
    showing_eval_q_id = st.session_state.get("showing_eval_for_q")
    eval_q = None
    if showing_eval_q_id:
        eval_q = next((q for q in session.questions if q.id == showing_eval_q_id), None)

    current = _current_open_question(session)

    # 1. PER-QUESTION EVALUATION VIEW (Req 1)
    if eval_q and eval_q.answer:
        _render_question_evaluation_view(session, eval_q)
        return

    # 2. FINAL VIVA RESULT VIEW (Req 2 & 3)
    if current is None:
        _render_final_viva_result_view(user, session)
        return

    # 3. ACTIVE QUESTION INPUT VIEW
    _inject_viva_security_and_fullscreen_js(session.id, session.allowed_exits, session.exit_count, is_completed=False)

    # Top Distraction-free Header
    h_col1, h_col2, h_col3 = st.columns([3, 2, 1])
    with h_col1:
        st.markdown(f"### 🧪 Ex {session.experiment.number}: {format_experiment_title(session.experiment.title, session.experiment.number)}")
        st.caption(f"{UNIVERSITY} · Viva Examination Mode")
    with h_col2:
        answered = sum(1 for q in session.questions if q.answer)
        total_q = len(session.questions)
        st.progress(answered / max(1, total_q))
        st.caption(f"Question {current.order_index} of {total_q} · Max Marks: {MAX_MARKS_PER_QUESTION}")
    with h_col3:
        exits_left = max(0, session.allowed_exits - session.exit_count)
        st.metric("Allowed Exits Left", f"{exits_left}/{session.allowed_exits}")

    st.divider()

    # Question Display & AI TTS Voice Controls
    clean_q = clean_question_text(current.question_text)
    
    st.markdown(f"#### ❓ Question {current.order_index}")
    st.markdown(f"<div style='background:#0b1f3a; color:#ffffff; padding:20px 24px; border-radius:14px; font-size:1.25rem; font-weight:600; line-height:1.5;'>{clean_q}</div>", unsafe_allow_html=True)
    
    speech_key = f"speech_{current.id}"
    if speech_key not in st.session_state:
        st.session_state[speech_key] = speak_question_bytes(clean_q)

    speech_bytes = st.session_state.get(speech_key)

    v_col1, v_col2 = st.columns([4, 1])
    with v_col1:
        if speech_bytes:
            st.audio(speech_bytes, format="audio/wav", autoplay=True)
    with v_col2:
        if st.button("🔊 Replay Question", key=f"replay_{current.id}", use_container_width=True):
            st.session_state[speech_key] = speak_question_bytes(clean_q)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Microphone Recording Controls & Transcript Editing
    st.markdown("#### 🎤 Answer Input (Voice or Typed)")
    st.info("🔒 **Anti-Cheating Active**: Copying and pasting into the answer box is disabled. Type your answer or use voice recording.")

    r_col1, r_col2, r_col3 = st.columns([1, 2, 1])
    with r_col1:
        duration = st.selectbox("Recording duration", [15, 30, 45, 60], index=1, key=f"dur_{current.id}")
    with r_col2:
        record_btn = st.button("🎤 Record Voice Answer", type="primary", use_container_width=True, key=f"rec_{current.id}")
    with r_col3:
        if st.session_state.get("transcript"):
            if st.button("🎤 Record Again", use_container_width=True, key=f"re_rec_{current.id}"):
                _reset_answer_state()
                st.rerun()

    if record_btn:
        status_box = st.empty()
        progress_box = st.progress(10)
        try:
            status_box.info(f"Recording for {duration} seconds... Speak clearly into microphone.")
            wav_path = audio_file_path(user.id)
            record_audio_dshow(wav_path, duration=duration, device_name=DEFAULT_AUDIO_DEVICE)
            progress_box.progress(55)

            status_box.info("Preprocessing audio & transcribing with Nemotron ASR...")
            transcript_text = transcribe_audio(wav_path, session.experiment.content)
            progress_box.progress(100)

            st.session_state.audio_path = str(wav_path)
            st.session_state.transcript = transcript_text
            box_key = f"transcript_box_{current.id}"
            st.session_state[box_key] = transcript_text
            status_box.success("Voice transcribed! Review and correct any ASR mistakes below before submitting.")
            st.rerun()
        except Exception as exc:
            progress_box.progress(100)
            status_box.error(f"Recording/transcription failed: {exc}")

    box_key = f"transcript_box_{current.id}"
    if box_key not in st.session_state:
        st.session_state[box_key] = st.session_state.get("transcript", "")

    user_answer = st.text_area(
        "Student Answer (Type directly or review voice transcript):",
        height=130,
        key=box_key,
        placeholder="Type your response or click 'Record Voice Answer' above...",
    )

    if st.button("📤 Submit Answer for Evaluation", type="primary", use_container_width=True, key=f"sub_{current.id}"):
        final_text = str(user_answer).strip()
        if not final_text:
            st.error("Please provide a spoken or typed answer before submitting.")
        else:
            eval_status = st.empty()
            eval_bar = st.progress(20)
            eval_status.info("Evaluating answer against experiment manual concepts...")
            try:
                eval_bar.progress(60)
                submit_answer(
                    current.id,
                    final_text,
                    st.session_state.get("audio_path"),
                )
                eval_bar.progress(100)
                eval_status.success("Answer evaluated successfully.")
                _reset_answer_state()
                st.session_state.pop(box_key, None)
                st.session_state.showing_eval_for_q = current.id
                st.rerun()
            except Exception as exc:
                eval_bar.progress(100)
                eval_status.error(f"Evaluation failed: {exc}")


def _render_question_evaluation_view(session, eval_q) -> None:
    """Renders immediate per-question evaluation breakdown immediately after answer submission (Req 1)."""
    ans = eval_q.answer
    clean_q = clean_question_text(eval_q.question_text)
    expected_list = load_json_list(eval_q.expected_concepts)
    matched_list = load_json_list(ans.matched_concepts)
    missing_list = load_json_list(ans.missing_concepts)

    _inject_viva_security_and_fullscreen_js(session.id, session.allowed_exits, session.exit_count, is_completed=False)

    st.markdown(f"### 📊 Question {eval_q.order_index} — Score: {ans.score:.1f}/{MAX_MARKS_PER_QUESTION}")
    
    # Score Metric Card
    sc_col1, sc_col2 = st.columns([1, 3])
    with sc_col1:
        st.metric("Marks Awarded", f"{ans.score:.1f} / {MAX_MARKS_PER_QUESTION}")
    with sc_col2:
        if ans.score >= 8.0:
            st.success("🌟 Excellent! Strong understanding of core experiment concepts.")
        elif ans.score >= 5.0:
            st.info("👍 Good effort. Core concepts covered with partial marks awarded.")
        else:
            st.warning("⚠️ Partial marks awarded. Fundamental concepts missing.")

    st.markdown(f"<div style='background:#f4efe4; border-left:4px solid #0b1f3a; padding:14px 18px; margin:12px 0; border-radius:8px;'><strong>Question:</strong> {clean_q}</div>", unsafe_allow_html=True)
    st.markdown(f"**Your Submitted Answer:** {ans.transcript}")
    
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 💡 Why This Answer Received These Marks")
        st.info(ans.feedback or "Evaluated using semantic concept matching.")

        st.markdown("#### ✓ Concepts Covered in Your Answer")
        if matched_list:
            for item in matched_list:
                st.markdown(f"- ✅ **{item}**")
        else:
            st.caption("No key concepts matched.")

        st.markdown("#### ✗ Missing or Incorrect Concepts")
        if missing_list:
            for item in missing_list:
                st.markdown(f"- ❌ **{item}**")
        else:
            st.caption("No major missing concepts recorded.")

    with c2:
        st.markdown("#### 🎯 Expected Answer & Core Concepts")
        st.write("A complete and strong answer for this question should explain:")
        if expected_list:
            for item in expected_list:
                st.markdown(f"- 📌 **{item}**")
        else:
            st.markdown("- 📌 Practical implementation steps, dataset parameters, and model evaluation metrics.")

        st.markdown("#### 📝 Improvement Guidance")
        st.success("Connect your explanation directly to the theoretical principles and workflow in the laboratory manual.")

    st.divider()

    next_open = _current_open_question(session)
    if next_open is not None:
        if st.button("Next Question ➔", type="primary", use_container_width=True, key=f"next_q_btn_{eval_q.id}"):
            st.session_state.pop("showing_eval_for_q", None)
            _reset_answer_state()
            st.rerun()
    else:
        if st.button("View Final Results ➔", type="primary", use_container_width=True, key=f"view_final_btn_{eval_q.id}"):
            st.session_state.pop("showing_eval_for_q", None)
            _reset_answer_state()
            st.rerun()


def _render_final_viva_result_view(user: User, session) -> None:
    """Renders comprehensive final viva result view & automatically exits fullscreen (Req 2 & 3)."""
    # Inject JS to automatically exit browser fullscreen cleanly & disable violation warning
    _inject_viva_security_and_fullscreen_js(session.id, session.allowed_exits, session.exit_count, is_completed=True)
    
    st.session_state.viva_active = False

    banner(
        f"Oral Viva Evaluation Complete · {session.experiment.title}",
        "Final total score, question-wise breakdown, overall strengths, weak concepts, and improvement feedback.",
        f"{UNIVERSITY} · Final Viva Result",
    )

    r_col1, r_col2, r_col3 = st.columns(3)
    r_col1.metric("Final Viva Score", f"{session.total_score:.1f} / {session.max_score:.0f}")
    r_col2.metric("Questions Answered", len(session.questions))
    r_col3.metric("Status", session.status.replace("_", " ").title())

    st.divider()

    # Aggregated Strengths & Weaknesses across all 5 questions
    all_matched = []
    all_missing = []
    for q in session.questions:
        if q.answer:
            all_matched.extend(load_json_list(q.answer.matched_concepts))
            all_missing.extend(load_json_list(q.answer.missing_concepts))
    
    all_matched = list(dict.fromkeys(all_matched))
    all_missing = list(dict.fromkeys(all_missing))

    c_str, c_weak = st.columns(2)
    with c_str:
        st.markdown("#### 💪 Key Strengths (Covered Concepts)")
        if all_matched:
            for item in all_matched[:10]:
                st.markdown(f"- ✅ **{item}**")
        else:
            st.caption("No key concepts recorded.")

    with c_weak:
        st.markdown("#### 🎯 Weak Concepts (Areas to Improve)")
        if all_missing:
            for item in all_missing[:10]:
                st.markdown(f"- ❌ **{item}**")
        else:
            st.caption("No major weaknesses recorded.")

    st.divider()
    st.markdown("#### 📝 Overall Assessment Feedback")
    percentage = (session.total_score / session.max_score) * 100.0 if session.max_score > 0 else 0
    if percentage >= 80:
        st.success("🌟 Excellent performance! You demonstrated a comprehensive understanding of the experiment, algorithms, and evaluation metrics.")
    elif percentage >= 60:
        st.info("👍 Good effort. You understand the core experimental workflow. Review parameter interpretations to strengthen your understanding.")
    else:
        st.warning("⚠️ Partial marks awarded. Please review the laboratory manual theory, algorithm steps, and parameter choices.")

    st.divider()
    st.markdown("#### 📋 Question-by-Question Score Breakdown")
    for q in session.questions:
        clean_q = clean_question_text(q.question_text)
        score_val = q.answer.score if q.answer else 0.0
        with st.expander(f"Question {q.order_index} — Score: {score_val:.1f}/{MAX_MARKS_PER_QUESTION}"):
            st.write(f"**Question:** {clean_q}")
            if q.answer:
                st.write(f"**Your Answer:** {q.answer.transcript}")
                st.write(f"**Score:** {q.answer.score:.1f}/{MAX_MARKS_PER_QUESTION}")
                st.write(f"**Why:** {q.answer.feedback}")
                st.write(f"**Covered:** {', '.join(load_json_list(q.answer.matched_concepts)) or 'None'}")
                st.write(f"**Missing:** {', '.join(load_json_list(q.answer.missing_concepts)) or 'None'}")

    st.divider()
    col_exit1, col_exit2 = st.columns([1, 2])
    with col_exit1:
        if st.button("🏁 Finish & Return to Dashboard", type="primary", use_container_width=True):
            st.session_state.pop("viva_session_id", None)
            st.session_state.viva_mode_started = False
            _reset_answer_state()
            st.rerun()
    with col_exit2:
        st.caption("Clicking Finish clears the active viva view and returns to your student profile.")
