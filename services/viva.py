from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import joinedload

from config import AUDIO_DIR, MANUAL_DIR, MAX_MARKS_PER_QUESTION, MAX_MARKS_TOTAL, QUESTIONS_PER_VIVA
from database import session_scope
from models import Experiment, Manual, Result, User, VivaAnswer, VivaQuestion, VivaSession
from services.llm import evaluate_answer, generate_viva_questions
from services.manual import detect_experiments, extract_manual_text
from utils.text import load_json_list


def create_manual_from_upload(faculty_id: int, title: str, filename: str, data: bytes) -> Manual:
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in filename)
    stored = MANUAL_DIR / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    stored.write_bytes(data)
    text = extract_manual_text(stored, raw=data)
    detected = detect_experiments(text)
    if not detected:
        raise RuntimeError("No readable text was extracted from the manual.")
    with session_scope() as db:
        manual = Manual(
            title=title.strip() or Path(filename).stem,
            filename=filename,
            stored_path=str(stored),
            extracted_text=text,
            faculty_id=faculty_id,
            is_published=False,
        )
        db.add(manual)
        db.flush()
        for item in detected:
            db.add(
                Experiment(
                    manual_id=manual.id,
                    number=int(item["number"]),
                    title=item["title"],
                    content=item["content"],
                )
            )
        db.refresh(manual)
        db.expunge(manual)
        return manual


def list_manuals(published_only: bool = False, faculty_id: int | None = None) -> list[Manual]:
    with session_scope() as db:
        query = db.query(Manual).options(joinedload(Manual.experiments), joinedload(Manual.faculty))
        if published_only:
            query = query.filter(Manual.is_published.is_(True))
        if faculty_id is not None:
            query = query.filter(Manual.faculty_id == faculty_id)
        manuals = query.order_by(Manual.created_at.desc()).all()
        for manual in manuals:
            manual.experiments
            manual.faculty
            db.expunge(manual)
        return manuals


def get_manual(manual_id: int) -> Manual | None:
    with session_scope() as db:
        manual = (
            db.query(Manual)
            .options(joinedload(Manual.experiments), joinedload(Manual.faculty))
            .filter(Manual.id == manual_id)
            .one_or_none()
        )
        if manual:
            _ = list(manual.experiments)
            db.expunge(manual)
        return manual


def update_experiment(experiment_id: int, title: str, content: str, number: int | None = None) -> None:
    with session_scope() as db:
        experiment = db.get(Experiment, experiment_id)
        if not experiment:
            raise RuntimeError("Experiment not found.")
        experiment.title = title.strip() or experiment.title
        experiment.content = content.strip()
        if number is not None:
            experiment.number = int(number)
        experiment.manual.updated_at = datetime.utcnow()


def add_experiment(manual_id: int, title: str, content: str) -> None:
    with session_scope() as db:
        manual = db.get(Manual, manual_id)
        if not manual:
            raise RuntimeError("Manual not found.")
        next_number = (db.query(func.max(Experiment.number)).filter(Experiment.manual_id == manual_id).scalar() or 0) + 1
        db.add(
            Experiment(
                manual_id=manual_id,
                number=next_number,
                title=title.strip() or f"Experiment {next_number}",
                content=content.strip(),
            )
        )
        manual.updated_at = datetime.utcnow()


def delete_experiment(experiment_id: int) -> None:
    with session_scope() as db:
        experiment = db.get(Experiment, experiment_id)
        if not experiment:
            return
        if experiment.sessions:
            raise RuntimeError("This experiment already has viva attempts and cannot be deleted.")
        db.delete(experiment)


def set_manual_published(manual_id: int, published: bool) -> None:
    with session_scope() as db:
        manual = db.get(Manual, manual_id)
        if not manual:
            raise RuntimeError("Manual not found.")
        if published and not manual.experiments:
            raise RuntimeError("Publish at least one experiment.")
        manual.is_published = published
        manual.updated_at = datetime.utcnow()


def published_experiments() -> list[Experiment]:
    with session_scope() as db:
        rows = (
            db.query(Experiment)
            .join(Manual)
            .options(joinedload(Experiment.manual).joinedload(Manual.faculty))
            .filter(Manual.is_published.is_(True))
            .order_by(Manual.title, Experiment.number)
            .all()
        )
        for row in rows:
            _ = row.manual.faculty
            db.expunge(row)
        return rows


def get_experiment(experiment_id: int) -> Experiment | None:
    with session_scope() as db:
        experiment = (
            db.query(Experiment)
            .options(joinedload(Experiment.manual).joinedload(Manual.faculty))
            .filter(Experiment.id == experiment_id)
            .one_or_none()
        )
        if experiment:
            _ = experiment.manual.faculty
            db.expunge(experiment)
        return experiment


def get_open_session(student_id: int, experiment_id: int) -> VivaSession | None:
    with session_scope() as db:
        session = (
            db.query(VivaSession)
            .options(joinedload(VivaSession.questions).joinedload(VivaQuestion.answer))
            .filter(
                VivaSession.student_id == student_id,
                VivaSession.experiment_id == experiment_id,
                VivaSession.status == "IN_PROGRESS",
            )
            .order_by(VivaSession.id.desc())
            .first()
        )
        if session:
            for question in session.questions:
                question.answer
            db.expunge(session)
        return session


def get_session(session_id: int) -> VivaSession | None:
    with session_scope() as db:
        session = (
            db.query(VivaSession)
            .options(
                joinedload(VivaSession.questions).joinedload(VivaQuestion.answer),
                joinedload(VivaSession.experiment).joinedload(Experiment.manual),
                joinedload(VivaSession.student),
            )
            .filter(VivaSession.id == session_id)
            .one_or_none()
        )
        if session:
            session.experiment.manual
            session.student
            for question in session.questions:
                question.answer
            db.expunge(session)
        return session


def update_manual_settings(manual_id: int, allowed_exits: int, question_count: int = QUESTIONS_PER_VIVA) -> None:
    with session_scope() as db:
        manual = db.get(Manual, manual_id)
        if not manual:
            raise RuntimeError("Manual not found.")
        manual.allowed_exits = max(0, int(allowed_exits))
        manual.question_count = max(1, min(10, int(question_count)))
        manual.updated_at = datetime.utcnow()


def record_fullscreen_exit(session_id: int) -> dict:
    """Record a fullscreen exit event for an active viva session."""
    with session_scope() as db:
        session = db.get(VivaSession, session_id)
        if not session:
            return {"status": "UNKNOWN", "exit_count": 0, "allowed_exits": 3}
        if session.status != "IN_PROGRESS":
            return {
                "status": session.status,
                "exit_count": session.exit_count,
                "allowed_exits": session.allowed_exits,
                "reason": session.terminated_reason,
            }

        session.exit_count = (session.exit_count or 0) + 1
        violations = load_json_list(session.exit_violations)
        violations.append(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        session.exit_violations = json.dumps(violations)

        if session.exit_count > session.allowed_exits:
            session.status = "TERMINATED"
            session.terminated_reason = f"Exceeded maximum allowed fullscreen exits ({session.allowed_exits})"
            session.completed_at = datetime.utcnow()

        db.flush()
        return {
            "status": session.status,
            "exit_count": session.exit_count,
            "allowed_exits": session.allowed_exits,
            "reason": session.terminated_reason,
        }


def terminate_viva_session(session_id: int, reason: str = "Fullscreen exit countdown expired") -> None:
    """Terminate an active viva session due to rule violation or timeout."""
    with session_scope() as db:
        session = db.get(VivaSession, session_id)
        if session and session.status == "IN_PROGRESS":
            session.status = "TERMINATED"
            session.terminated_reason = reason
            session.completed_at = datetime.utcnow()


def start_viva_session(student_id: int, experiment_id: int) -> VivaSession:
    existing = get_open_session(student_id, experiment_id)
    if existing and existing.questions and len(existing.questions) == QUESTIONS_PER_VIVA:
        return existing

    experiment = get_experiment(experiment_id)
    if not experiment or not experiment.manual.is_published:
        raise RuntimeError("This experiment is not available for viva assessment.")

    allowed = getattr(experiment.manual, "allowed_exits", 3) or 3
    q_count = getattr(experiment.manual, "question_count", QUESTIONS_PER_VIVA) or QUESTIONS_PER_VIVA

    questions = generate_viva_questions(experiment.content, q_count)
    if len(questions) != q_count:
        raise RuntimeError(f"Question generation failed: expected {q_count} questions, got {len(questions)}.")

    with session_scope() as db:
        if existing:
            session = db.get(VivaSession, existing.id)
            if session:
                db.query(VivaQuestion).filter(VivaQuestion.session_id == session.id).delete()
        else:
            session = VivaSession(
                student_id=student_id,
                experiment_id=experiment_id,
                status="IN_PROGRESS",
                allowed_exits=allowed,
                exit_count=0,
                exit_violations="[]",
                max_score=float(q_count * MAX_MARKS_PER_QUESTION),
            )
            db.add(session)
            db.flush()

        for index, item in enumerate(questions, start=1):
            db.add(
                VivaQuestion(
                    session_id=session.id,
                    order_index=index,
                    question_text=item["question"],
                    expected_concepts=json.dumps(item["expected_concepts"]),
                    difficulty=item["difficulty"],
                )
            )
        db.refresh(session)
        session_id = session.id

    loaded = get_session(session_id)
    if not loaded:
        raise RuntimeError("The viva session could not be initialized.")
    return loaded


def submit_answer(question_id: int, transcript: str, audio_path: str | None) -> VivaAnswer:
    with session_scope() as db:
        question = (
            db.query(VivaQuestion)
            .options(joinedload(VivaQuestion.session).joinedload(VivaSession.experiment))
            .filter(VivaQuestion.id == question_id)
            .one_or_none()
        )
        if not question:
            raise RuntimeError("Question not found.")
        if question.answer:
            raise RuntimeError("This question has already been evaluated.")
        question_text = question.question_text
        expected = load_json_list(question.expected_concepts)
        content = question.session.experiment.content
        session_id = question.session_id
        student_id = question.session.student_id
        experiment_id = question.session.experiment_id

    evaluation = evaluate_answer(
        question_text,
        transcript,
        content,
        expected,
        MAX_MARKS_PER_QUESTION,
    )
    score = max(0.0, min(float(MAX_MARKS_PER_QUESTION), float(evaluation["score"])))

    with session_scope() as db:
        question = db.get(VivaQuestion, question_id)
        if question is None:
            raise RuntimeError("Question not found.")
        if question.answer:
            raise RuntimeError("This question has already been evaluated.")

        answer = VivaAnswer(
            question_id=question_id,
            audio_path=audio_path,
            transcript=transcript,
            score=score,
            feedback=evaluation["feedback"],
            matched_concepts=json.dumps(evaluation["matched_concepts"]),
            missing_concepts=json.dumps(evaluation["missing_concepts"]),
            incorrect_claims=json.dumps(evaluation["incorrect_claims"]),
        )
        db.add(answer)
        db.add(
            Result(
                session_id=session_id,
                student_id=student_id,
                experiment_id=experiment_id,
                question_id=question_id,
                transcript=transcript,
                score=score,
                feedback=evaluation["feedback"],
                matched_concepts=json.dumps(evaluation["matched_concepts"]),
                missing_concepts=json.dumps(evaluation["missing_concepts"]),
            )
        )
        db.flush()

        answered = (
            db.query(VivaAnswer)
            .join(VivaQuestion)
            .filter(VivaQuestion.session_id == session_id)
            .all()
        )
        total = sum(item.score for item in answered)
        session = db.get(VivaSession, session_id)
        session.total_score = min(float(session.max_score), round(total, 1))
        if len(answered) >= len(session.questions):
            session.status = "COMPLETED"
            session.completed_at = datetime.utcnow()

        db.refresh(answer)
        db.expunge(answer)
        return answer


def student_sessions(student_id: int) -> list[VivaSession]:
    with session_scope() as db:
        rows = (
            db.query(VivaSession)
            .options(
                joinedload(VivaSession.experiment).joinedload(Experiment.manual),
                joinedload(VivaSession.questions).joinedload(VivaQuestion.answer),
            )
            .filter(VivaSession.student_id == student_id)
            .order_by(VivaSession.started_at.desc())
            .all()
        )
        for row in rows:
            row.experiment.manual
            for question in row.questions:
                question.answer
            db.expunge(row)
        return rows


def attempt_rows(faculty_id: int | None = None) -> list[dict]:
    with session_scope() as db:
        query = (
            db.query(Result, User, Experiment, Manual, VivaQuestion, VivaSession)
            .join(User, Result.student_id == User.id)
            .join(Experiment, Result.experiment_id == Experiment.id)
            .join(Manual, Experiment.manual_id == Manual.id)
            .join(VivaQuestion, Result.question_id == VivaQuestion.id)
            .join(VivaSession, Result.session_id == VivaSession.id)
        )
        if faculty_id is not None:
            query = query.filter(Manual.faculty_id == faculty_id)
        rows = query.order_by(Result.created_at.desc()).all()
        payload = []
        for result, student, experiment, manual, question, session in rows:
            payload.append(
                {
                    "result_id": result.id,
                    "created_at": result.created_at,
                    "student": student.full_name,
                    "email": student.email,
                    "university_id": student.university_id,
                    "manual": manual.title,
                    "experiment": f"Ex {experiment.number}: {experiment.title}",
                    "question": question.question_text,
                    "transcript": result.transcript,
                    "score": result.score,
                    "feedback": result.feedback,
                    "matched": ", ".join(load_json_list(result.matched_concepts)),
                    "missing": ", ".join(load_json_list(result.missing_concepts)),
                    "session_status": session.status,
                    "session_total": session.total_score,
                    "allowed_exits": session.allowed_exits,
                    "exit_count": session.exit_count,
                    "exit_violations": load_json_list(session.exit_violations),
                    "terminated_reason": session.terminated_reason,
                }
            )
        return payload


def hod_stats() -> dict:
    with session_scope() as db:
        students = db.query(User).filter(User.role == "STUDENT").count()
        faculty = db.query(User).filter(User.role == "FACULTY").count()
        manuals = db.query(Manual).count()
        experiments = db.query(Experiment).count()
        completed = db.query(VivaSession).filter(VivaSession.status == "COMPLETED").count()
        avg_question = db.query(func.avg(Result.score)).scalar()
        avg_session = db.query(func.avg(VivaSession.total_score)).filter(VivaSession.status == "COMPLETED").scalar()
        return {
            "students": students,
            "faculty": faculty,
            "manuals": manuals,
            "experiments": experiments,
            "completed_vivas": completed,
            "average_question_marks": float(avg_question or 0),
            "average_session_marks": float(avg_session or 0),
        }


def list_users() -> list[User]:
    with session_scope() as db:
        users = db.query(User).order_by(User.created_at.desc()).all()
        for user in users:
            db.expunge(user)
        return users


def audio_file_path(student_id: int) -> Path:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    return AUDIO_DIR / f"student_{student_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.wav"
