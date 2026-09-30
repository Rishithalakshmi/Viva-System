from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    manuals: Mapped[list[Manual]] = relationship(back_populates="faculty")
    sessions: Mapped[list[VivaSession]] = relationship(
        back_populates="student",
        foreign_keys="VivaSession.student_id",
    )

    @property
    def university_id(self) -> str:
        return self.email.split("@", 1)[0]


class Manual(Base):
    __tablename__ = "manuals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(500), nullable=False)
    extracted_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    faculty_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    is_published: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    allowed_exits: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    question_count: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    faculty: Mapped[User] = relationship(back_populates="manuals")
    experiments: Mapped[list[Experiment]] = relationship(
        back_populates="manual",
        cascade="all, delete-orphan",
        order_by="Experiment.number",
    )


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    manual_id: Mapped[int] = mapped_column(ForeignKey("manuals.id"), nullable=False)
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)

    manual: Mapped[Manual] = relationship(back_populates="experiments")
    sessions: Mapped[list[VivaSession]] = relationship(back_populates="experiment")

    __table_args__ = (UniqueConstraint("manual_id", "number", name="uq_manual_experiment_number"),)


class VivaSession(Base):
    __tablename__ = "viva_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="IN_PROGRESS", nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    total_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    max_score: Mapped[float] = mapped_column(Float, default=50.0, nullable=False)
    allowed_exits: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    exit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    exit_violations: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    terminated_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)

    student: Mapped[User] = relationship(back_populates="sessions", foreign_keys=[student_id])
    experiment: Mapped[Experiment] = relationship(back_populates="sessions")
    questions: Mapped[list[VivaQuestion]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="VivaQuestion.order_index",
    )


class VivaQuestion(Base):
    __tablename__ = "viva_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("viva_sessions.id"), nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    expected_concepts: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    difficulty: Mapped[str] = mapped_column(String(20), default="medium", nullable=False)

    session: Mapped[VivaSession] = relationship(back_populates="questions")
    answer: Mapped[VivaAnswer | None] = relationship(
        back_populates="question",
        cascade="all, delete-orphan",
        uselist=False,
    )

    __table_args__ = (UniqueConstraint("session_id", "order_index", name="uq_session_question_order"),)


class VivaAnswer(Base):
    __tablename__ = "viva_answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("viva_questions.id"), unique=True, nullable=False)
    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    transcript: Mapped[str] = mapped_column(Text, default="", nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    feedback: Mapped[str] = mapped_column(Text, default="", nullable=False)
    matched_concepts: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    missing_concepts: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    incorrect_claims: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    question: Mapped[VivaQuestion] = relationship(back_populates="answer")


class Result(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("viva_sessions.id"), nullable=False)
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    question_id: Mapped[int] = mapped_column(ForeignKey("viva_questions.id"), nullable=False)
    transcript: Mapped[str] = mapped_column(Text, default="", nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    feedback: Mapped[str] = mapped_column(Text, default="", nullable=False)
    matched_concepts: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    missing_concepts: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
