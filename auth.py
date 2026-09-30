from werkzeug.security import check_password_hash, generate_password_hash

from config import EMAIL_PATTERN, ROLES
from database import session_scope
from models import User


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def is_valid_university_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.fullmatch(normalize_email(email)))


def hash_password(password: str) -> str:
    return generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)


def verify_password(password_hash: str, password: str) -> bool:
    if not password_hash or password is None:
        return False
    return check_password_hash(password_hash, password)


def user_count() -> int:
    with session_scope() as db:
        return db.query(User).count()


def get_user_by_email(email: str) -> User | None:
    email = normalize_email(email)
    with session_scope() as db:
        user = db.query(User).filter(User.email == email).one_or_none()
        if user:
            db.expunge(user)
        return user


def get_user_by_id(user_id: int) -> User | None:
    with session_scope() as db:
        user = db.get(User, user_id)
        if user:
            db.expunge(user)
        return user


def authenticate(email: str, password: str) -> tuple[User | None, str]:
    if not is_valid_university_email(email):
        return None, "Only KL University emails of the form 99240040986@klu.ac.in are accepted."
    user = get_user_by_email(email)
    if user is None or not verify_password(user.password_hash, password):
        return None, "Incorrect email or password."
    if not user.is_active:
        return None, "This account is deactivated. Contact the administrator."
    return user, ""


def create_user(
    email: str,
    password: str,
    full_name: str,
    role: str,
    is_active: bool = True,
    created_by_id: int | None = None,
) -> tuple[User | None, str]:
    email = normalize_email(email)
    full_name = (full_name or "").strip()
    role = (role or "").strip().upper()
    if not is_valid_university_email(email):
        return None, "Email must match digits only, followed by @klu.ac.in."
    if len(password or "") < 8:
        return None, "Password must be at least 8 characters."
    if not full_name:
        return None, "Full name is required."
    if role not in ROLES:
        return None, "Invalid role."
    with session_scope() as db:
        if db.query(User).filter(User.email == email).one_or_none():
            return None, "An account with this email already exists."
        user = User(
            email=email,
            password_hash=hash_password(password),
            full_name=full_name,
            role=role,
            is_active=is_active,
            created_by_id=created_by_id,
        )
        db.add(user)
        db.flush()
        db.refresh(user)
        db.expunge(user)
        return user, ""


def register_student(email: str, password: str, full_name: str) -> tuple[User | None, str]:
    return create_user(email, password, full_name, "STUDENT", is_active=False)


def bootstrap_admin(email: str, password: str, full_name: str) -> tuple[User | None, str]:
    if user_count() > 0:
        return None, "An administrator already exists. Sign in instead."
    return create_user(email, password, full_name, "ADMIN", is_active=True)


def set_user_active(user_id: int, is_active: bool) -> None:
    with session_scope() as db:
        user = db.get(User, user_id)
        if user:
            user.is_active = is_active


def set_user_role(user_id: int, role: str) -> str:
    role = (role or "").strip().upper()
    if role not in ROLES:
        return "Invalid role."
    with session_scope() as db:
        user = db.get(User, user_id)
        if not user:
            return "User not found."
        user.role = role
    return ""


def reset_user_password(user_id: int, password: str) -> str:
    if len(password or "") < 8:
        return "Password must be at least 8 characters."
    with session_scope() as db:
        user = db.get(User, user_id)
        if not user:
            return "User not found."
        user.password_hash = hash_password(password)
    return ""
