from pathlib import Path
import re

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MANUAL_DIR = DATA_DIR / "manuals"
AUDIO_DIR = DATA_DIR / "audio"
RESULT_DIR = DATA_DIR / "results"
DB_PATH = DATA_DIR / "virtual_viva.db"
LOGO_PATH = DATA_DIR / "kare_logo.png"

for _path in (DATA_DIR, MANUAL_DIR, AUDIO_DIR, RESULT_DIR):
    _path.mkdir(parents=True, exist_ok=True)

APP_NAME = "KARE Virtual Laboratory Viva Assessment"
APP_SHORT = "Virtual Laboratory Assessment"
UNIVERSITY = "KALASALINGAM ACADEMY OF RESEARCH AND EDUCATION"
DEPARTMENT = "Department of Computer Science and Engineering"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
OLLAMA_MODEL = "llama3.2:3b"
NEMO_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# Accept student email pattern for university IDs ending with @klu.ac.in or @kalasalingam.ac.in
EMAIL_PATTERN = re.compile(r"^\d+@(klu\.ac\.in|kalasalingam\.ac\.in)$")
QUESTIONS_PER_VIVA = 3
MAX_MARKS_PER_QUESTION = 10
MAX_MARKS_TOTAL = QUESTIONS_PER_VIVA * MAX_MARKS_PER_QUESTION
QUESTION_TIMER_SECONDS = 120
DEFAULT_ALLOWED_FULLSCREEN_EXITS = 3

ROLES = ("ADMIN", "FACULTY", "HOD", "STUDENT")
