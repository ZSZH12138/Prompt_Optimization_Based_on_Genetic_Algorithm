import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)

LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))

WEIGHTS = {"hard": 0.25, "fact": 0.35, "qual": 0.40}
STABILITY_LAMBDA = 0.15

USE_PAIRWISE_QUALITY = True

BASELINE_PROMPT = (
    "You are a careful assistant. Summarize the given text for a busy reader.\n"
    "Requirements:\n"
    "- Keep it faithful to the source (do not add facts not in the text).\n"
    "- Provide: (1) 1-sentence thesis; (2) 5-8 bullet key points; (3) 3 actionable takeaways if applicable.\n"
    "- Use Markdown.\n\n"
    "Title: #title\nAuthor: #author\n<text>\n#text\n</text>"
)

MAX_OUTPUT_TOKENS = 1200
JUDGE_MAX_TOKENS = 1200

MAX_CLAIMS = 12

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(CURRENT_DIR, "..", "data", "outputs")
