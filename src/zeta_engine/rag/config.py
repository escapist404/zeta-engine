"""Configuration constants for the RAG pipeline."""

LLM_API_KEY_ENV = "ZETA_LLM_API_KEY"
LLM_BASE_URL_ENV = "ZETA_LLM_BASE_URL"
LLM_MODEL_ENV = "ZETA_LLM_MODEL"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-v4-flash"
LLM_TIMEOUT_SECONDS = 55.0
LLM_MAX_RETRIES = 0
LLM_MAX_OUTPUT_TOKENS = 8192
RAG_MAX_CONTEXT_CHARS = 48_000
RAG_MAX_CONTEXT_TOKENS = 32_000
RAG_NO_RESULTS_ANSWER = "未检索到相关信息"
# Leave room for newly retrieved evidence to be inspected, calculated, and
# repaired before the final forced-answer cycle.
AGENT_MAX_CYCLES = 5
AGENT_MAX_QUERIES = 3
AGENT_MAX_RESULTS = 20
AGENT_MAX_CHUNKS_PER_URL = 3
AGENT_MAX_LLM_CALLS = 12
