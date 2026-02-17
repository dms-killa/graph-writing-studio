"""
Centralized configuration for Graph Writing Studio.

Reads settings from a .env file (if present) and falls back to defaults.
All modules import from here instead of defining their own constants.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─── .env loader (no external dependency) ─────────────────────────────

def _load_dotenv(path: Path | str = ".env") -> None:
    """Load key=value pairs from a .env file into os.environ."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Don't override existing environment variables
        if key not in os.environ:
            os.environ[key] = value


# Load .env from the project root (same directory as this file)
_load_dotenv(Path(__file__).parent / ".env")

# ─── Ollama ────────────────────────────────────────────────────────────

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.1:70b")
OLLAMA_TEMPERATURE: float = float(os.environ.get("OLLAMA_TEMPERATURE", "0.1"))
OLLAMA_TIMEOUT: float = float(os.environ.get("OLLAMA_TIMEOUT", "300.0"))

# ─── Neo4j ─────────────────────────────────────────────────────────────

NEO4J_URI: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.environ.get("NEO4J_PASSWORD", "graphstudio")
