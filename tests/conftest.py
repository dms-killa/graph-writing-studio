"""Shared fixtures for Graph Writing Studio tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─── Sample Data ───────────────────────────────────────────────────────

SAMPLE_CONVERSATION_PATH = Path(__file__).parent.parent / "samples" / "democratic_backsliding_chat.md"


@pytest.fixture
def sample_conversation_text():
    """Load the sample conversation transcript."""
    return SAMPLE_CONVERSATION_PATH.read_text(encoding="utf-8")


@pytest.fixture
def sample_conversation_id():
    return "democratic_backsliding_chat"


@pytest.fixture
def sample_message_ids():
    """Message IDs that would be generated for the sample conversation."""
    return ["msg_001", "msg_002", "msg_003", "msg_004", "msg_005"]


@pytest.fixture
def sample_communities():
    """Simulated community detection output: community_id → message_ids."""
    return {
        0: ["msg_001", "msg_002", "msg_003"],
        1: ["msg_004", "msg_005"],
    }


@pytest.fixture
def sample_outline_conversation(sample_communities, sample_conversation_id):
    """Outline file content matching what outline-conversation produces."""
    return {
        "conversation_id": sample_conversation_id,
        "algorithm": "leiden",
        "sections": [
            {
                "section_id": i,
                "community_id": comm_id,
                "message_ids": msg_ids,
                "size": len(msg_ids),
            }
            for i, (comm_id, msg_ids) in enumerate(sample_communities.items())
        ],
    }


@pytest.fixture
def sample_entity_outline():
    """Outline file content matching what the entity 'outline' command produces."""
    return [
        {
            "community_id": 0,
            "members": ["Democratic Backsliding", "Voting Rights", "Republican Party"],
            "size": 3,
        },
        {
            "community_id": 1,
            "members": ["Brennan Center", "Heritage Foundation"],
            "size": 2,
        },
    ]


@pytest.fixture
def sample_section_data():
    """Sample return value from get_conversation_community_section_data."""
    return {
        "messages": [
            {
                "id": "msg_001",
                "speaker": "USER",
                "content": "I want to talk about democratic backsliding...",
                "turn": 1,
                "mentioned_entities": ["Democratic Backsliding", "Republican Party"],
                "tactics": [],
                "feedback": [],
            },
            {
                "id": "msg_002",
                "speaker": "ASSISTANT",
                "content": "Your claim rests on an inference about motive...",
                "turn": 2,
                "mentioned_entities": ["Heritage Foundation"],
                "tactics": [
                    {"move_type": "PREMATURE_PLURALISM", "evidence": "both sides", "confidence": 0.85}
                ],
                "feedback": [],
            },
        ],
        "entity_relations": [
            {
                "source": "Republican Party",
                "target": "Voting Rights",
                "type": "RELATED_TO",
                "context": "attacks on voting rights",
                "confidence": 0.9,
            }
        ],
        "message_ids": ["msg_001", "msg_002"],
    }


# ─── Mock helpers ──────────────────────────────────────────────────────

def make_mock_neo4j_record(data: dict):
    """Create a mock Neo4j record from a dict."""
    record = MagicMock()
    record.__getitem__ = lambda self, key: data[key]
    record.get = lambda key, default=None: data.get(key, default)
    return record


class MockAsyncResult:
    """Mock for async Neo4j result that supports async iteration."""

    def __init__(self, records: list[dict]):
        self._records = [make_mock_neo4j_record(r) for r in records]
        self._index = 0

    async def single(self):
        if self._records:
            return self._records[0]
        return None

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._records):
            raise StopAsyncIteration
        record = self._records[self._index]
        self._index += 1
        return record


@pytest.fixture
def mock_session():
    """Create a mock Neo4j async session."""
    session = AsyncMock()
    return session


@pytest.fixture
def mock_graph_store(mock_session):
    """Create a GraphStore with mocked Neo4j driver."""
    with patch("graph_store.AsyncGraphDatabase") as mock_db:
        mock_driver = MagicMock()
        mock_db.driver.return_value = mock_driver

        # The driver.session() returns an async context manager
        # that yields our mock_session
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        session_cm.__aexit__ = AsyncMock(return_value=False)
        mock_driver.session.return_value = session_cm
        mock_driver.close = AsyncMock()

        from graph_store import GraphStore
        store = GraphStore.__new__(GraphStore)
        store._driver = mock_driver
        yield store, mock_session
