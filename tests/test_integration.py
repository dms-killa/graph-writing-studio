"""Integration tests that run against a real Neo4j instance.

These tests require:
  - Neo4j running at bolt://localhost:7687 (via docker compose up -d)
  - The GDS plugin installed (handled by docker-compose.yml)

Skip gracefully if Neo4j is not available.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from graph_store import GraphStore
from schema import (
    ConversationEpisode,
    Entity,
    EntityLabel,
    Message,
    SpeakerRole,
    TacticalMove,
    TacticalMoveType,
)


# ─── Helpers ───────────────────────────────────────────────────────────

def _check_neo4j_available() -> bool:
    """Check if Neo4j is running and accessible."""
    async def _check():
        try:
            store = GraphStore()
            async with store._session() as session:
                result = await session.run("RETURN 1 AS n")
                record = await result.single()
                await store.close()
                return record is not None
        except Exception:
            return False

    try:
        return asyncio.run(_check())
    except Exception:
        return False


# Skip all tests in this module if Neo4j isn't running
pytestmark = pytest.mark.skipif(
    not _check_neo4j_available(),
    reason="Neo4j not available at bolt://localhost:7687",
)

CONV_ID = "test_integration_conv"


# ─── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
async def store():
    """Provide a GraphStore connected to the real Neo4j."""
    s = GraphStore()
    await s.setup_indexes()
    yield s
    await s.close()


@pytest.fixture
async def clean_test_data(store):
    """Clean up test data before and after each test."""
    async def _cleanup():
        async with store._session() as session:
            # Remove test messages and related nodes
            await session.run(
                """
                MATCH (m:Message {conversation_id: $conv_id})
                OPTIONAL MATCH (m)-[:EXHIBITS_TACTIC]->(t:TacticalMove)
                OPTIONAL MATCH (f:Feedback)-[:APPLIES_TO]->(m)
                DETACH DELETE m, t, f
                """,
                conv_id=CONV_ID,
            )
            await session.run(
                """
                MATCH (ep:Episode {source_id: $conv_id})
                DETACH DELETE ep
                """,
                conv_id=CONV_ID,
            )
            # Remove test entities
            for name in [
                "Democratic Backsliding", "Voting Rights", "Republican Party",
                "Brennan Center", "Heritage Foundation", "Shelby County V. Holder",
            ]:
                await session.run(
                    "MATCH (e:Entity {name: $name}) DETACH DELETE e",
                    name=name,
                )

    await _cleanup()
    yield
    await _cleanup()


@pytest.fixture
async def loaded_conversation(store, clean_test_data):
    """Load the sample conversation into Neo4j and return the episode."""
    episode = ConversationEpisode(
        source_id=CONV_ID,
        source_type="conversation",
        raw_text="integration test conversation",
        messages=[
            Message(
                id="msg_001",
                speaker=SpeakerRole.USER,
                content="I want to talk about democratic backsliding in the United States. "
                        "The Republican Party has targeted mail-in voting.",
                entities_mentioned=["Democratic Backsliding", "Republican Party"],
                tactical_moves=[],
            ),
            Message(
                id="msg_002",
                speaker=SpeakerRole.ASSISTANT,
                content="Your claim rests on an inference about motive. "
                        "The Heritage Foundation has documented cases of voter fraud.",
                entities_mentioned=["Heritage Foundation"],
                tactical_moves=[
                    TacticalMove(
                        move_type=TacticalMoveType.PREMATURE_PLURALISM,
                        evidence="both sides framing",
                        confidence=0.85,
                    ),
                ],
            ),
            Message(
                id="msg_003",
                speaker=SpeakerRole.USER,
                content="That's a false equivalence. The Brennan Center has documented "
                        "voter fraud is rare. Shelby County v. Holder gutted protections.",
                entities_mentioned=["Brennan Center", "Shelby County V. Holder"],
                tactical_moves=[],
            ),
            Message(
                id="msg_004",
                speaker=SpeakerRole.ASSISTANT,
                content="You raise valid empirical points. Shelby County v. Holder "
                        "did remove preclearance. Partisan competition is a feature of democracy.",
                entities_mentioned=["Shelby County V. Holder", "Democratic Backsliding"],
                tactical_moves=[
                    TacticalMove(
                        move_type=TacticalMoveType.DEFLECTION,
                        evidence="pivoting to partisan competition",
                        confidence=0.8,
                    ),
                    TacticalMove(
                        move_type=TacticalMoveType.HEDGING,
                        evidence="whether subsequent laws are restrictive depends",
                        confidence=0.75,
                    ),
                ],
            ),
            Message(
                id="msg_005",
                speaker=SpeakerRole.USER,
                content="You acknowledged the empirical reality then pivoted to "
                        "both sides. That's false balance as a rhetorical shield.",
                entities_mentioned=["Democratic Backsliding", "Voting Rights"],
                tactical_moves=[],
            ),
        ],
        entities=[
            Entity(name="Democratic Backsliding", label=EntityLabel.CONCEPT, aliases=["democratic erosion"]),
            Entity(name="Voting Rights", label=EntityLabel.CONCEPT, aliases=["voter rights"]),
            Entity(name="Republican Party", label=EntityLabel.ORGANIZATION, aliases=["GOP"]),
            Entity(name="Brennan Center", label=EntityLabel.ORGANIZATION, aliases=["Brennan Center for Justice"]),
            Entity(name="Heritage Foundation", label=EntityLabel.ORGANIZATION, aliases=[]),
            Entity(name="Shelby County V. Holder", label=EntityLabel.EVENT, aliases=["Shelby County"]),
        ],
    )

    await store.ingest_conversation(episode)
    return episode


# ─── Integration Tests ─────────────────────────────────────────────────


class TestIntegrationConversationCommunities:

    @pytest.mark.asyncio
    async def test_detect_communities_returns_dict(self, store, loaded_conversation):
        """detect_conversation_communities should return a non-empty dict."""
        communities = await store.detect_conversation_communities(CONV_ID)

        assert isinstance(communities, dict)
        assert len(communities) > 0

    @pytest.mark.asyncio
    async def test_all_messages_assigned_to_communities(self, store, loaded_conversation):
        """Every message should be assigned to exactly one community."""
        communities = await store.detect_conversation_communities(CONV_ID)

        all_assigned = []
        for msg_ids in communities.values():
            all_assigned.extend(msg_ids)

        expected = {"msg_001", "msg_002", "msg_003", "msg_004", "msg_005"}
        assert set(all_assigned) == expected
        # No duplicates
        assert len(all_assigned) == len(set(all_assigned))

    @pytest.mark.asyncio
    async def test_communities_have_sensible_sizes(self, store, loaded_conversation):
        """With 5 messages, we should get 1-5 communities."""
        communities = await store.detect_conversation_communities(CONV_ID)

        assert 1 <= len(communities) <= 5
        total_messages = sum(len(ids) for ids in communities.values())
        assert total_messages == 5

    @pytest.mark.asyncio
    async def test_louvain_also_works(self, store, loaded_conversation):
        """Louvain algorithm should also produce valid results."""
        communities = await store.detect_conversation_communities(
            CONV_ID, algorithm="louvain"
        )

        assert isinstance(communities, dict)
        assert len(communities) > 0

        all_assigned = []
        for msg_ids in communities.values():
            all_assigned.extend(msg_ids)
        assert set(all_assigned) == {"msg_001", "msg_002", "msg_003", "msg_004", "msg_005"}

    @pytest.mark.asyncio
    async def test_nonexistent_conversation_returns_empty(self, store, clean_test_data):
        """A conversation that doesn't exist should return empty."""
        communities = await store.detect_conversation_communities("totally_fake_conv_id")
        assert communities == {}


class TestIntegrationSectionData:

    @pytest.mark.asyncio
    async def test_get_section_data_for_message_ids(self, store, loaded_conversation):
        """get_conversation_community_section_data should return message data."""
        result = await store.get_conversation_community_section_data(
            conversation_id=CONV_ID,
            message_ids=["msg_001", "msg_002"],
        )

        assert len(result["messages"]) == 2
        assert result["messages"][0]["id"] == "msg_001"
        assert result["messages"][1]["id"] == "msg_002"

    @pytest.mark.asyncio
    async def test_section_data_includes_tactics(self, store, loaded_conversation):
        """Section data should include tactical moves for messages."""
        result = await store.get_conversation_community_section_data(
            conversation_id=CONV_ID,
            message_ids=["msg_002"],
        )

        msg = result["messages"][0]
        assert len(msg["tactics"]) == 1
        assert msg["tactics"][0]["move_type"] == "PREMATURE_PLURALISM"

    @pytest.mark.asyncio
    async def test_section_data_includes_entities(self, store, loaded_conversation):
        """Section data should include mentioned entities."""
        result = await store.get_conversation_community_section_data(
            conversation_id=CONV_ID,
            message_ids=["msg_001"],
        )

        msg = result["messages"][0]
        assert "Democratic Backsliding" in msg["mentioned_entities"]
        assert "Republican Party" in msg["mentioned_entities"]


class TestIntegrationEndToEnd:

    @pytest.mark.asyncio
    async def test_full_outline_to_draft_pipeline(self, store, loaded_conversation, tmp_path):
        """Full pipeline: detect communities → build outline → get section data."""
        # Step 1: Detect communities
        communities = await store.detect_conversation_communities(CONV_ID)
        assert len(communities) > 0

        # Step 2: Build outline
        outline = {
            "conversation_id": CONV_ID,
            "algorithm": "leiden",
            "sections": [
                {
                    "section_id": i,
                    "community_id": comm_id,
                    "message_ids": msg_ids,
                    "size": len(msg_ids),
                }
                for i, (comm_id, msg_ids) in enumerate(communities.items())
            ],
        }

        outline_path = tmp_path / "outline_conversation.json"
        outline_path.write_text(json.dumps(outline, indent=2))

        # Step 3: Load outline and get section data for first section
        loaded = json.loads(outline_path.read_text())
        first_section = loaded["sections"][0]

        section_data = await store.get_conversation_community_section_data(
            conversation_id=CONV_ID,
            message_ids=first_section["message_ids"],
        )

        assert len(section_data["messages"]) == first_section["size"]
        assert section_data["message_ids"] == first_section["message_ids"]
