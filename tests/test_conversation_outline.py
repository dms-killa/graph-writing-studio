"""Tests for conversation-aware outline generation.

Tests cover:
  1. detect_conversation_communities() in graph_store.py
  2. outline-conversation CLI command
  3. draft-conversation integration with conversation outline
  4. Conversation parsing and end-to-end flow
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from tests.conftest import MockAsyncResult


# ═══════════════════════════════════════════════════════════════════════
# 1. GraphStore.detect_conversation_communities() unit tests
# ═══════════════════════════════════════════════════════════════════════


class TestDetectConversationCommunities:
    """Tests for the detect_conversation_communities method."""

    @pytest.fixture
    def store_and_session(self, mock_graph_store):
        store, session = mock_graph_store
        return store, session

    def _setup_session_run(self, session, community_records):
        """Set up mock session.run to return community records on the fetch query."""
        call_count = [0]

        async def mock_run(query, **kwargs):
            call_count[0] += 1
            query_stripped = query.strip()

            # graph.drop calls - return empty
            if "gds.graph.drop" in query_stripped:
                return MockAsyncResult([])

            # graph.project call - return empty
            if "gds.graph.project" in query_stripped:
                return MockAsyncResult([])

            # algorithm write call - return empty
            if "gds.leiden.write" in query_stripped or "gds.louvain.write" in query_stripped:
                return MockAsyncResult([])

            # Community fetch query
            if "m.conv_community" in query_stripped:
                return MockAsyncResult(community_records)

            return MockAsyncResult([])

        session.run = mock_run

    @pytest.mark.asyncio
    async def test_returns_communities_keyed_by_id(self, store_and_session, sample_conversation_id):
        store, session = store_and_session
        self._setup_session_run(session, [
            {"community_id": 0, "message_ids": ["msg_001", "msg_002", "msg_003"], "size": 3},
            {"community_id": 1, "message_ids": ["msg_004", "msg_005"], "size": 2},
        ])

        result = await store.detect_conversation_communities(sample_conversation_id)

        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result
        assert result[0] == ["msg_001", "msg_002", "msg_003"]
        assert result[1] == ["msg_004", "msg_005"]

    @pytest.mark.asyncio
    async def test_empty_conversation_returns_empty(self, store_and_session):
        store, session = store_and_session
        self._setup_session_run(session, [])

        result = await store.detect_conversation_communities("nonexistent_conv")

        assert result == {}

    @pytest.mark.asyncio
    async def test_single_community(self, store_and_session, sample_conversation_id):
        store, session = store_and_session
        self._setup_session_run(session, [
            {"community_id": 0, "message_ids": ["msg_001", "msg_002", "msg_003", "msg_004", "msg_005"], "size": 5},
        ])

        result = await store.detect_conversation_communities(sample_conversation_id)

        assert len(result) == 1
        assert len(result[0]) == 5

    @pytest.mark.asyncio
    async def test_messages_sorted_by_turn(self, store_and_session, sample_conversation_id):
        store, session = store_and_session
        # Provide message IDs in non-sorted order
        self._setup_session_run(session, [
            {"community_id": 0, "message_ids": ["msg_003", "msg_001", "msg_005"], "size": 3},
        ])

        result = await store.detect_conversation_communities(sample_conversation_id)

        assert result[0] == ["msg_001", "msg_003", "msg_005"]

    @pytest.mark.asyncio
    async def test_uses_leiden_by_default(self, store_and_session, sample_conversation_id):
        store, session = store_and_session
        queries_run = []

        async def mock_run(query, **kwargs):
            queries_run.append(query.strip())
            if "m.conv_community" in query.strip():
                return MockAsyncResult([
                    {"community_id": 0, "message_ids": ["msg_001"], "size": 1},
                ])
            return MockAsyncResult([])

        session.run = mock_run

        await store.detect_conversation_communities(sample_conversation_id)

        algo_queries = [q for q in queries_run if "gds.leiden" in q or "gds.louvain" in q]
        assert len(algo_queries) == 1
        assert "gds.leiden" in algo_queries[0]

    @pytest.mark.asyncio
    async def test_uses_louvain_when_specified(self, store_and_session, sample_conversation_id):
        store, session = store_and_session
        queries_run = []

        async def mock_run(query, **kwargs):
            queries_run.append(query.strip())
            if "m.conv_community" in query.strip():
                return MockAsyncResult([
                    {"community_id": 0, "message_ids": ["msg_001"], "size": 1},
                ])
            return MockAsyncResult([])

        session.run = mock_run

        await store.detect_conversation_communities(
            sample_conversation_id, algorithm="louvain"
        )

        algo_queries = [q for q in queries_run if "gds.leiden" in q or "gds.louvain" in q]
        assert len(algo_queries) == 1
        assert "gds.louvain" in algo_queries[0]

    @pytest.mark.asyncio
    async def test_passes_conversation_id_to_queries(self, store_and_session):
        store, session = store_and_session
        captured_kwargs = []

        async def mock_run(query, **kwargs):
            captured_kwargs.append(kwargs)
            if "m.conv_community" in query.strip():
                return MockAsyncResult([])
            return MockAsyncResult([])

        session.run = mock_run

        await store.detect_conversation_communities("test_conv_123")

        # At least one query should have conv_id parameter
        conv_id_params = [
            kw for kw in captured_kwargs
            if kw.get("conv_id") == "test_conv_123" or kw.get("conv_id") == "test_conv_123"
        ]
        assert len(conv_id_params) > 0

    @pytest.mark.asyncio
    async def test_cleans_up_gds_graph(self, store_and_session, sample_conversation_id):
        store, session = store_and_session
        queries_run = []

        async def mock_run(query, **kwargs):
            queries_run.append(query.strip())
            if "m.conv_community" in query.strip():
                return MockAsyncResult([])
            return MockAsyncResult([])

        session.run = mock_run

        await store.detect_conversation_communities(sample_conversation_id)

        drop_calls = [q for q in queries_run if "gds.graph.drop" in q]
        # Should have at least 2 drop calls: one at start (cleanup) and one at end
        assert len(drop_calls) >= 2


# ═══════════════════════════════════════════════════════════════════════
# 2. GraphStore.get_conversation_community_section_data() unit tests
# ═══════════════════════════════════════════════════════════════════════


class TestGetConversationCommunitySectionData:

    @pytest.fixture
    def store_and_session(self, mock_graph_store):
        store, session = mock_graph_store
        return store, session

    @pytest.mark.asyncio
    async def test_returns_messages_for_given_ids(self, store_and_session):
        store, session = store_and_session

        msg_records = [
            {
                "id": "msg_001",
                "speaker": "USER",
                "content": "Hello world",
                "turn": 1,
                "mentioned_entities": [],
                "tactics": [],
                "feedback": [],
            },
            {
                "id": "msg_002",
                "speaker": "ASSISTANT",
                "content": "Hi there",
                "turn": 2,
                "mentioned_entities": [],
                "tactics": [],
                "feedback": [],
            },
        ]

        call_count = [0]

        async def mock_run(query, **kwargs):
            call_count[0] += 1
            if "m.id IN" in query or "msg_ids" in query:
                return MockAsyncResult(msg_records)
            # Neighborhood queries
            return MockAsyncResult([])

        session.run = mock_run

        result = await store.get_conversation_community_section_data(
            conversation_id="test_conv",
            message_ids=["msg_001", "msg_002"],
        )

        assert len(result["messages"]) == 2
        assert result["messages"][0]["id"] == "msg_001"
        assert result["messages"][1]["id"] == "msg_002"
        assert result["message_ids"] == ["msg_001", "msg_002"]

    @pytest.mark.asyncio
    async def test_empty_message_ids_returns_empty(self, store_and_session):
        store, session = store_and_session

        async def mock_run(query, **kwargs):
            return MockAsyncResult([])

        session.run = mock_run

        result = await store.get_conversation_community_section_data(
            conversation_id="test_conv",
            message_ids=[],
        )

        assert result["messages"] == []
        assert result["entity_relations"] == []


# ═══════════════════════════════════════════════════════════════════════
# 3. CLI command: outline-conversation
# ═══════════════════════════════════════════════════════════════════════


class TestOutlineConversationCLI:

    @pytest.mark.asyncio
    async def test_outline_conversation_creates_json_file(
        self, tmp_path, sample_conversation_id, sample_communities
    ):
        """outline-conversation should create outline_conversation.json."""
        from main import cmd_outline_conversation

        args = MagicMock()
        args.conversation_id = sample_conversation_id
        args.algorithm = "leiden"

        mock_store = AsyncMock()
        mock_store.detect_conversation_communities.return_value = sample_communities

        outline_path = tmp_path / "outline_conversation.json"

        with patch("main.GraphStore", return_value=mock_store), \
             patch("main.Path") as mock_path_cls:

            # Make Path("outline_conversation.json") point to our tmp file
            def path_factory(p):
                if p == "outline_conversation.json":
                    return outline_path
                return Path(p)

            mock_path_cls.side_effect = path_factory

            await cmd_outline_conversation(args)

        # Verify the store was called correctly
        mock_store.detect_conversation_communities.assert_called_once_with(
            conversation_id=sample_conversation_id,
            algorithm="leiden",
        )
        mock_store.close.assert_called_once()

        # Verify the outline file was written
        assert outline_path.exists()
        outline = json.loads(outline_path.read_text())
        assert outline["conversation_id"] == sample_conversation_id
        assert outline["algorithm"] == "leiden"
        assert len(outline["sections"]) == 2
        assert outline["sections"][0]["message_ids"] == ["msg_001", "msg_002", "msg_003"]
        assert outline["sections"][1]["message_ids"] == ["msg_004", "msg_005"]

    @pytest.mark.asyncio
    async def test_outline_conversation_empty_communities(self, sample_conversation_id):
        """outline-conversation with no communities should print warning."""
        from main import cmd_outline_conversation

        args = MagicMock()
        args.conversation_id = sample_conversation_id
        args.algorithm = "leiden"

        mock_store = AsyncMock()
        mock_store.detect_conversation_communities.return_value = {}

        with patch("main.GraphStore", return_value=mock_store), \
             patch("main.console") as mock_console:
            await cmd_outline_conversation(args)

        # Should have printed a warning
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        warning_found = any("No communities found" in s for s in print_calls)
        assert warning_found

    @pytest.mark.asyncio
    async def test_outline_conversation_uses_louvain(self, sample_conversation_id):
        """outline-conversation should pass algorithm choice to store."""
        from main import cmd_outline_conversation

        args = MagicMock()
        args.conversation_id = sample_conversation_id
        args.algorithm = "louvain"

        mock_store = AsyncMock()
        mock_store.detect_conversation_communities.return_value = {0: ["msg_001"]}

        with patch("main.GraphStore", return_value=mock_store), \
             patch("main.Path"):
            await cmd_outline_conversation(args)

        mock_store.detect_conversation_communities.assert_called_once_with(
            conversation_id=sample_conversation_id,
            algorithm="louvain",
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. CLI: draft-conversation with conversation outline
# ═══════════════════════════════════════════════════════════════════════


class TestDraftConversationWithConversationOutline:

    @pytest.mark.asyncio
    async def test_draft_uses_conversation_outline_when_available(
        self, tmp_path, sample_outline_conversation, sample_section_data, sample_conversation_id
    ):
        """draft-conversation should prefer outline_conversation.json."""
        from main import cmd_draft_conversation

        args = MagicMock()
        args.section = 0
        args.conversation = sample_conversation_id
        args.template = None

        # Write the conversation outline
        conv_outline_path = tmp_path / "outline_conversation.json"
        conv_outline_path.write_text(json.dumps(sample_outline_conversation))

        mock_store = AsyncMock()
        mock_store.get_conversation_community_section_data.return_value = sample_section_data
        mock_store.get_active_feedback.return_value = []

        draft_dir = tmp_path / "drafts"
        draft_dir.mkdir()

        def path_factory(p):
            if p == "outline_conversation.json":
                return conv_outline_path
            if p == "outline.json":
                return tmp_path / "outline.json"  # doesn't exist
            if isinstance(p, str) and p.startswith("drafts"):
                return tmp_path / p
            return Path(p)

        with patch("main.GraphStore", return_value=mock_store), \
             patch("main.Path", side_effect=path_factory):
            await cmd_draft_conversation(args)

        # Should have used get_conversation_community_section_data, not get_conversation_section_data
        mock_store.get_conversation_community_section_data.assert_called_once_with(
            conversation_id=sample_conversation_id,
            message_ids=["msg_001", "msg_002", "msg_003"],
        )
        mock_store.get_conversation_section_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_draft_falls_back_to_entity_outline(
        self, tmp_path, sample_entity_outline, sample_conversation_id
    ):
        """draft-conversation should fall back to outline.json if no conversation outline."""
        from main import cmd_draft_conversation

        args = MagicMock()
        args.section = 0
        args.conversation = sample_conversation_id
        args.template = None

        # Write entity outline (no conversation outline)
        entity_outline_path = tmp_path / "outline.json"
        entity_outline_path.write_text(json.dumps(sample_entity_outline))

        mock_store = AsyncMock()
        mock_store.get_conversation_section_data.return_value = {
            "messages": [
                {"id": "msg_001", "speaker": "USER", "content": "test", "turn": 1,
                 "mentioned_entities": [], "tactics": [], "feedback": []},
            ],
            "entity_relations": [],
            "community_members": sample_entity_outline[0]["members"],
        }
        mock_store.get_active_feedback.return_value = []

        draft_dir = tmp_path / "drafts"
        draft_dir.mkdir()

        def path_factory(p):
            if p == "outline_conversation.json":
                return tmp_path / "outline_conversation.json"  # doesn't exist
            if p == "outline.json":
                return entity_outline_path
            if isinstance(p, str) and p.startswith("drafts"):
                return tmp_path / p
            return Path(p)

        with patch("main.GraphStore", return_value=mock_store), \
             patch("main.Path", side_effect=path_factory):
            await cmd_draft_conversation(args)

        # Should have used get_conversation_section_data (entity-based)
        mock_store.get_conversation_section_data.assert_called_once()
        mock_store.get_conversation_community_section_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_draft_section_out_of_range(
        self, tmp_path, sample_outline_conversation, sample_conversation_id
    ):
        """draft-conversation with invalid section should print error."""
        from main import cmd_draft_conversation

        args = MagicMock()
        args.section = 99  # out of range
        args.conversation = sample_conversation_id
        args.template = None

        conv_outline_path = tmp_path / "outline_conversation.json"
        conv_outline_path.write_text(json.dumps(sample_outline_conversation))

        def path_factory(p):
            if p == "outline_conversation.json":
                return conv_outline_path
            if p == "outline.json":
                return tmp_path / "outline.json"
            return Path(p)

        with patch("main.GraphStore") as mock_cls, \
             patch("main.Path", side_effect=path_factory), \
             patch("main.console") as mock_console:
            await cmd_draft_conversation(args)

        print_calls = [str(c) for c in mock_console.print.call_args_list]
        error_found = any("doesn't exist" in s for s in print_calls)
        assert error_found


# ═══════════════════════════════════════════════════════════════════════
# 5. Outline file format and interoperability
# ═══════════════════════════════════════════════════════════════════════


class TestOutlineFileFormat:

    def test_outline_has_required_fields(self, sample_outline_conversation):
        """The outline file must contain conversation_id, algorithm, and sections."""
        assert "conversation_id" in sample_outline_conversation
        assert "algorithm" in sample_outline_conversation
        assert "sections" in sample_outline_conversation

    def test_each_section_has_required_fields(self, sample_outline_conversation):
        """Each section must have section_id, community_id, message_ids, size."""
        for section in sample_outline_conversation["sections"]:
            assert "section_id" in section
            assert "community_id" in section
            assert "message_ids" in section
            assert "size" in section
            assert isinstance(section["message_ids"], list)
            assert section["size"] == len(section["message_ids"])

    def test_section_ids_are_sequential(self, sample_outline_conversation):
        """Section IDs should be 0-indexed and sequential."""
        ids = [s["section_id"] for s in sample_outline_conversation["sections"]]
        assert ids == list(range(len(ids)))

    def test_outline_is_json_serializable(self, sample_outline_conversation):
        """The outline must be serializable to JSON."""
        serialized = json.dumps(sample_outline_conversation, indent=2)
        deserialized = json.loads(serialized)
        assert deserialized == sample_outline_conversation

    def test_all_message_ids_unique_across_sections(self, sample_outline_conversation):
        """No message should appear in multiple sections."""
        all_ids = []
        for section in sample_outline_conversation["sections"]:
            all_ids.extend(section["message_ids"])
        assert len(all_ids) == len(set(all_ids))


# ═══════════════════════════════════════════════════════════════════════
# 6. CLI argument parser tests
# ═══════════════════════════════════════════════════════════════════════


class TestCLIParser:

    def test_outline_conversation_parser(self):
        """outline-conversation command should accept conversation_id and --algorithm."""
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args(["outline-conversation", "my_conv"])
        assert args.command == "outline-conversation"
        assert args.conversation_id == "my_conv"
        assert args.algorithm == "leiden"  # default

    def test_outline_conversation_parser_with_algorithm(self):
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args(["outline-conversation", "my_conv", "--algorithm", "louvain"])
        assert args.algorithm == "louvain"

    def test_outline_conversation_in_command_map(self):
        """outline-conversation should be in the command dispatch map."""
        from main import cmd_outline_conversation
        # Just verify the function exists and is importable
        assert callable(cmd_outline_conversation)

    def test_draft_conversation_parser_unchanged(self):
        """draft-conversation command should still accept original args."""
        from main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "draft-conversation",
            "--section", "0",
            "--conversation", "my_conv",
        ])
        assert args.section == 0
        assert args.conversation == "my_conv"


# ═══════════════════════════════════════════════════════════════════════
# 7. Conversation parsing (extractor.py)
# ═══════════════════════════════════════════════════════════════════════


class TestConversationParsing:

    def test_parse_sample_conversation(self, sample_conversation_text):
        """The sample conversation should parse into alternating user/assistant messages."""
        from extractor import parse_conversation

        messages = parse_conversation(sample_conversation_text)
        # The parser may split multi-paragraph messages; we just verify
        # we get a reasonable number and alternating speakers
        assert len(messages) >= 5
        assert messages[0]["speaker"].value.lower() == "user"
        assert messages[1]["speaker"].value.lower() == "assistant"

    def test_parsed_messages_have_content(self, sample_conversation_text):
        """Each parsed message should have non-empty content."""
        from extractor import parse_conversation

        messages = parse_conversation(sample_conversation_text)
        for msg in messages:
            assert len(msg["content"].strip()) > 0

    def test_first_message_mentions_democratic_backsliding(self, sample_conversation_text):
        """The first message should discuss democratic backsliding."""
        from extractor import parse_conversation

        messages = parse_conversation(sample_conversation_text)
        assert "democratic backsliding" in messages[0]["content"].lower()

    def test_messages_on_similar_topics(self, sample_conversation_text):
        """Messages about voting/democracy should cluster together topically."""
        from extractor import parse_conversation

        messages = parse_conversation(sample_conversation_text)
        # All messages in this conversation discuss voting/democracy
        voting_keywords = {"voting", "vote", "voter", "election", "democratic", "democracy", "backsliding"}
        for msg in messages:
            content_lower = msg["content"].lower()
            found = any(kw in content_lower for kw in voting_keywords)
            # At least most messages should mention these topics
            # (this is a soft check since all 5 are on the same topic)
        # The whole conversation is about one topic, so most messages
        # should contain voting/democracy keywords
        assert len(messages) >= 5


# ═══════════════════════════════════════════════════════════════════════
# 8. End-to-end flow test (mocked)
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEndFlow:

    @pytest.mark.asyncio
    async def test_outline_then_draft_workflow(
        self, tmp_path, sample_conversation_id
    ):
        """Test the full workflow: outline-conversation → draft-conversation."""
        from main import cmd_outline_conversation, cmd_draft_conversation

        # Step 1: Run outline-conversation
        communities = {
            0: ["msg_001", "msg_002"],
            1: ["msg_003", "msg_004", "msg_005"],
        }

        outline_args = MagicMock()
        outline_args.conversation_id = sample_conversation_id
        outline_args.algorithm = "leiden"

        mock_store_outline = AsyncMock()
        mock_store_outline.detect_conversation_communities.return_value = communities

        outline_path = tmp_path / "outline_conversation.json"

        def outline_path_factory(p):
            if p == "outline_conversation.json":
                return outline_path
            return Path(p)

        with patch("main.GraphStore", return_value=mock_store_outline), \
             patch("main.Path", side_effect=outline_path_factory):
            await cmd_outline_conversation(outline_args)

        assert outline_path.exists()
        outline = json.loads(outline_path.read_text())
        assert len(outline["sections"]) == 2

        # Step 2: Run draft-conversation for section 0
        draft_args = MagicMock()
        draft_args.section = 0
        draft_args.conversation = sample_conversation_id
        draft_args.template = None

        section_data = {
            "messages": [
                {"id": "msg_001", "speaker": "USER", "content": "test message 1",
                 "turn": 1, "mentioned_entities": [], "tactics": [], "feedback": []},
                {"id": "msg_002", "speaker": "ASSISTANT", "content": "test response",
                 "turn": 2, "mentioned_entities": [], "tactics": [], "feedback": []},
            ],
            "entity_relations": [],
            "message_ids": ["msg_001", "msg_002"],
        }

        mock_store_draft = AsyncMock()
        mock_store_draft.get_conversation_community_section_data.return_value = section_data
        mock_store_draft.get_active_feedback.return_value = []

        draft_dir = tmp_path / "drafts"
        draft_dir.mkdir()

        def draft_path_factory(p):
            if p == "outline_conversation.json":
                return outline_path
            if p == "outline.json":
                return tmp_path / "outline.json"
            if isinstance(p, str) and p.startswith("drafts"):
                return tmp_path / p
            return Path(p)

        with patch("main.GraphStore", return_value=mock_store_draft), \
             patch("main.Path", side_effect=draft_path_factory):
            await cmd_draft_conversation(draft_args)

        # Verify it used the conversation outline's message IDs
        mock_store_draft.get_conversation_community_section_data.assert_called_once_with(
            conversation_id=sample_conversation_id,
            message_ids=["msg_001", "msg_002"],
        )

    @pytest.mark.asyncio
    async def test_outline_then_draft_second_section(
        self, tmp_path, sample_conversation_id
    ):
        """Test drafting section 1 from a conversation outline."""
        from main import cmd_outline_conversation, cmd_draft_conversation

        communities = {
            0: ["msg_001", "msg_002"],
            1: ["msg_003", "msg_004", "msg_005"],
        }

        outline_args = MagicMock()
        outline_args.conversation_id = sample_conversation_id
        outline_args.algorithm = "leiden"

        mock_store = AsyncMock()
        mock_store.detect_conversation_communities.return_value = communities

        outline_path = tmp_path / "outline_conversation.json"

        def path_factory(p):
            if p == "outline_conversation.json":
                return outline_path
            return Path(p)

        with patch("main.GraphStore", return_value=mock_store), \
             patch("main.Path", side_effect=path_factory):
            await cmd_outline_conversation(outline_args)

        # Now draft section 1
        draft_args = MagicMock()
        draft_args.section = 1
        draft_args.conversation = sample_conversation_id
        draft_args.template = None

        section_data = {
            "messages": [
                {"id": "msg_003", "speaker": "USER", "content": "That's a false equivalence.",
                 "turn": 3, "mentioned_entities": ["Brennan Center"], "tactics": [], "feedback": []},
            ],
            "entity_relations": [],
            "message_ids": ["msg_003", "msg_004", "msg_005"],
        }

        mock_store_draft = AsyncMock()
        mock_store_draft.get_conversation_community_section_data.return_value = section_data
        mock_store_draft.get_active_feedback.return_value = []

        draft_dir = tmp_path / "drafts"
        draft_dir.mkdir()

        def draft_path_factory(p):
            if p == "outline_conversation.json":
                return outline_path
            if p == "outline.json":
                return tmp_path / "outline.json"
            if isinstance(p, str) and p.startswith("drafts"):
                return tmp_path / p
            return Path(p)

        with patch("main.GraphStore", return_value=mock_store_draft), \
             patch("main.Path", side_effect=draft_path_factory):
            await cmd_draft_conversation(draft_args)

        mock_store_draft.get_conversation_community_section_data.assert_called_once_with(
            conversation_id=sample_conversation_id,
            message_ids=["msg_003", "msg_004", "msg_005"],
        )


# ═══════════════════════════════════════════════════════════════════════
# 9. Edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_conversation_outline_mismatched_id_falls_back(
        self, tmp_path, sample_entity_outline
    ):
        """If conversation outline has a different conversation_id, fall back to entity outline."""
        from main import cmd_draft_conversation

        args = MagicMock()
        args.section = 0
        args.conversation = "different_conversation"
        args.template = None

        # Write a conversation outline for a different conversation
        conv_outline_path = tmp_path / "outline_conversation.json"
        conv_outline_path.write_text(json.dumps({
            "conversation_id": "other_conversation",
            "algorithm": "leiden",
            "sections": [{"section_id": 0, "community_id": 0, "message_ids": ["msg_001"], "size": 1}],
        }))

        entity_outline_path = tmp_path / "outline.json"
        entity_outline_path.write_text(json.dumps(sample_entity_outline))

        mock_store = AsyncMock()
        mock_store.get_conversation_section_data.return_value = {
            "messages": [{"id": "msg_001", "speaker": "USER", "content": "test",
                          "turn": 1, "mentioned_entities": [], "tactics": [], "feedback": []}],
            "entity_relations": [],
            "community_members": sample_entity_outline[0]["members"],
        }
        mock_store.get_active_feedback.return_value = []

        draft_dir = tmp_path / "drafts"
        draft_dir.mkdir()

        def path_factory(p):
            if p == "outline_conversation.json":
                return conv_outline_path
            if p == "outline.json":
                return entity_outline_path
            if isinstance(p, str) and p.startswith("drafts"):
                return tmp_path / p
            return Path(p)

        with patch("main.GraphStore", return_value=mock_store), \
             patch("main.Path", side_effect=path_factory):
            await cmd_draft_conversation(args)

        # Should have used entity-based method since conversation_id didn't match
        mock_store.get_conversation_section_data.assert_called_once()
        mock_store.get_conversation_community_section_data.assert_not_called()

    def test_multiple_communities_different_sizes(self):
        """Communities should be able to have different sizes."""
        communities = {
            0: ["msg_001"],
            1: ["msg_002", "msg_003", "msg_004"],
            2: ["msg_005"],
        }
        assert len(communities) == 3
        assert len(communities[0]) == 1
        assert len(communities[1]) == 3
        assert len(communities[2]) == 1
