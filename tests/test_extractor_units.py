"""
Unit tests for extractor.py — pure functions that need no network or database.

Covers:
  - _parse_json_array:  all recovery paths for malformed LLM output
  - parse_conversation: all supported chat transcript formats + speaker mappings
  - extract_relations_raw: filtering logic (missing fields, invalid types)
  - extract_conversation_tactics: confidence/type filtering, assistant-only logic
  - extract_episode / extract_conversation: end-to-end with mocked _call_ollama
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════
# _parse_json_array — all recovery branches
# ═══════════════════════════════════════════════════════════════════════

class TestParseJsonArray:

    def _parse(self, raw: str):
        from extractor import _parse_json_array
        return _parse_json_array(raw)

    # ── happy-path ──────────────────────────────────────────────────

    def test_clean_json_array(self):
        result = self._parse('[{"name": "Alice", "label": "PERSON"}]')
        assert result == [{"name": "Alice", "label": "PERSON"}]

    def test_clean_empty_array(self):
        result = self._parse("[]")
        assert result == []

    def test_single_object_wrapped_in_array(self):
        result = self._parse('{"name": "Bob"}')
        assert result == [{"name": "Bob"}]

    def test_array_with_multiple_items(self):
        raw = '[{"a": 1}, {"b": 2}, {"c": 3}]'
        result = self._parse(raw)
        assert len(result) == 3
        assert result[0] == {"a": 1}

    # ── markdown fence stripping ────────────────────────────────────

    def test_markdown_json_fence(self):
        raw = '```json\n[{"name": "Alice"}]\n```'
        result = self._parse(raw)
        assert result == [{"name": "Alice"}]

    def test_markdown_generic_fence(self):
        raw = '```\n[{"name": "Bob"}]\n```'
        result = self._parse(raw)
        assert result == [{"name": "Bob"}]

    def test_markdown_fence_with_surrounding_whitespace(self):
        raw = "  ```json\n  [{\"x\": 1}]\n  ```  "
        result = self._parse(raw)
        assert result == [{"x": 1}]

    # ── trailing commas ─────────────────────────────────────────────

    def test_trailing_comma_in_object(self):
        raw = '[{"name": "Alice", "label": "PERSON",}]'
        result = self._parse(raw)
        assert result[0]["name"] == "Alice"

    def test_trailing_comma_in_array(self):
        raw = '[{"a": 1}, {"b": 2},]'
        result = self._parse(raw)
        assert len(result) == 2

    def test_trailing_comma_nested(self):
        raw = '[{"a": {"x": 1,}, "b": 2,},]'
        result = self._parse(raw)
        assert result[0]["b"] == 2

    # ── preamble text before JSON ────────────────────────────────────

    def test_preamble_before_array(self):
        raw = 'Here are the entities I found:\n[{"name": "Carol"}]'
        result = self._parse(raw)
        assert result == [{"name": "Carol"}]

    def test_preamble_before_object(self):
        raw = 'Sure, here is a single entity: {"name": "Dave"}'
        result = self._parse(raw)
        assert result == [{"name": "Dave"}]

    def test_long_preamble_with_array(self):
        raw = (
            "I analyzed the text carefully and extracted the following entities "
            "in JSON format as requested:\n\n"
            '[{"name": "Eve", "label": "PERSON"}]'
        )
        result = self._parse(raw)
        assert result[0]["name"] == "Eve"

    # ── last-ditch scraping of individual objects ────────────────────

    def test_multiple_objects_separated_by_text(self):
        # Not a valid JSON array, but individual objects can be scraped
        raw = 'First: {"a": 1} and second: {"b": 2}'
        result = self._parse(raw)
        assert len(result) == 2
        assert {"a": 1} in result
        assert {"b": 2} in result

    def test_single_scraped_object(self):
        raw = 'The entity is {"name": "Frank", "label": "PERSON"} as shown.'
        result = self._parse(raw)
        assert result == [{"name": "Frank", "label": "PERSON"}]

    # ── error case ──────────────────────────────────────────────────

    def test_no_json_raises_value_error(self):
        from extractor import _parse_json_array
        with pytest.raises(ValueError, match="No JSON array found"):
            _parse_json_array("This is just plain English with no JSON at all.")

    def test_only_whitespace_raises_value_error(self):
        from extractor import _parse_json_array
        with pytest.raises(ValueError):
            _parse_json_array("   \n\t  ")


# ═══════════════════════════════════════════════════════════════════════
# parse_conversation — all format variants
# ═══════════════════════════════════════════════════════════════════════

class TestParseConversation:

    def _parse(self, text: str):
        from extractor import parse_conversation
        return parse_conversation(text)

    # ── format variants ─────────────────────────────────────────────

    def test_plain_colon_format(self):
        text = "User: Hello there\nAssistant: Hi!"
        msgs = self._parse(text)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "Hello there"
        assert msgs[1]["content"] == "Hi!"

    def test_markdown_bold_format(self):
        text = "**User:** What is AI?\n\n**Assistant:** AI stands for artificial intelligence."
        msgs = self._parse(text)
        assert len(msgs) == 2
        assert "What is AI?" in msgs[0]["content"]
        assert "artificial intelligence" in msgs[1]["content"]

    def test_heading_format(self):
        text = "### User\nWhat's up?\n\n### Assistant\nNot much."
        msgs = self._parse(text)
        assert len(msgs) == 2
        assert "What's up?" in msgs[0]["content"]

    def test_blockquote_format(self):
        text = "> **User:** Blockquote question\n> **Assistant:** Blockquote answer"
        msgs = self._parse(text)
        assert len(msgs) == 2
        assert "Blockquote question" in msgs[0]["content"]

    def test_multiline_message_content(self):
        text = (
            "User: First line\nSecond line\nThird line\n"
            "Assistant: Reply"
        )
        msgs = self._parse(text)
        assert len(msgs) == 2
        assert "Second line" in msgs[0]["content"]
        assert "Third line" in msgs[0]["content"]

    def test_empty_transcript_returns_empty_list(self):
        msgs = self._parse("")
        assert msgs == []

    def test_no_speaker_labels_returns_empty_list(self):
        msgs = self._parse("This is just a paragraph with no speaker labels.")
        assert msgs == []

    def test_single_message(self):
        text = "User: Only one message here."
        msgs = self._parse(text)
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Only one message here."

    # ── speaker mapping ──────────────────────────────────────────────

    def test_human_maps_to_user_role(self):
        from schema import SpeakerRole
        msgs = self._parse("Human: Hello\nAssistant: Hi")
        assert msgs[0]["speaker"] == SpeakerRole.USER

    def test_ai_maps_to_assistant_role(self):
        from schema import SpeakerRole
        msgs = self._parse("User: Hi\nAI: Hello there")
        assert msgs[1]["speaker"] == SpeakerRole.ASSISTANT

    def test_claude_maps_to_assistant_role(self):
        from schema import SpeakerRole
        msgs = self._parse("User: Hi\nClaude: Hello!")
        assert msgs[1]["speaker"] == SpeakerRole.ASSISTANT

    def test_chatgpt_maps_to_assistant_role(self):
        from schema import SpeakerRole
        msgs = self._parse("User: Hi\nChatGPT: Hello!")
        assert msgs[1]["speaker"] == SpeakerRole.ASSISTANT

    def test_system_maps_to_system_role(self):
        from schema import SpeakerRole
        msgs = self._parse("System: System prompt here\nUser: Hi")
        assert msgs[0]["speaker"] == SpeakerRole.SYSTEM

    def test_case_insensitive_speaker_labels(self):
        from schema import SpeakerRole
        msgs = self._parse("user: hello\nassistant: world")
        assert msgs[0]["speaker"] == SpeakerRole.USER
        assert msgs[1]["speaker"] == SpeakerRole.ASSISTANT

    # ── content on same line as speaker label ────────────────────────

    def test_content_on_same_line_as_label(self):
        text = "User: Inline content here\nAssistant: Inline reply"
        msgs = self._parse(text)
        assert msgs[0]["content"] == "Inline content here"
        assert msgs[1]["content"] == "Inline reply"

    def test_content_strip_trailing_blank_lines(self):
        text = "User: Hello\n\n\nAssistant: Hi"
        msgs = self._parse(text)
        # Content should be stripped
        assert msgs[0]["content"] == "Hello"

    # ── ordering preserved ───────────────────────────────────────────

    def test_message_order_preserved(self):
        from schema import SpeakerRole
        text = (
            "User: First\n"
            "Assistant: Second\n"
            "User: Third\n"
            "Assistant: Fourth"
        )
        msgs = self._parse(text)
        assert len(msgs) == 4
        assert msgs[0]["speaker"] == SpeakerRole.USER
        assert msgs[1]["speaker"] == SpeakerRole.ASSISTANT
        assert msgs[2]["speaker"] == SpeakerRole.USER
        assert msgs[3]["speaker"] == SpeakerRole.ASSISTANT
        assert "First" in msgs[0]["content"]
        assert "Fourth" in msgs[3]["content"]


# ═══════════════════════════════════════════════════════════════════════
# extract_relations_raw — filtering invalid raw dicts
# ═══════════════════════════════════════════════════════════════════════

class TestExtractRelationsRawFiltering:
    """
    Tests for the pre-validation filtering inside extract_relations_raw.

    We mock _call_ollama so we control exactly what "LLM" returns.
    """

    @pytest.mark.asyncio
    async def test_valid_relation_passes_through(self):
        from extractor import extract_relations_raw
        raw = '[{"source_entity": "Alice", "target_entity": "Acme", "relationship_type": "WORKS_FOR"}]'
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value=raw):
            result = await extract_relations_raw("text", [
                {"name": "Alice", "label": "PERSON"},
                {"name": "Acme", "label": "ORGANIZATION"},
            ])
        assert len(result) == 1
        assert result[0]["relationship_type"] == "WORKS_FOR"

    @pytest.mark.asyncio
    async def test_missing_source_entity_skipped(self):
        from extractor import extract_relations_raw
        raw = '[{"target_entity": "Acme", "relationship_type": "WORKS_FOR"}]'
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value=raw):
            result = await extract_relations_raw("text", [
                {"name": "Alice", "label": "PERSON"},
                {"name": "Acme", "label": "ORGANIZATION"},
            ])
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_target_entity_skipped(self):
        from extractor import extract_relations_raw
        raw = '[{"source_entity": "Alice", "relationship_type": "WORKS_FOR"}]'
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value=raw):
            result = await extract_relations_raw("text", [{"name": "Alice", "label": "PERSON"}])
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_relationship_type_skipped(self):
        from extractor import extract_relations_raw
        raw = '[{"source_entity": "Alice", "target_entity": "Acme"}]'
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value=raw):
            result = await extract_relations_raw("text", [
                {"name": "Alice", "label": "PERSON"},
                {"name": "Acme", "label": "ORGANIZATION"},
            ])
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_source_entity_string_skipped(self):
        from extractor import extract_relations_raw
        raw = '[{"source_entity": "", "target_entity": "Acme", "relationship_type": "WORKS_FOR"}]'
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value=raw):
            result = await extract_relations_raw("text", [{"name": "Acme", "label": "ORGANIZATION"}])
        assert result == []

    @pytest.mark.asyncio
    async def test_parse_failure_returns_empty_list(self):
        from extractor import extract_relations_raw
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value="not json at all"):
            result = await extract_relations_raw("text", [{"name": "Alice", "label": "PERSON"}])
        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_relations_mixed_validity(self):
        from extractor import extract_relations_raw
        raw = '''[
            {"source_entity": "Alice", "target_entity": "Acme", "relationship_type": "WORKS_FOR"},
            {"target_entity": "Acme", "relationship_type": "LEADS"},
            {"source_entity": "Alice", "relationship_type": "EXPERT_IN"}
        ]'''
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value=raw):
            result = await extract_relations_raw("text", [
                {"name": "Alice", "label": "PERSON"},
                {"name": "Acme", "label": "ORGANIZATION"},
            ])
        # Only the first has all three required fields
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════
# extract_conversation_tactics — filtering logic
# ═══════════════════════════════════════════════════════════════════════

class TestExtractConversationTactics:

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_assistant_messages(self):
        from extractor import extract_conversation_tactics
        from schema import SpeakerRole
        msgs = [
            {"id": "msg_001", "speaker": SpeakerRole.USER, "content": "Hello"},
            {"id": "msg_002", "speaker": "user", "content": "Another user message"},
        ]
        result = await extract_conversation_tactics(msgs)
        assert result == {}

    @pytest.mark.asyncio
    async def test_tactics_below_confidence_threshold_filtered(self):
        from extractor import extract_conversation_tactics
        from schema import SpeakerRole
        msgs = [{"id": "msg_001", "speaker": SpeakerRole.ASSISTANT, "content": "Answer"}]
        low_conf_response = '''[{
            "message_id": "msg_001",
            "moves": [{"move_type": "EVASION", "evidence": "avoided", "confidence": 0.5}]
        }]'''
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value=low_conf_response):
            result = await extract_conversation_tactics(msgs)
        # 0.5 < 0.6 threshold → filtered out
        assert result == {}

    @pytest.mark.asyncio
    async def test_tactics_at_threshold_included(self):
        from extractor import extract_conversation_tactics
        from schema import SpeakerRole
        msgs = [{"id": "msg_001", "speaker": SpeakerRole.ASSISTANT, "content": "Answer"}]
        response = '''[{
            "message_id": "msg_001",
            "moves": [{"move_type": "EVASION", "evidence": "avoided directly", "confidence": 0.6}]
        }]'''
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value=response):
            result = await extract_conversation_tactics(msgs)
        assert "msg_001" in result
        assert result["msg_001"][0]["move_type"] == "EVASION"

    @pytest.mark.asyncio
    async def test_invalid_move_type_filtered(self):
        from extractor import extract_conversation_tactics
        from schema import SpeakerRole
        msgs = [{"id": "msg_001", "speaker": SpeakerRole.ASSISTANT, "content": "Answer"}]
        response = '''[{
            "message_id": "msg_001",
            "moves": [
                {"move_type": "NOT_A_REAL_TACTIC", "evidence": "ev", "confidence": 0.9},
                {"move_type": "EVASION", "evidence": "real ev", "confidence": 0.9}
            ]
        }]'''
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value=response):
            result = await extract_conversation_tactics(msgs)
        assert "msg_001" in result
        assert len(result["msg_001"]) == 1
        assert result["msg_001"][0]["move_type"] == "EVASION"

    @pytest.mark.asyncio
    async def test_parse_failure_returns_empty(self):
        from extractor import extract_conversation_tactics
        from schema import SpeakerRole
        msgs = [{"id": "msg_001", "speaker": SpeakerRole.ASSISTANT, "content": "Answer"}]
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value="not json"):
            result = await extract_conversation_tactics(msgs)
        assert result == {}

    @pytest.mark.asyncio
    async def test_message_with_no_moves_excluded_from_result(self):
        from extractor import extract_conversation_tactics
        from schema import SpeakerRole
        msgs = [
            {"id": "msg_001", "speaker": SpeakerRole.ASSISTANT, "content": "Clean answer"},
            {"id": "msg_002", "speaker": SpeakerRole.ASSISTANT, "content": "Evasive answer"},
        ]
        response = '''[
            {"message_id": "msg_001", "moves": []},
            {"message_id": "msg_002", "moves": [{"move_type": "DEFLECTION", "evidence": "ev", "confidence": 0.8}]}
        ]'''
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value=response):
            result = await extract_conversation_tactics(msgs)
        assert "msg_001" not in result
        assert "msg_002" in result

    @pytest.mark.asyncio
    async def test_evidence_truncated_to_500_chars(self):
        from extractor import extract_conversation_tactics
        from schema import SpeakerRole
        msgs = [{"id": "msg_001", "speaker": SpeakerRole.ASSISTANT, "content": "Answer"}]
        long_evidence = "x" * 600
        response = f'''[{{"message_id": "msg_001", "moves": [
            {{"move_type": "HEDGING", "evidence": "{long_evidence}", "confidence": 0.9}}
        ]}}]'''
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value=response):
            result = await extract_conversation_tactics(msgs)
        if "msg_001" in result:
            assert len(result["msg_001"][0]["evidence"]) <= 500

    @pytest.mark.asyncio
    async def test_string_speaker_assistant_also_processed(self):
        """Messages with speaker as string 'assistant' (not enum) should be included."""
        from extractor import extract_conversation_tactics
        msgs = [{"id": "msg_001", "speaker": "assistant", "content": "Answer"}]
        response = '''[{
            "message_id": "msg_001",
            "moves": [{"move_type": "EVASION", "evidence": "ev", "confidence": 0.7}]
        }]'''
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value=response):
            result = await extract_conversation_tactics(msgs)
        assert "msg_001" in result


# ═══════════════════════════════════════════════════════════════════════
# extract_episode — confidence and entity filtering
# ═══════════════════════════════════════════════════════════════════════

class TestExtractEpisode:

    @pytest.mark.asyncio
    async def test_low_confidence_relation_filtered(self):
        from extractor import extract_episode
        entity_response = '[{"name": "Alice", "label": "PERSON", "aliases": []}, {"name": "Acme", "label": "ORGANIZATION", "aliases": []}]'
        relation_response = '[{"source_entity": "Alice", "target_entity": "Acme", "relationship_type": "WORKS_FOR", "context": "ctx", "confidence": 0.3}]'
        call_count = [0]
        async def mock_ollama(prompt, model=None, temperature=None):
            call_count[0] += 1
            return entity_response if call_count[0] == 1 else relation_response
        with patch("extractor._call_ollama", side_effect=mock_ollama):
            episode = await extract_episode("text", "src_01", min_confidence=0.5)
        # Confidence 0.3 < 0.5 threshold → relation filtered
        alice = next(e for e in episode.entities if e.name == "Alice")
        assert alice.relations == []

    @pytest.mark.asyncio
    async def test_relation_with_unknown_entity_filtered(self):
        from extractor import extract_episode
        entity_response = '[{"name": "Alice", "label": "PERSON", "aliases": []}]'
        # Bob is not in the entity list
        relation_response = '[{"source_entity": "Alice", "target_entity": "Bob", "relationship_type": "WORKS_FOR", "context": "ctx", "confidence": 0.9}]'
        call_count = [0]
        async def mock_ollama(prompt, model=None, temperature=None):
            call_count[0] += 1
            return entity_response if call_count[0] == 1 else relation_response
        with patch("extractor._call_ollama", side_effect=mock_ollama):
            episode = await extract_episode("text", "src_02")
        alice = episode.entities[0]
        assert alice.relations == []

    @pytest.mark.asyncio
    async def test_invalid_entity_label_filtered(self):
        from extractor import extract_episode
        entity_response = '[{"name": "Dog", "label": "ANIMAL", "aliases": []}, {"name": "Alice", "label": "PERSON", "aliases": []}]'
        relation_response = '[]'
        call_count = [0]
        async def mock_ollama(prompt, model=None, temperature=None):
            call_count[0] += 1
            return entity_response if call_count[0] == 1 else relation_response
        with patch("extractor._call_ollama", side_effect=mock_ollama):
            episode = await extract_episode("text", "src_03")
        entity_names = {e.name for e in episode.entities}
        # "Dog" with label "ANIMAL" should be dropped
        assert "Dog" not in entity_names
        assert "Alice" in entity_names

    @pytest.mark.asyncio
    async def test_invalid_relation_type_filtered(self):
        from extractor import extract_episode
        entity_response = '[{"name": "Alice", "label": "PERSON", "aliases": []}, {"name": "Acme", "label": "ORGANIZATION", "aliases": []}]'
        relation_response = '[{"source_entity": "Alice", "target_entity": "Acme", "relationship_type": "HATES", "context": "ctx", "confidence": 0.9}]'
        call_count = [0]
        async def mock_ollama(prompt, model=None, temperature=None):
            call_count[0] += 1
            return entity_response if call_count[0] == 1 else relation_response
        with patch("extractor._call_ollama", side_effect=mock_ollama):
            episode = await extract_episode("text", "src_04")
        alice = next(e for e in episode.entities if e.name == "Alice")
        assert alice.relations == []

    @pytest.mark.asyncio
    async def test_source_timestamp_parsed(self):
        from extractor import extract_episode
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value="[]"):
            episode = await extract_episode("text", "src_05", source_timestamp="2023-06-15")
        assert episode.source_timestamp is not None
        assert episode.source_timestamp.year == 2023
        assert episode.source_timestamp.month == 6

    @pytest.mark.asyncio
    async def test_invalid_source_timestamp_ignored(self):
        from extractor import extract_episode
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value="[]"):
            episode = await extract_episode("text", "src_06", source_timestamp="not-a-date")
        assert episode.source_timestamp is None

    @pytest.mark.asyncio
    async def test_empty_extraction_produces_valid_episode(self):
        from extractor import extract_episode
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value="[]"):
            episode = await extract_episode("text", "src_07")
        assert episode.source_id == "src_07"
        assert episode.entities == []
        assert episode.raw_text == "text"

    @pytest.mark.asyncio
    async def test_entity_name_normalized_in_episode(self):
        from extractor import extract_episode
        entity_response = '[{"name": "  john smith  ", "label": "PERSON", "aliases": []}]'
        call_count = [0]
        async def mock_ollama(prompt, model=None, temperature=None):
            call_count[0] += 1
            return entity_response if call_count[0] == 1 else "[]"
        with patch("extractor._call_ollama", side_effect=mock_ollama):
            episode = await extract_episode("text about john smith", "src_08")
        assert episode.entities[0].name == "John Smith"


# ═══════════════════════════════════════════════════════════════════════
# extract_conversation — message ID assignment
# ═══════════════════════════════════════════════════════════════════════

class TestExtractConversation:

    @pytest.mark.asyncio
    async def test_message_ids_assigned_sequentially(self):
        from extractor import extract_conversation
        text = "User: Hello\nAssistant: Hi\nUser: Bye\nAssistant: Goodbye"
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value="[]"):
            episode = await extract_conversation(text, "conv_01", extract_tactics=False)
        ids = [m.id for m in episode.messages]
        assert ids == ["msg_000", "msg_001", "msg_002", "msg_003"]

    @pytest.mark.asyncio
    async def test_no_messages_raises_value_error(self):
        from extractor import extract_conversation
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value="[]"):
            with pytest.raises(ValueError, match="No messages found"):
                await extract_conversation("No speaker labels here.", "conv_02")

    @pytest.mark.asyncio
    async def test_entities_matched_to_messages(self):
        from extractor import extract_conversation
        text = "User: Tell me about Python.\nAssistant: Python is a language."
        entity_response = '[{"name": "Python", "label": "TECHNOLOGY", "aliases": []}]'
        call_count = [0]
        async def mock_ollama(prompt, model=None, temperature=None):
            call_count[0] += 1
            return entity_response if call_count[0] == 1 else "[]"
        with patch("extractor._call_ollama", side_effect=mock_ollama):
            episode = await extract_conversation(text, "conv_03", extract_tactics=False)
        # "python" appears in both messages (case-insensitive match)
        msgs_mentioning_python = [
            m for m in episode.messages if "Python" in m.entities_mentioned
        ]
        assert len(msgs_mentioning_python) == 2

    @pytest.mark.asyncio
    async def test_extract_tactics_false_skips_tactic_extraction(self):
        """When extract_tactics=False, _call_ollama should be called only twice (entities + relations)."""
        from extractor import extract_conversation
        text = "User: Hello\nAssistant: Hi"
        call_count = [0]
        async def mock_ollama(prompt, model=None, temperature=None):
            call_count[0] += 1
            return "[]"
        with patch("extractor._call_ollama", side_effect=mock_ollama):
            episode = await extract_conversation(text, "conv_04", extract_tactics=False)
        # entities call + relations call = 2, no tactics call
        assert call_count[0] == 2
        for msg in episode.messages:
            assert msg.tactical_moves == []

    @pytest.mark.asyncio
    async def test_extract_tactics_true_makes_additional_call(self):
        """When extract_tactics=True, _call_ollama should be called 3 times."""
        from extractor import extract_conversation
        text = "User: Hello\nAssistant: Hi there, great question!"
        call_count = [0]
        async def mock_ollama(prompt, model=None, temperature=None):
            call_count[0] += 1
            return "[]"
        with patch("extractor._call_ollama", side_effect=mock_ollama):
            await extract_conversation(text, "conv_05", extract_tactics=True)
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_source_timestamp_parsed_in_conversation(self):
        from extractor import extract_conversation
        text = "User: Hi\nAssistant: Hello"
        with patch("extractor._call_ollama", new_callable=AsyncMock, return_value="[]"):
            episode = await extract_conversation(
                text, "conv_06", source_timestamp="2024-01-01", extract_tactics=False
            )
        assert episode.source_timestamp is not None
        assert episode.source_timestamp.year == 2024
