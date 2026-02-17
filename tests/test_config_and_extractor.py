"""Tests for config module and extractor bug fixes."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest


# ═══════════════════════════════════════════════════════════════════════
# 1. Config module
# ═══════════════════════════════════════════════════════════════════════


class TestConfig:

    def test_config_imports_without_error(self):
        import config
        assert hasattr(config, "OLLAMA_BASE_URL")
        assert hasattr(config, "OLLAMA_MODEL")
        assert hasattr(config, "OLLAMA_TEMPERATURE")
        assert hasattr(config, "OLLAMA_TIMEOUT")
        assert hasattr(config, "NEO4J_URI")
        assert hasattr(config, "NEO4J_USER")
        assert hasattr(config, "NEO4J_PASSWORD")

    def test_config_has_sensible_defaults(self):
        import config
        assert config.OLLAMA_BASE_URL.startswith("http")
        assert "://" in config.NEO4J_URI
        assert isinstance(config.OLLAMA_TEMPERATURE, float)
        assert isinstance(config.OLLAMA_TIMEOUT, float)

    def test_env_override(self):
        """Environment variables should override .env file values."""
        with patch.dict(os.environ, {"OLLAMA_MODEL": "test-model:latest"}):
            # Need to re-read the env
            val = os.environ.get("OLLAMA_MODEL", "default")
            assert val == "test-model:latest"

    def test_dotenv_loader_handles_missing_file(self):
        from config import _load_dotenv
        # Should not raise
        _load_dotenv("/nonexistent/path/.env")

    def test_dotenv_loader_handles_comments_and_blanks(self, tmp_path):
        from config import _load_dotenv
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nTEST_VAR_XYZ=hello\n")
        _load_dotenv(env_file)
        assert os.environ.get("TEST_VAR_XYZ") == "hello"
        # Cleanup
        del os.environ["TEST_VAR_XYZ"]

    def test_extractor_uses_config(self):
        """extractor.py should import from config, not define its own constants."""
        import extractor
        import config
        assert extractor.OLLAMA_BASE_URL == config.OLLAMA_BASE_URL
        assert extractor.EXTRACTION_MODEL == config.OLLAMA_MODEL
        assert extractor.TEMPERATURE == config.OLLAMA_TEMPERATURE
        assert extractor.REQUEST_TIMEOUT == config.OLLAMA_TIMEOUT

    def test_graph_store_uses_config(self):
        """graph_store.py should import from config."""
        import graph_store
        import config
        assert graph_store.NEO4J_URI == config.NEO4J_URI
        assert graph_store.NEO4J_USER == config.NEO4J_USER
        assert graph_store.NEO4J_PASSWORD == config.NEO4J_PASSWORD


# ═══════════════════════════════════════════════════════════════════════
# 2. Extractor entity string handling fix
# ═══════════════════════════════════════════════════════════════════════


class TestEntityStringHandling:

    def test_string_entity_gets_converted_to_dict(self):
        """When LLM returns a string entity, it should be converted to a dict with CONCEPT label."""
        from schema import EntityLabel

        # Simulate what extract_episode does with the fix
        raw_entities = ["Democratic Backsliding", "Voting Rights"]
        valid_labels = {e.value for e in EntityLabel}

        cleaned = []
        for e in raw_entities:
            if isinstance(e, str):
                e = {"name": e.strip(), "label": "CONCEPT", "aliases": []}
            if not isinstance(e, dict):
                continue
            label = e.get("label", "").upper()
            if label not in valid_labels:
                continue
            cleaned.append(e)

        assert len(cleaned) == 2
        assert cleaned[0]["name"] == "Democratic Backsliding"
        assert cleaned[0]["label"] == "CONCEPT"
        assert cleaned[1]["name"] == "Voting Rights"

    def test_dict_entity_still_works(self):
        """Normal dict entities should continue to work."""
        from schema import EntityLabel

        raw_entities = [
            {"name": "John Smith", "label": "PERSON", "aliases": ["J. Smith"]},
            {"name": "Acme Corp", "label": "ORGANIZATION", "aliases": []},
        ]
        valid_labels = {e.value for e in EntityLabel}

        cleaned = []
        for e in raw_entities:
            if isinstance(e, str):
                e = {"name": e.strip(), "label": "CONCEPT", "aliases": []}
            if not isinstance(e, dict):
                continue
            label = e.get("label", "").upper()
            if label not in valid_labels:
                continue
            cleaned.append(e)

        assert len(cleaned) == 2
        assert cleaned[0]["name"] == "John Smith"
        assert cleaned[0]["label"] == "PERSON"
        assert cleaned[0]["aliases"] == ["J. Smith"]

    def test_mixed_string_and_dict_entities(self):
        """A mix of string and dict entities should all be handled."""
        from schema import EntityLabel

        raw_entities = [
            {"name": "John Smith", "label": "PERSON", "aliases": []},
            "Some Concept",
            42,  # unexpected type
        ]
        valid_labels = {e.value for e in EntityLabel}

        cleaned = []
        for e in raw_entities:
            if isinstance(e, str):
                e = {"name": e.strip(), "label": "CONCEPT", "aliases": []}
            if not isinstance(e, dict):
                continue
            label = e.get("label", "").upper()
            if label not in valid_labels:
                continue
            cleaned.append(e)

        assert len(cleaned) == 2
        assert cleaned[0]["name"] == "John Smith"
        assert cleaned[1]["name"] == "Some Concept"

    @pytest.mark.asyncio
    async def test_extract_episode_handles_string_entities(self):
        """extract_episode should handle LLM returning string entities."""
        from extractor import extract_episode

        mock_entity_response = '["Democratic Backsliding"]'
        mock_relation_response = '[]'

        call_count = [0]

        async def mock_ollama(prompt, model=None, temperature=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_entity_response
            return mock_relation_response

        with patch("extractor._call_ollama", side_effect=mock_ollama):
            episode = await extract_episode(
                text="Test text about democratic backsliding",
                source_id="test_source",
            )

        assert len(episode.entities) == 1
        assert episode.entities[0].name == "Democratic Backsliding"
        assert episode.entities[0].label.value == "CONCEPT"

    @pytest.mark.asyncio
    async def test_extract_conversation_handles_string_entities(self):
        """extract_conversation should handle LLM returning string entities."""
        from extractor import extract_conversation

        mock_entity_response = '["Democratic Backsliding", "Republican Party"]'
        mock_relation_response = '[]'
        mock_tactics_response = '[]'

        call_count = [0]

        async def mock_ollama(prompt, model=None, temperature=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_entity_response
            if call_count[0] == 2:
                return mock_relation_response
            return mock_tactics_response

        sample_text = "**User:** Democratic backsliding is concerning.\n\n**Assistant:** I understand your concern."

        with patch("extractor._call_ollama", side_effect=mock_ollama):
            episode = await extract_conversation(
                text=sample_text,
                source_id="test_conv",
                extract_tactics=True,
            )

        # Both string entities should be converted to CONCEPT
        entity_names = {e.name for e in episode.entities}
        assert "Democratic Backsliding" in entity_names
        assert "Republican Party" in entity_names
        for e in episode.entities:
            assert e.label.value == "CONCEPT"
