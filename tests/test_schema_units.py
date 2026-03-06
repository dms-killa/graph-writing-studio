"""
Unit tests for schema.py — validators, computed properties, and constraints.

These tests catch bugs without any external dependencies (no Ollama, no Neo4j).
Each test exercises one path through the Pydantic models.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError


# ═══════════════════════════════════════════════════════════════════════
# Relation model
# ═══════════════════════════════════════════════════════════════════════

class TestRelation:

    def test_target_entity_is_title_cased(self):
        from schema import Relation, RelationshipType
        rel = Relation(
            target_entity="  john smith  ",
            relationship_type=RelationshipType.WORKS_FOR,
            context="He works there.",
        )
        assert rel.target_entity == "John Smith"

    def test_target_entity_already_title_case_unchanged(self):
        from schema import Relation, RelationshipType
        rel = Relation(
            target_entity="Acme Corp",
            relationship_type=RelationshipType.LEADS,
            context="She leads Acme Corp.",
        )
        assert rel.target_entity == "Acme Corp"

    def test_confidence_default_is_0_8(self):
        from schema import Relation, RelationshipType
        rel = Relation(
            target_entity="Target",
            relationship_type=RelationshipType.RELATED_TO,
            context="Some context",
        )
        assert rel.confidence == 0.8

    def test_confidence_upper_bound(self):
        from schema import Relation, RelationshipType
        rel = Relation(
            target_entity="Target",
            relationship_type=RelationshipType.RELATED_TO,
            context="ctx",
            confidence=1.0,
        )
        assert rel.confidence == 1.0

    def test_confidence_lower_bound(self):
        from schema import Relation, RelationshipType
        rel = Relation(
            target_entity="Target",
            relationship_type=RelationshipType.RELATED_TO,
            context="ctx",
            confidence=0.0,
        )
        assert rel.confidence == 0.0

    def test_confidence_above_1_raises(self):
        from schema import Relation, RelationshipType
        with pytest.raises(ValidationError):
            Relation(
                target_entity="Target",
                relationship_type=RelationshipType.RELATED_TO,
                context="ctx",
                confidence=1.1,
            )

    def test_confidence_below_0_raises(self):
        from schema import Relation, RelationshipType
        with pytest.raises(ValidationError):
            Relation(
                target_entity="Target",
                relationship_type=RelationshipType.RELATED_TO,
                context="ctx",
                confidence=-0.1,
            )

    def test_context_max_length_300_enforced(self):
        from schema import Relation, RelationshipType
        long_context = "x" * 301
        with pytest.raises(ValidationError):
            Relation(
                target_entity="Target",
                relationship_type=RelationshipType.RELATED_TO,
                context=long_context,
            )

    def test_context_exactly_300_chars_allowed(self):
        from schema import Relation, RelationshipType
        ok_context = "x" * 300
        rel = Relation(
            target_entity="Target",
            relationship_type=RelationshipType.RELATED_TO,
            context=ok_context,
        )
        assert len(rel.context) == 300

    def test_valid_from_and_valid_to_default_to_none(self):
        from schema import Relation, RelationshipType
        rel = Relation(
            target_entity="Target",
            relationship_type=RelationshipType.STARTED,
            context="ctx",
        )
        assert rel.valid_from is None
        assert rel.valid_to is None

    def test_valid_from_accepts_datetime(self):
        from schema import Relation, RelationshipType
        dt = datetime(2023, 1, 15)
        rel = Relation(
            target_entity="Target",
            relationship_type=RelationshipType.STARTED,
            context="ctx",
            valid_from=dt,
        )
        assert rel.valid_from == dt

    def test_invalid_relationship_type_raises(self):
        from schema import Relation
        with pytest.raises(ValidationError):
            Relation(
                target_entity="Target",
                relationship_type="NOT_A_REAL_TYPE",
                context="ctx",
            )


# ═══════════════════════════════════════════════════════════════════════
# Entity model
# ═══════════════════════════════════════════════════════════════════════

class TestEntity:

    def test_name_is_title_cased(self):
        from schema import Entity, EntityLabel
        e = Entity(name="  john smith  ", label=EntityLabel.PERSON)
        assert e.name == "John Smith"

    def test_name_all_caps_becomes_title(self):
        from schema import Entity, EntityLabel
        e = Entity(name="ACME CORP", label=EntityLabel.ORGANIZATION)
        assert e.name == "Acme Corp"

    def test_aliases_default_to_empty_list(self):
        from schema import Entity, EntityLabel
        e = Entity(name="Alice", label=EntityLabel.PERSON)
        assert e.aliases == []

    def test_relations_default_to_empty_list(self):
        from schema import Entity, EntityLabel
        e = Entity(name="Alice", label=EntityLabel.PERSON)
        assert e.relations == []

    def test_invalid_label_raises(self):
        from schema import Entity
        with pytest.raises(ValidationError):
            Entity(name="Alice", label="ANIMAL")

    def test_all_valid_labels_accepted(self):
        from schema import Entity, EntityLabel
        for label in EntityLabel:
            e = Entity(name="Test", label=label)
            assert e.label == label


# ═══════════════════════════════════════════════════════════════════════
# Episode model
# ═══════════════════════════════════════════════════════════════════════

class TestEpisode:

    def test_episode_defaults(self):
        from schema import Episode
        ep = Episode(source_id="test_001", raw_text="Some text.")
        assert ep.source_type == "contact"
        assert ep.entities == []
        assert ep.source_timestamp is None
        assert isinstance(ep.ingested_at, datetime)

    def test_episode_with_entities(self):
        from schema import Episode, Entity, EntityLabel
        entities = [
            Entity(name="Alice", label=EntityLabel.PERSON),
            Entity(name="Acme", label=EntityLabel.ORGANIZATION),
        ]
        ep = Episode(source_id="test_002", raw_text="text", entities=entities)
        assert len(ep.entities) == 2


# ═══════════════════════════════════════════════════════════════════════
# ConversationEpisode model
# ═══════════════════════════════════════════════════════════════════════

class TestConversationEpisode:

    def _make_message(self, msg_id: str, content: str = "hello"):
        from schema import Message, SpeakerRole
        return Message(id=msg_id, speaker=SpeakerRole.USER, content=content)

    def test_unique_message_ids_accepted(self):
        from schema import ConversationEpisode
        msgs = [self._make_message("msg_001"), self._make_message("msg_002")]
        ep = ConversationEpisode(source_id="conv_01", raw_text="t", messages=msgs)
        assert len(ep.messages) == 2

    def test_duplicate_message_ids_raise_validation_error(self):
        from schema import ConversationEpisode
        msgs = [self._make_message("msg_001"), self._make_message("msg_001")]
        with pytest.raises(ValidationError, match="Duplicate message IDs"):
            ConversationEpisode(source_id="conv_01", raw_text="t", messages=msgs)

    def test_empty_messages_allowed(self):
        from schema import ConversationEpisode
        ep = ConversationEpisode(source_id="conv_01", raw_text="t", messages=[])
        assert ep.messages == []

    def test_source_type_default_is_conversation(self):
        from schema import ConversationEpisode
        ep = ConversationEpisode(source_id="conv_01", raw_text="t")
        assert ep.source_type == "conversation"


# ═══════════════════════════════════════════════════════════════════════
# Message model
# ═══════════════════════════════════════════════════════════════════════

class TestMessage:

    def test_message_basic_fields(self):
        from schema import Message, SpeakerRole
        msg = Message(id="msg_001", speaker=SpeakerRole.USER, content="Hello!")
        assert msg.id == "msg_001"
        assert msg.speaker == SpeakerRole.USER
        assert msg.content == "Hello!"

    def test_message_defaults(self):
        from schema import Message, SpeakerRole
        msg = Message(id="msg_001", speaker=SpeakerRole.ASSISTANT, content="Hi")
        assert msg.entities_mentioned == []
        assert msg.tactical_moves == []
        assert msg.timestamp is None

    def test_invalid_speaker_raises(self):
        from schema import Message
        with pytest.raises(ValidationError):
            Message(id="msg_001", speaker="robot", content="beep")


# ═══════════════════════════════════════════════════════════════════════
# TacticalMove model
# ═══════════════════════════════════════════════════════════════════════

class TestTacticalMove:

    def test_valid_tactical_move(self):
        from schema import TacticalMove, TacticalMoveType
        move = TacticalMove(
            move_type=TacticalMoveType.EVASION,
            evidence="The assistant avoided answering directly.",
        )
        assert move.move_type == TacticalMoveType.EVASION
        assert move.confidence == 0.8  # default

    def test_evidence_max_length_500_enforced(self):
        from schema import TacticalMove, TacticalMoveType
        with pytest.raises(ValidationError):
            TacticalMove(
                move_type=TacticalMoveType.DEFLECTION,
                evidence="x" * 501,
            )

    def test_confidence_bounds_on_tactical_move(self):
        from schema import TacticalMove, TacticalMoveType
        with pytest.raises(ValidationError):
            TacticalMove(
                move_type=TacticalMoveType.HEDGING,
                evidence="some evidence",
                confidence=1.5,
            )

    def test_all_tactic_types_accepted(self):
        from schema import TacticalMove, TacticalMoveType
        for ttype in TacticalMoveType:
            move = TacticalMove(move_type=ttype, evidence="ev")
            assert move.move_type == ttype


# ═══════════════════════════════════════════════════════════════════════
# FeedbackNode model
# ═══════════════════════════════════════════════════════════════════════

class TestFeedbackNode:

    def test_feedback_defaults(self):
        from schema import FeedbackNode, FeedbackType
        fb = FeedbackNode(
            feedback_type=FeedbackType.AVOID_TOPIC,
            instruction="Do not mention salaries.",
        )
        assert fb.active is True
        assert fb.target_entity is None
        assert fb.target_community is None

    def test_instruction_max_length_500_enforced(self):
        from schema import FeedbackNode, FeedbackType
        with pytest.raises(ValidationError):
            FeedbackNode(
                feedback_type=FeedbackType.PREFER_STYLE,
                instruction="x" * 501,
            )

    def test_all_feedback_types_accepted(self):
        from schema import FeedbackNode, FeedbackType
        for ftype in FeedbackType:
            fb = FeedbackNode(feedback_type=ftype, instruction="some instruction")
            assert fb.feedback_type == ftype


# ═══════════════════════════════════════════════════════════════════════
# DensityReport computed properties
# ═══════════════════════════════════════════════════════════════════════

class TestDensityReport:

    def _make_report(self, **kwargs):
        from schema import DensityReport
        defaults = dict(
            section_id="sec_1",
            total_tokens=100,
            unique_entities_referenced=5,
            unique_edges_referenced=3,
            subgraph_entities_available=10,
            subgraph_edges_available=15,
        )
        defaults.update(kwargs)
        return DensityReport(**defaults)

    def test_entity_density_normal(self):
        report = self._make_report(
            unique_entities_referenced=5,
            subgraph_entities_available=10,
        )
        assert report.entity_density == pytest.approx(0.5)

    def test_entity_density_zero_when_no_entities_available(self):
        report = self._make_report(
            unique_entities_referenced=0,
            subgraph_entities_available=0,
        )
        assert report.entity_density == 0.0

    def test_entity_density_full_coverage(self):
        report = self._make_report(
            unique_entities_referenced=10,
            subgraph_entities_available=10,
        )
        assert report.entity_density == pytest.approx(1.0)

    def test_edge_density_normal(self):
        report = self._make_report(
            unique_edges_referenced=6,
            subgraph_edges_available=15,
        )
        assert report.edge_density == pytest.approx(0.4)

    def test_edge_density_zero_when_no_edges_available(self):
        report = self._make_report(
            unique_edges_referenced=0,
            subgraph_edges_available=0,
        )
        assert report.edge_density == 0.0

    def test_facts_per_100_tokens_normal(self):
        # 10 edges / 200 tokens * 100 = 5.0
        report = self._make_report(
            unique_edges_referenced=10,
            total_tokens=200,
        )
        assert report.facts_per_100_tokens == pytest.approx(5.0)

    def test_facts_per_100_tokens_zero_when_no_tokens(self):
        report = self._make_report(total_tokens=0, unique_edges_referenced=5)
        assert report.facts_per_100_tokens == 0.0

    def test_facts_per_100_tokens_zero_edges(self):
        report = self._make_report(unique_edges_referenced=0, total_tokens=100)
        assert report.facts_per_100_tokens == 0.0

    def test_density_report_negative_fields_rejected(self):
        from schema import DensityReport
        with pytest.raises(ValidationError):
            DensityReport(
                section_id="s",
                total_tokens=-1,
                unique_entities_referenced=0,
                unique_edges_referenced=0,
                subgraph_entities_available=0,
                subgraph_edges_available=0,
            )

    def test_density_report_zero_fields_allowed(self):
        report = self._make_report(
            total_tokens=0,
            unique_entities_referenced=0,
            unique_edges_referenced=0,
            subgraph_entities_available=0,
            subgraph_edges_available=0,
        )
        assert report.entity_density == 0.0
        assert report.edge_density == 0.0
        assert report.facts_per_100_tokens == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Enum completeness (guard against accidental removal)
# ═══════════════════════════════════════════════════════════════════════

class TestEnumCompleteness:

    def test_entity_labels_count(self):
        from schema import EntityLabel
        assert len(EntityLabel) == 8

    def test_relationship_types_count(self):
        from schema import RelationshipType
        # 20 types: 5 professional + 6 project lifecycle + 3 knowledge +
        # 2 temporal + 1 generic + 3 conversation
        assert len(RelationshipType) == 20

    def test_tactical_move_types_count(self):
        from schema import TacticalMoveType
        # 15 types (14 named + OTHER)
        assert len(TacticalMoveType) == 15

    def test_speaker_roles_count(self):
        from schema import SpeakerRole
        assert len(SpeakerRole) == 3

    def test_feedback_types_count(self):
        from schema import FeedbackType
        assert len(FeedbackType) == 6
