"""
Schema definitions for the Graph Writing Studio.

These Pydantic models serve as the typed contract between Ollama's output
and the Neo4j graph. Every fact extracted must pass through these validators
before touching the database.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Entity Labels ────────────────────────────────────────────────────

class EntityLabel(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    PROJECT = "PROJECT"
    MILESTONE = "MILESTONE"
    TECHNOLOGY = "TECHNOLOGY"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"


# ─── Relationship Types ──────────────────────────────────────────────

class RelationshipType(str, Enum):
    # Professional
    WORKS_FOR = "WORKS_FOR"
    LEADS = "LEADS"
    PARTICIPATES_IN = "PARTICIPATES_IN"
    REPORTS_TO = "REPORTS_TO"
    COLLABORATES_WITH = "COLLABORATES_WITH"

    # Project lifecycle
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    DELAYED = "DELAYED"
    CANCELLED = "CANCELLED"
    BLOCKED_BY = "BLOCKED_BY"
    DEPENDS_ON = "DEPENDS_ON"

    # Knowledge / skills
    EXPERT_IN = "EXPERT_IN"
    USES = "USES"
    CREATED = "CREATED"

    # Temporal state changes
    TRANSITIONED_TO = "TRANSITIONED_TO"
    REPLACED_BY = "REPLACED_BY"

    # Generic fallback
    RELATED_TO = "RELATED_TO"

    # Conversation relationships
    REPLIES_TO = "REPLIES_TO"
    MENTIONS = "MENTIONS"
    EXHIBITS_TACTIC = "EXHIBITS_TACTIC"


# ─── Core Models ──────────────────────────────────────────────────────

class Relation(BaseModel):
    """A single directed relationship between two entities."""

    target_entity: str = Field(
        description="Name of the entity being related to"
    )
    relationship_type: RelationshipType = Field(
        description="The type of relationship"
    )
    context: str = Field(
        description="Brief evidence from the source text (1-2 sentences)",
        max_length=300,
    )
    valid_from: Optional[datetime] = Field(
        default=None,
        description="When this relationship became true (ISO 8601)",
    )
    valid_to: Optional[datetime] = Field(
        default=None,
        description="When this relationship ceased to be true (ISO 8601). "
        "None means still active.",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Model's confidence in this extraction (0-1)",
    )

    @field_validator("target_entity")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        return v.strip().title()


class Entity(BaseModel):
    """An entity extracted from a source document."""

    name: str = Field(description="Primary canonical name")
    label: EntityLabel = Field(description="Entity category")
    aliases: list[str] = Field(
        default_factory=list,
        description="Known alternative names (e.g., 'Bob' for 'Robert Smith')",
    )
    relations: list[Relation] = Field(
        default_factory=list,
        description="Outgoing relationships to other entities",
    )

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        return v.strip().title()


class Episode(BaseModel):
    """
    A single ingestion event — one contact record, one meeting note, etc.
    
    This is the temporal unit: all entities/relations extracted from one
    source share the same episode timestamp, enabling point-in-time queries.
    """

    source_id: str = Field(
        description="Unique identifier for the source document"
    )
    source_type: str = Field(
        default="contact",
        description="Type of source (contact, meeting_note, email, etc.)",
    )
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    source_timestamp: Optional[datetime] = Field(
        default=None,
        description="Original timestamp of the source document, if known",
    )
    entities: list[Entity] = Field(default_factory=list)
    raw_text: str = Field(
        description="The original text that was processed"
    )


# ─── Conversation Models ─────────────────────────────────────────────

class SpeakerRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TacticalMoveType(str, Enum):
    """Taxonomy of rhetorical/evasive tactics in conversation."""
    PREMATURE_PLURALISM = "PREMATURE_PLURALISM"
    BURDEN_SHIFTING = "BURDEN_SHIFTING"
    PALTERING = "PALTERING"
    RETROACTIVE_REFRAMING = "RETROACTIVE_REFRAMING"
    FALSE_EQUIVALENCE = "FALSE_EQUIVALENCE"
    MOTTE_AND_BAILEY = "MOTTE_AND_BAILEY"
    STRATEGIC_OMISSION = "STRATEGIC_OMISSION"
    NEUTRALITY_DISTORTION = "NEUTRALITY_DISTORTION"
    EVASION = "EVASION"
    DEFLECTION = "DEFLECTION"
    HEDGING = "HEDGING"
    SYCOPHANCY = "SYCOPHANCY"
    TONE_POLICING = "TONE_POLICING"
    APPEAL_TO_COMPLEXITY = "APPEAL_TO_COMPLEXITY"
    OTHER = "OTHER"


class TacticalMove(BaseModel):
    """A classified rhetorical or evasive move within a message."""
    move_type: TacticalMoveType
    evidence: str = Field(
        description="Quote or description supporting this classification",
        max_length=500,
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class Message(BaseModel):
    """A single message in a conversation transcript."""
    id: str = Field(description="Unique identifier (e.g., 'msg_001' or turn number)")
    speaker: SpeakerRole
    content: str = Field(description="Full text of the message")
    timestamp: Optional[datetime] = None
    entities_mentioned: list[str] = Field(
        default_factory=list,
        description="Entity names found in this message",
    )
    tactical_moves: list[TacticalMove] = Field(
        default_factory=list,
        description="Classified rhetorical moves in this message",
    )


class ConversationEpisode(BaseModel):
    """
    A conversation transcript ingested as a series of linked messages.

    Extends the Episode concept for multi-turn conversations. Messages
    are ordered and linked via REPLIES_TO relationships. Entities
    extracted from messages are connected to both the message and the
    global entity graph.
    """
    source_id: str = Field(description="Unique identifier for the conversation")
    source_type: str = Field(default="conversation")
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    source_timestamp: Optional[datetime] = None
    messages: list[Message] = Field(default_factory=list)
    entities: list[Entity] = Field(
        default_factory=list,
        description="All entities extracted across the conversation",
    )
    raw_text: str = Field(description="The original transcript text")

    @model_validator(mode="after")
    def validate_message_order(self) -> "ConversationEpisode":
        """Ensure messages have unique IDs."""
        ids = [m.id for m in self.messages]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate message IDs found")
        return self


# ─── Feedback / Constraint Nodes ─────────────────────────────────────

class FeedbackType(str, Enum):
    AVOID_TOPIC = "AVOID_TOPIC"
    PREFER_STYLE = "PREFER_STYLE"
    CORRECT_FACT = "CORRECT_FACT"
    MERGE_ENTITIES = "MERGE_ENTITIES"
    CONFIRM_RELATION = "CONFIRM_RELATION"
    REJECT_RELATION = "REJECT_RELATION"


class FeedbackNode(BaseModel):
    """
    Human feedback stored as a graph node, linked to the relevant
    entity or community. This is how the system remembers editorial
    preferences across iterations.
    """

    feedback_type: FeedbackType
    target_entity: Optional[str] = Field(
        default=None,
        description="Entity this feedback applies to (if entity-level)",
    )
    target_community: Optional[int] = Field(
        default=None,
        description="Community ID this feedback applies to (if community-level)",
    )
    instruction: str = Field(
        description="The human's directive (e.g., 'Don't mention budget details')",
        max_length=500,
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = Field(default=True)


# ─── Density Metrics ─────────────────────────────────────────────────

class DensityReport(BaseModel):
    """Output of the fact-to-token ratio measurement."""

    section_id: str
    total_tokens: int = Field(ge=0)
    unique_entities_referenced: int = Field(ge=0)
    unique_edges_referenced: int = Field(ge=0)
    subgraph_entities_available: int = Field(ge=0)
    subgraph_edges_available: int = Field(ge=0)

    @property
    def entity_density(self) -> float:
        """Fraction of available entities actually used."""
        if self.subgraph_entities_available == 0:
            return 0.0
        return self.unique_entities_referenced / self.subgraph_entities_available

    @property
    def edge_density(self) -> float:
        """Fraction of available edges actually used."""
        if self.subgraph_edges_available == 0:
            return 0.0
        return self.unique_edges_referenced / self.subgraph_edges_available

    @property
    def facts_per_100_tokens(self) -> float:
        """Primary de-slop metric: graph edges referenced per 100 tokens."""
        if self.total_tokens == 0:
            return 0.0
        return (self.unique_edges_referenced / self.total_tokens) * 100
