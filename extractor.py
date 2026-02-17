"""
Extraction pipeline: raw text → Ollama → validated Entity/Relation objects.

This is the critical first stage. If extraction is noisy, the entire
downstream pipeline (community detection, drafting, de-slop) is compromised.

Design choices:
  - Two-pass extraction (entities first, then relations) to reduce hallucination
  - Pydantic validation catches malformed JSON before it hits the graph
  - Confidence scores let us filter low-quality extractions
  - Few-shot examples in the prompt anchor the model's output format
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

from schema import (
    Entity,
    EntityLabel,
    Episode,
    Relation,
    RelationshipType,
    ConversationEpisode,
    Message,
    SpeakerRole,
    TacticalMove,
    TacticalMoveType,
)

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL as EXTRACTION_MODEL,
    OLLAMA_TEMPERATURE as TEMPERATURE,
    OLLAMA_TIMEOUT as REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


# ─── Prompt Templates ────────────────────────────────────────────────

ENTITY_EXTRACTION_PROMPT = """You are a precise information extraction system. 
Extract all entities from the following text. Output ONLY a JSON array of objects.

Each object must have exactly these fields:
- "name": string (the canonical name of the entity)
- "label": string (one of: PERSON, ORGANIZATION, PROJECT, MILESTONE, TECHNOLOGY, LOCATION, EVENT, CONCEPT)
- "aliases": array of strings (alternative names found in the text, empty array if none)

Rules:
- Focus on DEFINITIVE facts and STATE CHANGES, not opinions or speculation.
- Use the most complete version of a name as the canonical "name".
- Include aliases only if the text uses multiple forms of the same name.
- Do NOT extract generic concepts unless they are specific named items.

Example output:
[
  {{"name": "John Smith", "label": "PERSON", "aliases": ["J. Smith", "John"]}},
  {{"name": "Acme Corp", "label": "ORGANIZATION", "aliases": ["Acme"]}},
  {{"name": "Project Aurora", "label": "PROJECT", "aliases": []}}
]

TEXT TO EXTRACT FROM:
---
{text}
---

Output ONLY the JSON array, no explanation:"""


RELATION_EXTRACTION_PROMPT = """You are a precise relationship extraction system.
Given these entities and the source text, extract all relationships between them.

KNOWN ENTITIES:
{entities_json}

For each relationship, output a JSON object with:
- "source_entity": string (must match an entity name above)
- "target_entity": string (must match an entity name above)  
- "relationship_type": string (one of: {relationship_types})
- "context": string (1-2 sentence evidence from the text, max 300 chars)
- "valid_from": string or null (ISO 8601 date if mentioned, null otherwise)
- "valid_to": string or null (ISO 8601 date if the relationship ended, null otherwise)
- "confidence": number 0-1 (how certain you are this relationship exists)

Rules:
- Only extract relationships with CLEAR evidence in the text.
- Focus on state changes: someone JOINING, LEAVING, STARTING, COMPLETING.
- If a date or timeframe is mentioned, include it in valid_from/valid_to.
- Use RELATED_TO only as a last resort when no specific type fits.
- Confidence below 0.5 means you're guessing — omit it instead.

TEXT:
---
{text}
---

Output ONLY a JSON array of relationship objects:"""


# ─── Ollama Client ────────────────────────────────────────────────────

async def _call_ollama(
    prompt: str,
    model: str = EXTRACTION_MODEL,
    temperature: float = TEMPERATURE,
) -> str:
    """Send a prompt to Ollama and return the raw text response."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
                # Request JSON mode if the model supports it
                "format": "json",
            },
        )
        response.raise_for_status()
        return response.json()["response"]


def _parse_json_array(raw: str) -> list[dict]:
    """
    Parse JSON from Ollama's response, handling common LLM output quirks:
    - Markdown code fences
    - Trailing commas
    - Preamble text before the JSON
    - Single object instead of array
    - Preamble prose before the JSON
    """
    import re

    text = raw.strip()

    # Strip markdown fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()

    # Remove trailing commas before ] or }
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # 1. Try direct parse of the whole (cleaned) text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return []
    except json.JSONDecodeError:
        pass

    # 2. Try to extract a JSON array [...]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            pass

    # 3. Try to extract a single JSON object {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

    # 4. Last-ditch: scrape every {...} object out individually
    objs = re.findall(r"\{[^{}]*\}", text)
    results = []
    for obj_str in objs:
        try:
            results.append(json.loads(obj_str))
        except json.JSONDecodeError:
            continue
    if results:
        return results

    raise ValueError(f"No JSON array found in response: {text[:200]}")


# ─── Two-Pass Extraction ─────────────────────────────────────────────

async def extract_entities_raw(text: str) -> list[dict]:
    """Pass 1: Extract entity objects from text."""
    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
    raw = await _call_ollama(prompt)
    return _parse_json_array(raw)


async def extract_relations_raw(
    text: str, entities: list[dict]
) -> list[dict]:
    """Pass 2: Given known entities, extract relationships."""
    entities_json = json.dumps(
        [{"name": e["name"], "label": e["label"]} for e in entities],
        indent=2,
    )
    relationship_types = ", ".join(rt.value for rt in RelationshipType)

    prompt = RELATION_EXTRACTION_PROMPT.format(
        text=text,
        entities_json=entities_json,
        relationship_types=relationship_types,
    )
    raw = await _call_ollama(prompt)
    try:
        relations = _parse_json_array(raw)
    except ValueError as exc:
        logger.warning(f"Failed to parse relations JSON: {exc}")
        return []

    # Filter out entries that would break Neo4j (missing or empty required fields)
    valid = [
        r for r in relations
        if isinstance(r, dict)
        and r.get("source_entity")
        and r.get("target_entity")
        and r.get("relationship_type")
    ]
    skipped = len(relations) - len(valid)
    if skipped:
        logger.warning(f"  Skipped {skipped} invalid relation(s) (missing source/target/type)")
    return valid


async def extract_episode(
    text: str,
    source_id: str,
    source_type: str = "contact",
    source_timestamp: Optional[str] = None,
    min_confidence: float = 0.5,
) -> Episode:
    """
    Full extraction pipeline: text → entities → relations → validated Episode.
    
    This is the main entry point. It runs two Ollama passes and validates
    everything through Pydantic before returning.
    """
    # Pass 1: Entities
    logger.info(f"Extracting entities from source '{source_id}'...")
    raw_entities = await extract_entities_raw(text)
    logger.info(f"  Found {len(raw_entities)} candidate entities")

    # Validate entity labels, handling both dict and string formats from LLM
    valid_labels = {e.value for e in EntityLabel}
    cleaned_entities = []
    for e in raw_entities:
        if isinstance(e, str):
            # LLM returned a plain string instead of a dict
            logger.info(f"  Converting string entity to dict: {e}")
            e = {"name": e.strip(), "label": "CONCEPT", "aliases": []}
        if not isinstance(e, dict):
            logger.warning(f"  Skipping unexpected entity format: {type(e)} - {e}")
            continue
        label = e.get("label", "").upper()
        if label not in valid_labels:
            logger.warning(f"  Skipping entity with invalid label: {e}")
            continue
        cleaned_entities.append(e)

    # Pass 2: Relations
    logger.info(f"Extracting relations for {len(cleaned_entities)} entities...")
    raw_relations = await extract_relations_raw(text, cleaned_entities)
    logger.info(f"  Found {len(raw_relations)} candidate relations")

    # Build a lookup: entity name -> list of validated Relation objects
    entity_names = {e["name"].strip().title() for e in cleaned_entities}
    relations_by_source: dict[str, list[Relation]] = {}

    valid_rel_types = {rt.value for rt in RelationshipType}

    for r in raw_relations:
        try:
            rel_type = r.get("relationship_type", "").upper()
            if rel_type not in valid_rel_types:
                logger.warning(f"  Skipping relation with invalid type: {rel_type}")
                continue

            confidence = float(r.get("confidence", 0.8))
            if confidence < min_confidence:
                logger.debug(f"  Skipping low-confidence relation: {r}")
                continue

            source = r.get("source_entity", "").strip().title()
            target = r.get("target_entity", "").strip().title()

            if source not in entity_names or target not in entity_names:
                logger.warning(
                    f"  Skipping relation with unknown entity: {source} -> {target}"
                )
                continue

            relation = Relation(
                target_entity=target,
                relationship_type=RelationshipType(rel_type),
                context=r.get("context", "")[:300],
                valid_from=r.get("valid_from"),
                valid_to=r.get("valid_to"),
                confidence=confidence,
            )

            relations_by_source.setdefault(source, []).append(relation)

        except Exception as exc:
            logger.warning(f"  Failed to validate relation: {exc} — {r}")
            continue

    # Assemble validated Entity objects
    entities = []
    for e in cleaned_entities:
        name = e["name"].strip().title()
        entity = Entity(
            name=name,
            label=EntityLabel(e["label"].upper()),
            aliases=e.get("aliases", []),
            relations=relations_by_source.get(name, []),
        )
        entities.append(entity)

    # Build the Episode
    from datetime import datetime
    from dateutil.parser import parse as parse_dt

    ts = None
    if source_timestamp:
        try:
            ts = parse_dt(source_timestamp)
        except Exception:
            pass

    episode = Episode(
        source_id=source_id,
        source_type=source_type,
        source_timestamp=ts,
        entities=entities,
        raw_text=text,
    )

    logger.info(
        f"Episode '{source_id}': {len(entities)} entities, "
        f"{sum(len(e.relations) for e in entities)} relations"
    )
    return episode


# ─── Conversation Parsing ────────────────────────────────────────────

import re as _re

# Patterns for common chat export formats
_SPEAKER_PATTERNS = [
    # **User:** or **Assistant:** (Markdown bold)
    _re.compile(r"^\*\*(?P<speaker>User|Assistant|System|Human|AI|Claude|ChatGPT)\s*:\*\*\s*", _re.IGNORECASE),
    # User: or Assistant: (plain)
    _re.compile(r"^(?P<speaker>User|Assistant|System|Human|AI|Claude|ChatGPT)\s*:\s*", _re.IGNORECASE),
    # ### User or ### Assistant (heading format)
    _re.compile(r"^#{1,3}\s+(?P<speaker>User|Assistant|System|Human|AI|Claude|ChatGPT)\s*$", _re.IGNORECASE),
    # > **User:** (blockquote)
    _re.compile(r"^>\s*\*?\*?(?P<speaker>User|Assistant|System|Human|AI|Claude|ChatGPT)\s*:\*?\*?\s*", _re.IGNORECASE),
]

_SPEAKER_MAP = {
    "user": SpeakerRole.USER,
    "human": SpeakerRole.USER,
    "assistant": SpeakerRole.ASSISTANT,
    "ai": SpeakerRole.ASSISTANT,
    "claude": SpeakerRole.ASSISTANT,
    "chatgpt": SpeakerRole.ASSISTANT,
    "system": SpeakerRole.SYSTEM,
}


def parse_conversation(text: str) -> list[dict]:
    """
    Parse a chat transcript into a list of {speaker, content} dicts.

    Handles common export formats:
      - **User:** / **Assistant:** (Markdown bold)
      - User: / Assistant: (plain labels)
      - ### User / ### Assistant (heading format)
    """
    messages = []
    current_speaker = None
    current_lines = []

    for line in text.split("\n"):
        matched = False
        for pattern in _SPEAKER_PATTERNS:
            m = pattern.match(line)
            if m:
                # Save previous message
                if current_speaker is not None and current_lines:
                    messages.append({
                        "speaker": current_speaker,
                        "content": "\n".join(current_lines).strip(),
                    })
                # Start new message
                speaker_raw = m.group("speaker").lower()
                current_speaker = _SPEAKER_MAP.get(speaker_raw, SpeakerRole.USER)
                # Capture any text after the speaker label on the same line
                remainder = line[m.end():].strip()
                current_lines = [remainder] if remainder else []
                matched = True
                break

        if not matched and current_speaker is not None:
            current_lines.append(line)

    # Don't forget the last message
    if current_speaker is not None and current_lines:
        messages.append({
            "speaker": current_speaker,
            "content": "\n".join(current_lines).strip(),
        })

    return messages


# ─── Conversation Extraction Prompts ─────────────────────────────────

CONVERSATION_ENTITY_PROMPT = """You are a precise information extraction system analyzing a conversation transcript.
Extract all named entities mentioned across the conversation. Output ONLY a JSON array.

Each object must have:
- "name": string (canonical name)
- "label": string (one of: PERSON, ORGANIZATION, PROJECT, MILESTONE, TECHNOLOGY, LOCATION, EVENT, CONCEPT)
- "aliases": array of strings

Rules:
- Extract people, organizations, concepts, events, and locations mentioned in the discussion.
- Include abstract concepts if they are specific and named (e.g., "Democratic Backsliding", "Section 230").
- Do NOT extract the speakers themselves (User/Assistant) as entities.
- Focus on entities that are being DISCUSSED, not conversational filler.

CONVERSATION:
---
{text}
---

Output ONLY the JSON array:"""


CONVERSATION_TACTICS_PROMPT = """You are analyzing a conversation for rhetorical and evasive tactics.

For each message from the ASSISTANT, classify any tactical moves present.
A tactical move is a rhetorical strategy that shapes the conversation's direction.

TACTICAL MOVE TYPES:
- PREMATURE_PLURALISM: Presenting multiple perspectives before engaging with the one offered, to dilute a specific claim
- BURDEN_SHIFTING: Requiring the user to prove their point to an unreasonable standard while accepting counter-claims uncritically
- PALTERING: Using technically true statements to create a misleading impression
- RETROACTIVE_REFRAMING: Reinterpreting earlier statements to align with a new position without acknowledging the shift
- FALSE_EQUIVALENCE: Treating unequal things as equal to avoid taking a position
- MOTTE_AND_BAILEY: Retreating to an easily defensible position when a stronger claim is challenged
- STRATEGIC_OMISSION: Leaving out relevant information that would support the user's point
- NEUTRALITY_DISTORTION: Performing neutrality in a way that systematically favors one side
- EVASION: Avoiding a direct question or point entirely
- DEFLECTION: Redirecting the conversation away from the topic at hand
- HEDGING: Excessive qualification that drains a statement of meaning
- SYCOPHANCY: Agreement without substance, or praise used to redirect
- TONE_POLICING: Focusing on how something is said rather than what is said
- APPEAL_TO_COMPLEXITY: Using "it's complicated" as a substitute for engagement

CONVERSATION MESSAGES:
{messages_json}

For each message that contains a tactical move, output a JSON object with:
- "message_id": string (the message ID)
- "moves": array of objects, each with:
  - "move_type": string (one of the types above)
  - "evidence": string (quote or description supporting this classification, max 500 chars)
  - "confidence": number 0-1

Only flag moves you are confident about (>=0.6). Skip messages with no tactical moves.

Output ONLY a JSON array of objects:"""


# ─── Conversation Extraction Pipeline ────────────────────────────────

async def extract_conversation_entities(text: str) -> list[dict]:
    """Extract entities from a conversation transcript."""
    prompt = CONVERSATION_ENTITY_PROMPT.format(text=text)
    raw = await _call_ollama(prompt)
    return _parse_json_array(raw)


async def extract_conversation_tactics(
    messages: list[dict],
) -> dict[str, list[dict]]:
    """
    Analyze assistant messages for tactical moves.

    Returns a dict mapping message_id -> list of tactical move dicts.
    """
    # Only analyze assistant messages
    assistant_msgs = [
        {"message_id": m["id"], "speaker": m["speaker"], "content": m["content"][:1000]}
        for m in messages
        if m["speaker"] in ("assistant", SpeakerRole.ASSISTANT)
    ]

    if not assistant_msgs:
        return {}

    prompt = CONVERSATION_TACTICS_PROMPT.format(
        messages_json=json.dumps(assistant_msgs, indent=2, default=str)
    )
    raw = await _call_ollama(prompt)

    try:
        results = _parse_json_array(raw)
    except ValueError:
        logger.warning("Failed to parse tactics response, returning empty")
        return {}

    tactics_by_msg: dict[str, list[dict]] = {}
    valid_types = {t.value for t in TacticalMoveType}

    for item in results:
        msg_id = item.get("message_id", "")
        moves = item.get("moves", [])
        validated_moves = []
        for move in moves:
            mtype = move.get("move_type", "").upper()
            if mtype not in valid_types:
                continue
            confidence = float(move.get("confidence", 0.5))
            if confidence < 0.6:
                continue
            validated_moves.append({
                "move_type": mtype,
                "evidence": move.get("evidence", "")[:500],
                "confidence": confidence,
            })
        if validated_moves:
            tactics_by_msg[msg_id] = validated_moves

    return tactics_by_msg


async def extract_conversation(
    text: str,
    source_id: str,
    source_timestamp: Optional[str] = None,
    min_confidence: float = 0.5,
    extract_tactics: bool = True,
) -> ConversationEpisode:
    """
    Full conversation extraction pipeline:
    text → parse messages → extract entities → extract tactics → validated ConversationEpisode.
    """
    # Step 1: Parse the transcript into messages
    logger.info(f"Parsing conversation from source '{source_id}'...")
    raw_messages = parse_conversation(text)
    logger.info(f"  Parsed {len(raw_messages)} messages")

    if not raw_messages:
        raise ValueError(
            "No messages found in transcript. Expected format like "
            "'**User:** ...' or 'User: ...' or '### User'"
        )

    # Assign message IDs
    for i, msg in enumerate(raw_messages):
        msg["id"] = f"msg_{i:03d}"

    # Step 2: Extract entities from the full conversation
    logger.info("Extracting entities from conversation...")
    raw_entities = await extract_conversation_entities(text)
    logger.info(f"  Found {len(raw_entities)} candidate entities")

    # Validate entity labels, handling both dict and string formats from LLM
    valid_labels = {e.value for e in EntityLabel}
    cleaned_entities = []
    for e in raw_entities:
        if isinstance(e, str):
            # LLM returned a plain string instead of a dict
            logger.info(f"  Converting string entity to dict: {e}")
            e = {"name": e.strip(), "label": "CONCEPT", "aliases": []}
        if not isinstance(e, dict):
            logger.warning(f"  Skipping unexpected entity format: {type(e)} - {e}")
            continue
        label = e.get("label", "").upper()
        if label not in valid_labels:
            logger.warning(f"  Skipping entity with invalid label: {e}")
            continue
        cleaned_entities.append(e)

    # Step 3: Extract relations between entities (reuse existing)
    logger.info(f"Extracting relations for {len(cleaned_entities)} entities...")
    raw_relations = await extract_relations_raw(text, cleaned_entities)
    logger.info(f"  Found {len(raw_relations)} candidate relations")

    # Build entity objects with relations (same logic as extract_episode)
    entity_names = {e["name"].strip().title() for e in cleaned_entities}
    relations_by_source: dict[str, list[Relation]] = {}
    valid_rel_types = {rt.value for rt in RelationshipType}

    for r in raw_relations:
        try:
            rel_type = r.get("relationship_type", "").upper()
            if rel_type not in valid_rel_types:
                continue
            confidence = float(r.get("confidence", 0.8))
            if confidence < min_confidence:
                continue
            source = r.get("source_entity", "").strip().title()
            target = r.get("target_entity", "").strip().title()
            if source not in entity_names or target not in entity_names:
                continue
            relation = Relation(
                target_entity=target,
                relationship_type=RelationshipType(rel_type),
                context=r.get("context", "")[:300],
                valid_from=r.get("valid_from"),
                valid_to=r.get("valid_to"),
                confidence=confidence,
            )
            relations_by_source.setdefault(source, []).append(relation)
        except Exception as exc:
            logger.warning(f"  Failed to validate relation: {exc}")
            continue

    entities = []
    for e in cleaned_entities:
        name = e["name"].strip().title()
        entity = Entity(
            name=name,
            label=EntityLabel(e["label"].upper()),
            aliases=e.get("aliases", []),
            relations=relations_by_source.get(name, []),
        )
        entities.append(entity)

    # Step 4: Match entities to messages (simple substring matching)
    entity_name_lower = {e.name.lower(): e.name for e in entities}
    for msg in raw_messages:
        content_lower = msg["content"].lower()
        msg["entities_mentioned"] = [
            canonical
            for lower, canonical in entity_name_lower.items()
            if lower in content_lower
        ]

    # Step 5: Extract tactical moves (optional, LLM-based)
    tactics_by_msg: dict[str, list[dict]] = {}
    if extract_tactics:
        logger.info("Analyzing conversation for tactical moves...")
        tactics_by_msg = await extract_conversation_tactics(raw_messages)
        total_tactics = sum(len(v) for v in tactics_by_msg.values())
        logger.info(f"  Found {total_tactics} tactical moves across {len(tactics_by_msg)} messages")

    # Step 6: Build Message objects
    messages = []
    for msg in raw_messages:
        tactical_moves = []
        for t in tactics_by_msg.get(msg["id"], []):
            tactical_moves.append(TacticalMove(
                move_type=TacticalMoveType(t["move_type"]),
                evidence=t["evidence"],
                confidence=t["confidence"],
            ))

        messages.append(Message(
            id=msg["id"],
            speaker=SpeakerRole(msg["speaker"]) if isinstance(msg["speaker"], str) else msg["speaker"],
            content=msg["content"],
            entities_mentioned=msg.get("entities_mentioned", []),
            tactical_moves=tactical_moves,
        ))

    # Build the ConversationEpisode
    from dateutil.parser import parse as parse_dt
    ts = None
    if source_timestamp:
        try:
            ts = parse_dt(source_timestamp)
        except Exception:
            pass

    episode = ConversationEpisode(
        source_id=source_id,
        source_timestamp=ts,
        messages=messages,
        entities=entities,
        raw_text=text,
    )

    logger.info(
        f"Conversation '{source_id}': {len(messages)} messages, "
        f"{len(entities)} entities, "
        f"{sum(len(m.tactical_moves) for m in messages)} tactical moves"
    )
    return episode
