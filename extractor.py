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

from schema import Entity, EntityLabel, Episode, Relation, RelationshipType

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
EXTRACTION_MODEL = "llama3.1:70b"  # Use 70b for extraction quality; 8b for speed
TEMPERATURE = 0.1  # Low temperature for deterministic extraction
REQUEST_TIMEOUT = 300.0  # 70b can be slow on consumer hardware


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
    """
    text = raw.strip()

    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                text = part
                break

    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in response: {text[:200]}")

    json_str = text[start : end + 1]

    # Remove trailing commas before ] or }
    import re
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    return json.loads(json_str)


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
    return _parse_json_array(raw)


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

    # Validate entity labels
    valid_labels = {e.value for e in EntityLabel}
    cleaned_entities = []
    for e in raw_entities:
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
