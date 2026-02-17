"""
Graph store: loads validated Episodes into Neo4j with temporal properties.

Key design decisions:
  - Every entity becomes a node with label matching its EntityLabel
  - Every relation becomes a typed relationship with valid_from/valid_to
  - Episodes are tracked as nodes so we can query "what was ingested when"
  - Deduplication uses name + aliases to merge nodes on insert
  - Feedback nodes are first-class citizens in the graph
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from schema import (
    ConversationEpisode,
    DensityReport,
    Entity,
    Episode,
    FeedbackNode,
    Relation,
)

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger(__name__)


# ─── Graph Store ──────────────────────────────────────────────────────

class GraphStore:
    """Async Neo4j interface for the Graph Writing Studio."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            uri, auth=(user, password)
        )

    async def close(self):
        await self._driver.close()

    @asynccontextmanager
    async def _session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self._driver.session() as session:
            yield session

    # ─── Schema Setup ─────────────────────────────────────────────

    async def setup_indexes(self):
        """Create indexes and constraints for performance."""
        queries = [
            # Uniqueness on entity names within each label
            "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX episode_id IF NOT EXISTS FOR (n:Episode) ON (n.source_id)",
            "CREATE INDEX feedback_id IF NOT EXISTS FOR (n:Feedback) ON (n.created_at)",
            # Full-text index for entity search
            "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS "
            "FOR (n:Entity) ON EACH [n.name, n.aliases_text]",
        ]
        async with self._session() as session:
            for q in queries:
                try:
                    await session.run(q)
                except Exception as e:
                    logger.debug(f"Index creation note: {e}")

    # ─── Episode Ingestion ────────────────────────────────────────

    async def ingest_episode(self, episode: Episode) -> dict:
        """
        Load a validated Episode into Neo4j.
        
        Returns a summary dict with counts of created/merged nodes and rels.
        """
        stats = {"nodes_created": 0, "nodes_merged": 0, "rels_created": 0}

        async with self._session() as session:
            # 1. Create the Episode node
            await session.run(
                """
                MERGE (ep:Episode {source_id: $source_id})
                SET ep.source_type = $source_type,
                    ep.ingested_at = datetime($ingested_at),
                    ep.source_timestamp = CASE 
                        WHEN $source_ts IS NOT NULL THEN datetime($source_ts)
                        ELSE null 
                    END,
                    ep.raw_text_preview = left($raw_text, 500)
                """,
                source_id=episode.source_id,
                source_type=episode.source_type,
                ingested_at=episode.ingested_at.isoformat(),
                source_ts=(
                    episode.source_timestamp.isoformat()
                    if episode.source_timestamp
                    else None
                ),
                raw_text=episode.raw_text,
            )

            # 2. Upsert each entity
            for entity in episode.entities:
                result = await session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET 
                        e.label = $label,
                        e.aliases = $aliases,
                        e.aliases_text = $aliases_text,
                        e.created_at = datetime(),
                        e.updated_at = datetime()
                    ON MATCH SET
                        e.updated_at = datetime(),
                        e.aliases = CASE 
                            WHEN size($aliases) > size(coalesce(e.aliases, []))
                            THEN $aliases ELSE e.aliases 
                        END,
                        e.aliases_text = CASE
                            WHEN size($aliases) > size(coalesce(e.aliases, []))
                            THEN $aliases_text ELSE e.aliases_text
                        END
                    RETURN e.created_at = e.updated_at AS is_new
                    """,
                    name=entity.name,
                    label=entity.label.value,
                    aliases=entity.aliases,
                    aliases_text=" | ".join(entity.aliases),
                )
                record = await result.single()
                if record and record["is_new"]:
                    stats["nodes_created"] += 1
                else:
                    stats["nodes_merged"] += 1

                # Link entity to episode
                await session.run(
                    """
                    MATCH (e:Entity {name: $name})
                    MATCH (ep:Episode {source_id: $source_id})
                    MERGE (e)-[:EXTRACTED_FROM]->(ep)
                    """,
                    name=entity.name,
                    source_id=episode.source_id,
                )

                # 3. Create relationships
                for rel in entity.relations:
                    await session.run(
                        f"""
                        MATCH (src:Entity {{name: $src_name}})
                        MATCH (tgt:Entity {{name: $tgt_name}})
                        CREATE (src)-[r:{rel.relationship_type.value} {{
                            context: $context,
                            valid_from: CASE 
                                WHEN $valid_from IS NOT NULL 
                                THEN datetime($valid_from) ELSE null 
                            END,
                            valid_to: CASE 
                                WHEN $valid_to IS NOT NULL 
                                THEN datetime($valid_to) ELSE null 
                            END,
                            confidence: $confidence,
                            episode_id: $episode_id
                        }}]->(tgt)
                        """,
                        src_name=entity.name,
                        tgt_name=rel.target_entity,
                        context=rel.context,
                        valid_from=(
                            rel.valid_from.isoformat() if rel.valid_from else None
                        ),
                        valid_to=(
                            rel.valid_to.isoformat() if rel.valid_to else None
                        ),
                        confidence=rel.confidence,
                        episode_id=episode.source_id,
                    )
                    stats["rels_created"] += 1

        logger.info(f"Ingested episode '{episode.source_id}': {stats}")
        return stats

    # ─── Conversation Ingestion ─────────────────────────────────────

    async def ingest_conversation(self, episode: ConversationEpisode) -> dict:
        """
        Load a ConversationEpisode into Neo4j.

        Creates Message nodes linked by REPLIES_TO, connects entities
        mentioned in messages, and stores tactical move tags.
        """
        stats = {
            "messages_created": 0,
            "nodes_created": 0,
            "nodes_merged": 0,
            "rels_created": 0,
            "tactics_stored": 0,
        }

        async with self._session() as session:
            # 1. Create the Episode node
            await session.run(
                """
                MERGE (ep:Episode {source_id: $source_id})
                SET ep.source_type = 'conversation',
                    ep.ingested_at = datetime($ingested_at),
                    ep.source_timestamp = CASE
                        WHEN $source_ts IS NOT NULL THEN datetime($source_ts)
                        ELSE null
                    END,
                    ep.message_count = $msg_count,
                    ep.raw_text_preview = left($raw_text, 500)
                """,
                source_id=episode.source_id,
                ingested_at=episode.ingested_at.isoformat(),
                source_ts=(
                    episode.source_timestamp.isoformat()
                    if episode.source_timestamp
                    else None
                ),
                msg_count=len(episode.messages),
                raw_text=episode.raw_text,
            )

            # 2. Create Message nodes and REPLIES_TO chain
            prev_msg_id = None
            for msg in episode.messages:
                await session.run(
                    """
                    CREATE (m:Message {
                        id: $msg_id,
                        speaker: $speaker,
                        content: $content,
                        conversation_id: $conv_id,
                        turn_number: $turn
                    })
                    WITH m
                    MATCH (ep:Episode {source_id: $conv_id})
                    CREATE (m)-[:PART_OF]->(ep)
                    """,
                    msg_id=msg.id,
                    speaker=msg.speaker.value,
                    content=msg.content,
                    conv_id=episode.source_id,
                    turn=int(msg.id.split("_")[-1]),
                )
                stats["messages_created"] += 1

                # Link to previous message
                if prev_msg_id is not None:
                    await session.run(
                        """
                        MATCH (curr:Message {id: $curr_id, conversation_id: $conv_id})
                        MATCH (prev:Message {id: $prev_id, conversation_id: $conv_id})
                        CREATE (curr)-[:REPLIES_TO]->(prev)
                        """,
                        curr_id=msg.id,
                        prev_id=prev_msg_id,
                        conv_id=episode.source_id,
                    )
                    stats["rels_created"] += 1

                prev_msg_id = msg.id

                # 3. Store tactical moves as TacticalMove nodes
                for tactic in msg.tactical_moves:
                    await session.run(
                        """
                        MATCH (m:Message {id: $msg_id, conversation_id: $conv_id})
                        CREATE (t:TacticalMove {
                            move_type: $move_type,
                            evidence: $evidence,
                            confidence: $confidence
                        })
                        CREATE (m)-[:EXHIBITS_TACTIC]->(t)
                        """,
                        msg_id=msg.id,
                        conv_id=episode.source_id,
                        move_type=tactic.move_type.value,
                        evidence=tactic.evidence,
                        confidence=tactic.confidence,
                    )
                    stats["tactics_stored"] += 1

            # 4. Upsert entities and link to messages
            for entity in episode.entities:
                result = await session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET
                        e.label = $label,
                        e.aliases = $aliases,
                        e.aliases_text = $aliases_text,
                        e.created_at = datetime(),
                        e.updated_at = datetime()
                    ON MATCH SET
                        e.updated_at = datetime(),
                        e.aliases = CASE
                            WHEN size($aliases) > size(coalesce(e.aliases, []))
                            THEN $aliases ELSE e.aliases
                        END,
                        e.aliases_text = CASE
                            WHEN size($aliases) > size(coalesce(e.aliases, []))
                            THEN $aliases_text ELSE e.aliases_text
                        END
                    RETURN e.created_at = e.updated_at AS is_new
                    """,
                    name=entity.name,
                    label=entity.label.value,
                    aliases=entity.aliases,
                    aliases_text=" | ".join(entity.aliases),
                )
                record = await result.single()
                if record and record["is_new"]:
                    stats["nodes_created"] += 1
                else:
                    stats["nodes_merged"] += 1

                # Link entity to episode
                await session.run(
                    """
                    MATCH (e:Entity {name: $name})
                    MATCH (ep:Episode {source_id: $source_id})
                    MERGE (e)-[:EXTRACTED_FROM]->(ep)
                    """,
                    name=entity.name,
                    source_id=episode.source_id,
                )

                # Create entity relationships
                for rel in entity.relations:
                    await session.run(
                        f"""
                        MATCH (src:Entity {{name: $src_name}})
                        MATCH (tgt:Entity {{name: $tgt_name}})
                        CREATE (src)-[r:{rel.relationship_type.value} {{
                            context: $context,
                            valid_from: CASE
                                WHEN $valid_from IS NOT NULL
                                THEN datetime($valid_from) ELSE null
                            END,
                            valid_to: CASE
                                WHEN $valid_to IS NOT NULL
                                THEN datetime($valid_to) ELSE null
                            END,
                            confidence: $confidence,
                            episode_id: $episode_id
                        }}]->(tgt)
                        """,
                        src_name=entity.name,
                        tgt_name=rel.target_entity,
                        context=rel.context,
                        valid_from=(
                            rel.valid_from.isoformat() if rel.valid_from else None
                        ),
                        valid_to=(
                            rel.valid_to.isoformat() if rel.valid_to else None
                        ),
                        confidence=rel.confidence,
                        episode_id=episode.source_id,
                    )
                    stats["rels_created"] += 1

            # 5. Link entities to messages where they're mentioned
            for msg in episode.messages:
                for entity_name in msg.entities_mentioned:
                    await session.run(
                        """
                        MATCH (m:Message {id: $msg_id, conversation_id: $conv_id})
                        MATCH (e:Entity {name: $entity_name})
                        MERGE (m)-[:MENTIONS]->(e)
                        """,
                        msg_id=msg.id,
                        conv_id=episode.source_id,
                        entity_name=entity_name,
                    )
                    stats["rels_created"] += 1

        logger.info(f"Ingested conversation '{episode.source_id}': {stats}")
        return stats

    # ─── Conversation Queries ──────────────────────────────────────

    async def get_conversation_thread(
        self, conversation_id: str
    ) -> list[dict]:
        """Return all messages in a conversation, ordered by turn number."""
        async with self._session() as session:
            result = await session.run(
                """
                MATCH (m:Message {conversation_id: $conv_id})
                OPTIONAL MATCH (m)-[:EXHIBITS_TACTIC]->(t:TacticalMove)
                WITH m, collect(t {.move_type, .evidence, .confidence}) AS tactics
                RETURN m.id AS id, m.speaker AS speaker, m.content AS content,
                       m.turn_number AS turn, tactics
                ORDER BY m.turn_number
                """,
                conv_id=conversation_id,
            )
            messages = []
            async for record in result:
                messages.append({
                    "id": record["id"],
                    "speaker": record["speaker"],
                    "content": record["content"],
                    "turn": record["turn"],
                    "tactics": [dict(t) for t in record["tactics"] if t.get("move_type")],
                })
            return messages

    async def get_message_context(
        self, message_id: str, conversation_id: str, hops: int = 2
    ) -> dict:
        """
        Return the subgraph around a message: nearby turns, linked entities,
        tactical moves, and feedback.
        """
        async with self._session() as session:
            # Get the message and surrounding messages
            result = await session.run(
                """
                MATCH (center:Message {id: $msg_id, conversation_id: $conv_id})
                OPTIONAL MATCH (center)-[:EXHIBITS_TACTIC]->(t:TacticalMove)
                WITH center, collect(t {.move_type, .evidence, .confidence}) AS center_tactics

                // Get surrounding messages via REPLIES_TO chain
                OPTIONAL MATCH path = (center)-[:REPLIES_TO*1..{hops}]-(neighbor:Message)
                WITH center, center_tactics, collect(DISTINCT neighbor) AS neighbors

                // Get mentioned entities
                OPTIONAL MATCH (center)-[:MENTIONS]->(e:Entity)
                WITH center, center_tactics, neighbors, collect(DISTINCT e) AS entities

                RETURN center {.id, .speaker, .content, .turn_number} AS message,
                       center_tactics AS tactics,
                       [n IN neighbors | n {.id, .speaker, .content, .turn_number}] AS nearby_messages,
                       [e IN entities | e {.name, .label}] AS mentioned_entities
                """.replace("{hops}", str(hops)),
                msg_id=message_id,
                conv_id=conversation_id,
            )
            record = await result.single()
            if not record:
                return {"message": None, "tactics": [], "nearby_messages": [], "mentioned_entities": []}

            return {
                "message": dict(record["message"]),
                "tactics": [dict(t) for t in record["tactics"] if t.get("move_type")],
                "nearby_messages": sorted(
                    [dict(m) for m in record["nearby_messages"]],
                    key=lambda m: m.get("turn_number", 0),
                ),
                "mentioned_entities": [dict(e) for e in record["mentioned_entities"]],
            }

    async def tag_message_feedback(
        self,
        message_id: str,
        conversation_id: str,
        feedback_type: str,
        instruction: str,
    ) -> None:
        """
        Tag a message with feedback (e.g., tactical classification).

        This uses the existing Feedback node system but links to a Message node.
        """
        async with self._session() as session:
            await session.run(
                """
                MATCH (m:Message {id: $msg_id, conversation_id: $conv_id})
                CREATE (f:Feedback {
                    feedback_type: $ftype,
                    instruction: $instruction,
                    created_at: datetime(),
                    active: true
                })
                CREATE (f)-[:APPLIES_TO]->(m)
                """,
                msg_id=message_id,
                conv_id=conversation_id,
                ftype=feedback_type,
                instruction=instruction,
            )

    async def get_conversation_section_data(
        self, conversation_id: str, community_members: list[str]
    ) -> dict:
        """
        Retrieve all messages, tactics, and entities relevant to a community
        for conversation-based drafting.

        Finds messages that mention any entity in the community, plus their
        tactical tags and feedback.
        """
        async with self._session() as session:
            result = await session.run(
                """
                // Find messages that mention community entities
                MATCH (m:Message {conversation_id: $conv_id})-[:MENTIONS]->(e:Entity)
                WHERE e.name IN $members
                WITH DISTINCT m, collect(DISTINCT e.name) AS mentioned_entities
                OPTIONAL MATCH (m)-[:EXHIBITS_TACTIC]->(t:TacticalMove)
                WITH m, mentioned_entities,
                     collect(t {.move_type, .evidence, .confidence}) AS tactics
                OPTIONAL MATCH (f:Feedback {active: true})-[:APPLIES_TO]->(m)
                WITH m, mentioned_entities, tactics,
                     collect(f {.feedback_type, .instruction}) AS feedback
                RETURN m.id AS id, m.speaker AS speaker, m.content AS content,
                       m.turn_number AS turn, mentioned_entities, tactics, feedback
                ORDER BY m.turn_number
                """,
                conv_id=conversation_id,
                members=community_members,
            )

            messages = []
            async for record in result:
                messages.append({
                    "id": record["id"],
                    "speaker": record["speaker"],
                    "content": record["content"],
                    "turn": record["turn"],
                    "mentioned_entities": record["mentioned_entities"],
                    "tactics": [dict(t) for t in record["tactics"] if t.get("move_type")],
                    "feedback": [dict(f) for f in record["feedback"] if f.get("feedback_type")],
                })

            # Also get entity relationships for community members
            entity_relations = []
            for member in community_members:
                neighborhood = await self.get_neighborhood(
                    entity_name=member, hops=1, min_confidence=0.5
                )
                entity_relations.extend(neighborhood.get("relations", []))

            # Deduplicate relations
            seen = set()
            unique_relations = []
            for r in entity_relations:
                key = (r["source"], r["target"], r["type"])
                if key not in seen:
                    seen.add(key)
                    unique_relations.append(r)

            return {
                "messages": messages,
                "entity_relations": unique_relations,
                "community_members": community_members,
            }

    # ─── Neighborhood Queries ─────────────────────────────────────

    async def get_neighborhood(
        self,
        entity_name: str,
        hops: int = 2,
        min_confidence: float = 0.5,
        active_only: bool = True,
    ) -> dict:
        """
        Get the N-hop neighborhood around an entity.
        
        Returns:
            {
                "entities": [{"name": ..., "label": ...}, ...],
                "relations": [{"source": ..., "target": ..., "type": ..., ...}, ...]
            }
        """
        time_filter = ""
        if active_only:
            time_filter = "AND (rel.valid_to IS NULL)"

        query = f"""
        MATCH path = (start:Entity {{name: $name}})-[*1..{hops}]-(neighbor:Entity)
        WHERE ALL(rel IN relationships(path)
                  WHERE rel.confidence >= $min_conf {time_filter})
        WITH DISTINCT neighbor, relationships(path) AS rels
        UNWIND rels AS rel
        WITH COLLECT(DISTINCT neighbor) AS neighbors,
             COLLECT(DISTINCT rel) AS all_rels
        RETURN neighbors, all_rels
        """

        async with self._session() as session:
            result = await session.run(
                query,
                name=entity_name,
                min_conf=min_confidence,
            )
            record = await result.single()

            if not record:
                return {"entities": [], "relations": []}

            entities = []
            for node in record["neighbors"]:
                entities.append({
                    "name": node["name"],
                    "label": node.get("label", "UNKNOWN"),
                })

            relations = []
            seen_rels = set()
            for rel in record["all_rels"]:
                key = (
                    rel.start_node["name"],
                    rel.end_node["name"],
                    rel.type,
                )
                if key not in seen_rels:
                    seen_rels.add(key)
                    relations.append({
                        "source": rel.start_node["name"],
                        "target": rel.end_node["name"],
                        "type": rel.type,
                        "context": rel.get("context", ""),
                        "confidence": rel.get("confidence", 0.0),
                    })

            return {"entities": entities, "relations": relations}

    # ─── Community Detection ──────────────────────────────────────

    async def run_community_detection(
        self, algorithm: str = "louvain"
    ) -> list[dict]:
        """
        Run community detection on the entity graph using Neo4j GDS.
        
        Returns a list of communities with their member entities.
        Requires the Graph Data Science plugin.
        """
        async with self._session() as session:
            # Project the graph into GDS using Cypher aggregation
            try:
                await session.run(
                    "CALL gds.graph.drop('entity_graph', false)"
                )
            except Exception:
                pass

            await session.run(
                """
                MATCH (source:Entity)-[r]->(target:Entity)
                WHERE r.confidence IS NOT NULL
                WITH gds.graph.project(
                    'entity_graph',
                    source,
                    target,
                    {relationshipProperties: {confidence: r.confidence}},
                    {undirectedRelationshipTypes: ['*']}
                ) AS g
                RETURN g.graphName
                """
            )

            # Run community detection
            if algorithm == "leiden":
                algo_call = """
                CALL gds.leiden.write('entity_graph', {
                    writeProperty: 'community',
                    relationshipWeightProperty: 'confidence'
                })
                """
            else:
                algo_call = """
                CALL gds.louvain.write('entity_graph', {
                    writeProperty: 'community',
                    relationshipWeightProperty: 'confidence'
                })
                """

            await session.run(algo_call)

            # Fetch communities
            result = await session.run(
                """
                MATCH (e:Entity)
                WHERE e.community IS NOT NULL
                RETURN e.community AS community_id,
                       collect(e.name) AS members,
                       count(e) AS size
                ORDER BY size DESC
                """
            )

            communities = []
            async for record in result:
                communities.append({
                    "community_id": record["community_id"],
                    "members": record["members"],
                    "size": record["size"],
                })

            # Cleanup
            await session.run(
                "CALL gds.graph.drop('entity_graph', false)"
            )

            return communities

    # ─── Conversation Community Detection ─────────────────────────

    async def detect_conversation_communities(
        self, conversation_id: str, algorithm: str = "leiden"
    ) -> dict[int, list[str]]:
        """
        Run community detection on a conversation subgraph.

        Projects Message nodes (filtered by conversation_id) and their
        REPLIES_TO relationships, plus optional MENTIONS links to Entity
        nodes and EXHIBITS_TACTIC links to TacticalMove nodes, into a
        GDS in-memory graph. Then runs the specified community detection
        algorithm (Leiden by default) to cluster messages into sections.

        Returns:
            A mapping of community_id → list of message IDs.
        """
        graph_name = f"conversation_{conversation_id}"

        async with self._session() as session:
            # Early exit if no messages exist for this conversation
            count_result = await session.run(
                "MATCH (m:Message {conversation_id: $conv_id}) RETURN count(m) AS cnt",
                conv_id=conversation_id,
            )
            count_record = await count_result.single()
            if not count_record or count_record["cnt"] == 0:
                return {}

            # Drop any existing projection with the same name
            try:
                await session.run(
                    "CALL gds.graph.drop($name, false)",
                    name=graph_name,
                )
            except Exception:
                pass

            # Project the conversation subgraph using Cypher aggregation.
            # We include Message nodes connected by REPLIES_TO, plus
            # connections through shared Entity mentions (Message-MENTIONS->Entity<-MENTIONS-Message)
            # to capture topical similarity.
            #
            # Step 1: Create temporary TOPIC_SIMILAR relationships for shared entity mentions
            await session.run(
                """
                MATCH (m1:Message {conversation_id: $conv_id})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(m2:Message {conversation_id: $conv_id})
                WHERE elementId(m1) < elementId(m2)
                MERGE (m1)-[:TOPIC_SIMILAR]->(m2)
                """,
                conv_id=conversation_id,
            )

            # Step 2: Project using Cypher aggregation (GDS 2.x+ compatible)
            # Leiden/Louvain require undirected graphs.
            # The 4th parameter (dataConfig) is required by GDS — pass at minimum {}.
            await session.run(
                """
                MATCH (source:Message {conversation_id: $conv_id})-[r:REPLIES_TO|TOPIC_SIMILAR]-(target:Message {conversation_id: $conv_id})
                WITH gds.graph.project(
                    $graph_name,
                    source,
                    target,
                    {},
                    {undirectedRelationshipTypes: ['*']}
                ) AS g
                RETURN g.graphName
                """,
                graph_name=graph_name,
                conv_id=conversation_id,
            )

            # Run community detection
            if algorithm == "leiden":
                algo_call = """
                CALL gds.leiden.write($graph_name, {
                    writeProperty: 'conv_community'
                })
                """
            else:
                algo_call = """
                CALL gds.louvain.write($graph_name, {
                    writeProperty: 'conv_community'
                })
                """

            await session.run(algo_call, graph_name=graph_name)

            # Fetch the community assignments for messages in this conversation
            result = await session.run(
                """
                MATCH (m:Message {conversation_id: $conv_id})
                WHERE m.conv_community IS NOT NULL
                RETURN m.conv_community AS community_id,
                       collect(m.id) AS message_ids,
                       count(m) AS size
                ORDER BY size DESC
                """,
                conv_id=conversation_id,
            )

            communities: dict[int, list[str]] = {}
            async for record in result:
                comm_id = record["community_id"]
                communities[comm_id] = sorted(
                    record["message_ids"],
                    key=lambda mid: int(mid.split("_")[-1]),
                )

            # Cleanup: drop GDS graph and temporary relationships
            try:
                await session.run(
                    "CALL gds.graph.drop($name, false)",
                    name=graph_name,
                )
            except Exception:
                pass

            # Remove temporary TOPIC_SIMILAR relationships
            await session.run(
                """
                MATCH (m1:Message {conversation_id: $conv_id})-[r:TOPIC_SIMILAR]-(m2:Message)
                DELETE r
                """,
                conv_id=conversation_id,
            )

            return communities

    async def get_conversation_community_section_data(
        self, conversation_id: str, message_ids: list[str]
    ) -> dict:
        """
        Retrieve messages, tactics, entities, and feedback for a set of
        message IDs within a conversation. Used by draft-conversation
        when working from a conversation outline.

        Returns:
            {
                "messages": [...],
                "entity_relations": [...],
                "message_ids": [...]
            }
        """
        async with self._session() as session:
            result = await session.run(
                """
                MATCH (m:Message {conversation_id: $conv_id})
                WHERE m.id IN $msg_ids
                OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
                WITH m, collect(DISTINCT e.name) AS mentioned_entities
                OPTIONAL MATCH (m)-[:EXHIBITS_TACTIC]->(t:TacticalMove)
                WITH m, mentioned_entities,
                     collect(t {.move_type, .evidence, .confidence}) AS tactics
                OPTIONAL MATCH (f:Feedback {active: true})-[:APPLIES_TO]->(m)
                WITH m, mentioned_entities, tactics,
                     collect(f {.feedback_type, .instruction}) AS feedback
                RETURN m.id AS id, m.speaker AS speaker, m.content AS content,
                       m.turn_number AS turn, mentioned_entities, tactics, feedback
                ORDER BY m.turn_number
                """,
                conv_id=conversation_id,
                msg_ids=message_ids,
            )

            messages = []
            all_entity_names = set()
            async for record in result:
                mentioned = record["mentioned_entities"]
                all_entity_names.update(mentioned)
                messages.append({
                    "id": record["id"],
                    "speaker": record["speaker"],
                    "content": record["content"],
                    "turn": record["turn"],
                    "mentioned_entities": mentioned,
                    "tactics": [
                        dict(t) for t in record["tactics"]
                        if t.get("move_type")
                    ],
                    "feedback": [
                        dict(f) for f in record["feedback"]
                        if f.get("feedback_type")
                    ],
                })

            # Get entity relationships for mentioned entities
            entity_relations = []
            for name in all_entity_names:
                neighborhood = await self.get_neighborhood(
                    entity_name=name, hops=1, min_confidence=0.5
                )
                entity_relations.extend(neighborhood.get("relations", []))

            # Deduplicate relations
            seen = set()
            unique_relations = []
            for r in entity_relations:
                key = (r["source"], r["target"], r["type"])
                if key not in seen:
                    seen.add(key)
                    unique_relations.append(r)

            return {
                "messages": messages,
                "entity_relations": unique_relations,
                "message_ids": message_ids,
            }

    # ─── Feedback Storage ─────────────────────────────────────────

    async def store_feedback(self, feedback: FeedbackNode) -> None:
        """Store a human feedback node and link it to the relevant entity/community."""
        async with self._session() as session:
            await session.run(
                """
                CREATE (f:Feedback {
                    feedback_type: $ftype,
                    instruction: $instruction,
                    created_at: datetime($created_at),
                    active: $active
                })
                WITH f
                // Link to entity if specified
                FOREACH (_ IN CASE WHEN $target_entity IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (e:Entity {name: $target_entity})
                    CREATE (f)-[:APPLIES_TO]->(e)
                )
                // Link to community if specified
                FOREACH (_ IN CASE WHEN $target_community IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (c:Community {community_id: $target_community})
                    CREATE (f)-[:APPLIES_TO]->(c)
                )
                """,
                ftype=feedback.feedback_type.value,
                instruction=feedback.instruction,
                created_at=feedback.created_at.isoformat(),
                active=feedback.active,
                target_entity=feedback.target_entity,
                target_community=feedback.target_community,
            )

    async def get_active_feedback(
        self, entity_name: Optional[str] = None
    ) -> list[dict]:
        """Retrieve active feedback, optionally filtered by entity."""
        if entity_name:
            query = """
            MATCH (f:Feedback {active: true})-[:APPLIES_TO]->(e:Entity {name: $name})
            RETURN f.feedback_type AS type, f.instruction AS instruction
            """
            params = {"name": entity_name}
        else:
            query = """
            MATCH (f:Feedback {active: true})
            RETURN f.feedback_type AS type, f.instruction AS instruction
            """
            params = {}

        async with self._session() as session:
            result = await session.run(query, **params)
            return [
                {"type": r["type"], "instruction": r["instruction"]}
                async for r in result
            ]

    # ─── Density Measurement ──────────────────────────────────────

    async def measure_density(
        self,
        section_id: str,
        referenced_entities: list[str],
        referenced_edges: list[tuple[str, str, str]],
        total_tokens: int,
        subgraph_entity_count: int,
        subgraph_edge_count: int,
    ) -> DensityReport:
        """
        Compute the fact-to-token density for a drafted section.
        
        The draft generator should output the entity names and edges it used,
        which get passed here for verification against the actual graph.
        """
        # Verify entities exist
        verified_entities = set()
        async with self._session() as session:
            for name in referenced_entities:
                result = await session.run(
                    "MATCH (e:Entity {name: $name}) RETURN e.name AS name",
                    name=name,
                )
                record = await result.single()
                if record:
                    verified_entities.add(record["name"])

        # Verify edges exist
        verified_edges = set()
        async with self._session() as session:
            for src, tgt, rel_type in referenced_edges:
                result = await session.run(
                    f"""
                    MATCH (s:Entity {{name: $src}})-[r:{rel_type}]->(t:Entity {{name: $tgt}})
                    RETURN type(r) AS rtype
                    """,
                    src=src,
                    tgt=tgt,
                )
                record = await result.single()
                if record:
                    verified_edges.add((src, tgt, rel_type))

        return DensityReport(
            section_id=section_id,
            total_tokens=total_tokens,
            unique_entities_referenced=len(verified_entities),
            unique_edges_referenced=len(verified_edges),
            subgraph_entities_available=subgraph_entity_count,
            subgraph_edges_available=subgraph_edge_count,
        )
