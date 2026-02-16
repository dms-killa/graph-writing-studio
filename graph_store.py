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
    DensityReport,
    Entity,
    Episode,
    FeedbackNode,
    Relation,
)

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "graphstudio"


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
            time_filter = "AND (r.valid_to IS NULL)"

        query = f"""
        MATCH path = (start:Entity {{name: $name}})-[r*1..{hops}]-(neighbor:Entity)
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
            # Project the graph into GDS
            try:
                await session.run(
                    "CALL gds.graph.drop('entity_graph', false)"
                )
            except Exception:
                pass

            await session.run(
                """
                CALL gds.graph.project(
                    'entity_graph',
                    'Entity',
                    '*',
                    {relationshipProperties: 'confidence'}
                )
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
