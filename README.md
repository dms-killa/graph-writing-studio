# Graph Writing Studio

A local-first, graph-powered writing pipeline that extracts knowledge from
documents, organizes it via community detection, and generates dense, factual
prose — with human-in-the-loop feedback at every stage.

## Architecture

```
Text Files → Ollama Extraction → Pydantic Validation → Neo4j Graph
                                                           ↓
                        Drafting ← 2-hop Subgraph ← Community Detection
                            ↓
                    Density Measurement → De-slop Passes
                            ↓
                    Human Feedback → Constraint Nodes → (repeat)
```

## Prerequisites

- **Docker & Docker Compose** — for Neo4j with Graph Data Science plugin
- **Ollama** — running locally with `llama3.1:70b` (or `8b` for faster iteration)
- **Python 3.11+**

## Quick Start

### 1. Start Neo4j

```bash
docker compose up -d
```

Neo4j Browser will be available at http://localhost:7474 (login: neo4j / graphstudio).

### 2. Pull the Ollama model

```bash
# For best extraction quality (requires ~40GB RAM):
ollama pull llama3.1:70b

# For faster iteration on weaker hardware:
ollama pull llama3.1:8b
# Then edit extractor.py: EXTRACTION_MODEL = "llama3.1:8b"
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Ingest sample contacts

```bash
# Dry run (extraction only, no Neo4j):
python main.py ingest --source samples/john_smith.txt --dry-run

# Full ingestion:
python main.py ingest --source samples/john_smith.txt
python main.py ingest --source samples/sarah_chen.txt
```

### 5. Discover the outline

```bash
python main.py outline
# Or with Leiden algorithm:
python main.py outline --algorithm leiden
```

### 6. Draft a section

```bash
python main.py draft --section 0
# This generates a prompt in drafts/section_0_prompt.txt
# Send it to Ollama manually or pipe it:
# ollama run llama3.1:70b < drafts/section_0_prompt.txt
```

### 7. Store feedback

```bash
python main.py feedback \
    --type AVOID_TOPIC \
    --entity "John Smith" \
    --instruction "Don't mention salary or compensation details"
```

## Project Structure

```
graph-writing-studio/
├── docker-compose.yml     # Neo4j + GDS plugin
├── requirements.txt
├── schema.py              # Pydantic models (Entity, Relation, Episode, etc.)
├── extractor.py           # Ollama extraction pipeline (two-pass)
├── graph_store.py         # Neo4j operations (ingest, query, community, feedback)
├── main.py                # CLI orchestrator
├── samples/               # Sample text files for testing
│   ├── john_smith.txt
│   └── sarah_chen.txt
├── extractions/           # Auto-generated JSON from extraction runs
├── drafts/                # Auto-generated drafting prompts
└── outline.json           # Auto-generated community outline
```

## Key Design Decisions

### Two-Pass Extraction
Entities are extracted first, then relations are extracted with knowledge of
the entity set. This reduces hallucinated relationships and ensures targets
are valid entities.

### Temporal Properties
Every relationship carries `valid_from` and `valid_to` timestamps. This
enables point-in-time queries and automatic conflict resolution when facts
change across episodes.

### Fact-to-Token Density
The de-slop metric counts verified graph edges referenced per 100 tokens of
draft text. Target: >5 facts/100 tokens for dense prose.

### Feedback as Graph Nodes
Human editorial preferences (avoid topics, correct facts, merge entities) are
stored as `Feedback` nodes linked to the relevant entity or community. The
drafting prompt automatically includes active constraints.

## Configuration

Edit the constants at the top of each module:

| Setting | File | Default |
|---------|------|---------|
| Ollama URL | `extractor.py` | `http://localhost:11434` |
| Ollama model | `extractor.py` | `llama3.1:70b` |
| Temperature | `extractor.py` | `0.1` |
| Neo4j URI | `graph_store.py` | `bolt://localhost:7687` |
| Neo4j password | `graph_store.py` / `docker-compose.yml` | `graphstudio` |

## Next Steps

- [ ] Add deduplication pass (embedding-based entity merging)
- [ ] Implement the de-slop compression pass with Ollama
- [ ] Build Streamlit UI for HITL review
- [ ] Add batch ingestion for multiple files
- [ ] Integrate with Graphiti for richer temporal semantics
