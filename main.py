"""
Graph Writing Studio — Main Orchestrator

This is the top-level pipeline that chains:
  1. Extraction (text → validated entities/relations)
  2. Ingestion (entities → Neo4j graph)
  3. Community Detection (graph → outline clusters)
  4. Drafting (cluster + 2-hop neighborhood → prose)
  5. Density Measurement (draft → fact-to-token score)
  6. De-slop passes (compress low-density sections)
  7. Feedback capture (human edits → constraint nodes)

Usage:
    python main.py ingest --source contacts/john_smith.txt
    python main.py outline
    python main.py draft --section 0
    python main.py density --section 0
    python main.py feedback --entity "John Smith" --type AVOID_TOPIC --instruction "Don't mention salary"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler

from schema import FeedbackNode, FeedbackType
from extractor import extract_episode
from graph_store import GraphStore

# ─── Setup ────────────────────────────────────────────────────────────

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("gws")


# ─── Commands ─────────────────────────────────────────────────────────

async def cmd_ingest(args: argparse.Namespace):
    """Ingest a text file as an episode into the graph."""
    source_path = Path(args.source)
    if not source_path.exists():
        console.print(f"[red]File not found: {source_path}[/red]")
        return

    text = source_path.read_text(encoding="utf-8")
    source_id = source_path.stem

    console.print(Panel(
        f"Ingesting [bold]{source_id}[/bold]\n"
        f"Text length: {len(text)} chars",
        title="📥 Extraction",
    ))

    # Extract
    episode = await extract_episode(
        text=text,
        source_id=source_id,
        source_type=args.source_type,
        source_timestamp=args.timestamp,
        min_confidence=args.min_confidence,
    )

    # Display extracted entities
    table = Table(title="Extracted Entities")
    table.add_column("Name", style="cyan")
    table.add_column("Label", style="green")
    table.add_column("Aliases")
    table.add_column("Relations", justify="right")

    for entity in episode.entities:
        table.add_row(
            entity.name,
            entity.label.value,
            ", ".join(entity.aliases) if entity.aliases else "—",
            str(len(entity.relations)),
        )

    console.print(table)

    # Display relations
    if any(e.relations for e in episode.entities):
        rel_table = Table(title="Extracted Relations")
        rel_table.add_column("Source", style="cyan")
        rel_table.add_column("→ Type", style="yellow")
        rel_table.add_column("Target", style="cyan")
        rel_table.add_column("Confidence", justify="right")
        rel_table.add_column("Context", max_width=50)

        for entity in episode.entities:
            for rel in entity.relations:
                rel_table.add_row(
                    entity.name,
                    rel.relationship_type.value,
                    rel.target_entity,
                    f"{rel.confidence:.2f}",
                    rel.context[:50] + "…" if len(rel.context) > 50 else rel.context,
                )

        console.print(rel_table)

    # Load into Neo4j
    if not args.dry_run:
        store = GraphStore()
        await store.setup_indexes()
        stats = await store.ingest_episode(episode)
        await store.close()

        console.print(Panel(
            f"Nodes created: {stats['nodes_created']}\n"
            f"Nodes merged:  {stats['nodes_merged']}\n"
            f"Rels created:  {stats['rels_created']}",
            title="✅ Loaded into Neo4j",
        ))
    else:
        console.print("[dim]Dry run — skipped Neo4j ingestion[/dim]")

    # Save extraction as JSON for debugging
    output_path = Path("extractions") / f"{source_id}.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(
        episode.model_dump_json(indent=2),
        encoding="utf-8",
    )
    console.print(f"[dim]Saved extraction to {output_path}[/dim]")


async def cmd_outline(args: argparse.Namespace):
    """Run community detection and display the proposed outline."""
    store = GraphStore()

    console.print(Panel("Running community detection…", title="🗂️ Outline Discovery"))

    communities = await store.run_community_detection(
        algorithm=args.algorithm
    )

    if not communities:
        console.print("[yellow]No communities found. Ingest more data first.[/yellow]")
        await store.close()
        return

    table = Table(title=f"Proposed Outline ({len(communities)} sections)")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Members", style="cyan")
    table.add_column("Size", justify="right")

    for i, comm in enumerate(communities):
        table.add_row(
            str(i),
            ", ".join(comm["members"][:8])
            + (f" (+{comm['size'] - 8} more)" if comm["size"] > 8 else ""),
            str(comm["size"]),
        )

    console.print(table)
    console.print(
        "\n[dim]Review these clusters. Use 'feedback' command to merge/drop "
        "communities before drafting.[/dim]"
    )

    # Save for drafting
    Path("outline.json").write_text(
        json.dumps(communities, indent=2),
        encoding="utf-8",
    )

    await store.close()


async def cmd_draft(args: argparse.Namespace):
    """Draft a section using the 2-hop neighborhood of a community's entities."""
    outline_path = Path("outline.json")
    if not outline_path.exists():
        console.print("[red]No outline found. Run 'outline' first.[/red]")
        return

    communities = json.loads(outline_path.read_text())

    if args.section >= len(communities):
        console.print(f"[red]Section {args.section} doesn't exist (max: {len(communities) - 1})[/red]")
        return

    community = communities[args.section]
    console.print(Panel(
        f"Drafting section {args.section}\n"
        f"Entities: {', '.join(community['members'][:5])}…",
        title="✍️ Drafting",
    ))

    store = GraphStore()

    # Gather 2-hop neighborhoods for all community members
    all_entities = set()
    all_relations = []
    for member in community["members"]:
        neighborhood = await store.get_neighborhood(
            entity_name=member,
            hops=2,
            active_only=True,
        )
        for e in neighborhood["entities"]:
            all_entities.add(e["name"])
        all_relations.extend(neighborhood["relations"])

    # Get active feedback constraints
    feedback = await store.get_active_feedback()
    await store.close()

    # Deduplicate relations
    seen = set()
    unique_relations = []
    for r in all_relations:
        key = (r["source"], r["target"], r["type"])
        if key not in seen:
            seen.add(key)
            unique_relations.append(r)

    # Build the drafting prompt
    facts_block = "\n".join(
        f"- {r['source']} --[{r['type']}]--> {r['target']}: {r['context']}"
        for r in unique_relations
    )

    constraints_block = ""
    if feedback:
        constraints_block = "\n\nCONSTRAINTS (from human feedback):\n" + "\n".join(
            f"- [{f['type']}] {f['instruction']}" for f in feedback
        )

    drafting_prompt = f"""You are writing one section of a factual report.

AVAILABLE FACTS (from knowledge graph):
{facts_block}
{constraints_block}

INSTRUCTIONS:
1. First, list the key facts you will use (Chain of Thought).
2. Then write 2-4 paragraphs of dense, factual prose.
3. After the prose, list the entity names and relationships you referenced, 
   in this exact JSON format:
   {{"entities_used": ["Name1", "Name2"], "edges_used": [["Source", "Target", "REL_TYPE"]]}}

Write the section now:"""

    console.print(Panel(drafting_prompt[:500] + "…", title="📋 Prompt Preview"))
    console.print(
        f"\n[bold]Subgraph stats:[/bold] {len(all_entities)} entities, "
        f"{len(unique_relations)} relations"
    )

    # In production, this prompt goes to Ollama. For now, save it.
    draft_dir = Path("drafts")
    draft_dir.mkdir(exist_ok=True)
    (draft_dir / f"section_{args.section}_prompt.txt").write_text(
        drafting_prompt, encoding="utf-8"
    )
    console.print(f"\n[dim]Prompt saved to drafts/section_{args.section}_prompt.txt[/dim]")
    console.print("[dim]Send this prompt to Ollama to generate the draft.[/dim]")


async def cmd_feedback(args: argparse.Namespace):
    """Store a human feedback node in the graph."""
    try:
        ftype = FeedbackType(args.type.upper())
    except ValueError:
        valid = ", ".join(ft.value for ft in FeedbackType)
        console.print(f"[red]Invalid feedback type. Valid: {valid}[/red]")
        return

    feedback = FeedbackNode(
        feedback_type=ftype,
        target_entity=args.entity,
        target_community=args.community,
        instruction=args.instruction,
    )

    store = GraphStore()
    await store.store_feedback(feedback)
    await store.close()

    console.print(Panel(
        f"Type: {feedback.feedback_type.value}\n"
        f"Target: {feedback.target_entity or f'Community {feedback.target_community}'}\n"
        f"Instruction: {feedback.instruction}",
        title="💬 Feedback Stored",
    ))


async def cmd_density(args: argparse.Namespace):
    """Measure the fact-to-token density of a draft (placeholder)."""
    console.print(Panel(
        "Density measurement requires a completed draft.\n"
        "After running Ollama on the draft prompt, parse the JSON footer\n"
        "to get entities_used and edges_used, then call:\n\n"
        "  store.measure_density(\n"
        "      section_id='section_0',\n"
        "      referenced_entities=[...],\n"
        "      referenced_edges=[...],\n"
        "      total_tokens=<token_count>,\n"
        "      subgraph_entity_count=<from outline>,\n"
        "      subgraph_edge_count=<from outline>,\n"
        "  )\n\n"
        "Target: > 5 facts per 100 tokens for dense prose.",
        title="📊 Density Check",
    ))


# ─── CLI ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gws",
        description="Graph Writing Studio — local, graph-powered content generation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Extract and load a text source")
    p_ingest.add_argument("--source", required=True, help="Path to text file")
    p_ingest.add_argument("--source-type", default="contact", help="Type of source")
    p_ingest.add_argument("--timestamp", default=None, help="Source timestamp (ISO 8601)")
    p_ingest.add_argument("--min-confidence", type=float, default=0.5)
    p_ingest.add_argument("--dry-run", action="store_true", help="Extract only, skip Neo4j")

    # outline
    p_outline = subparsers.add_parser("outline", help="Discover outline via community detection")
    p_outline.add_argument("--algorithm", default="louvain", choices=["louvain", "leiden"])

    # draft
    p_draft = subparsers.add_parser("draft", help="Generate a draft for a section")
    p_draft.add_argument("--section", type=int, required=True, help="Section index from outline")

    # feedback
    p_fb = subparsers.add_parser("feedback", help="Store editorial feedback")
    p_fb.add_argument("--type", required=True, help="Feedback type")
    p_fb.add_argument("--entity", default=None, help="Target entity name")
    p_fb.add_argument("--community", type=int, default=None, help="Target community ID")
    p_fb.add_argument("--instruction", required=True, help="Feedback directive")

    # density
    p_density = subparsers.add_parser("density", help="Measure draft density")
    p_density.add_argument("--section", type=int, required=True)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cmd_map = {
        "ingest": cmd_ingest,
        "outline": cmd_outline,
        "draft": cmd_draft,
        "feedback": cmd_feedback,
        "density": cmd_density,
    }

    asyncio.run(cmd_map[args.command](args))


if __name__ == "__main__":
    main()
