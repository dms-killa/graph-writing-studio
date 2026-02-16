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
    python main.py ingest --source chat.md --conversation
    python main.py outline
    python main.py draft --section 0
    python main.py draft-conversation --section 0 --conversation chat_id
    python main.py tag-message msg_003 --conversation chat_id --type EVASION --instruction "explain"
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

from schema import FeedbackNode, FeedbackType, TacticalMoveType
from extractor import extract_episode, extract_conversation
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

    # Branch based on whether this is a conversation
    if args.conversation:
        await _ingest_conversation(text, source_id, args)
    else:
        await _ingest_document(text, source_id, args)


async def _ingest_document(text: str, source_id: str, args: argparse.Namespace):
    """Ingest a standard document (original behavior)."""
    console.print(Panel(
        f"Ingesting [bold]{source_id}[/bold]\n"
        f"Text length: {len(text)} chars",
        title="Extraction",
    ))

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
        rel_table.add_column("-> Type", style="yellow")
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
                    rel.context[:50] + "..." if len(rel.context) > 50 else rel.context,
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
            title="Loaded into Neo4j",
        ))
    else:
        console.print("[dim]Dry run -- skipped Neo4j ingestion[/dim]")

    # Save extraction as JSON for debugging
    output_path = Path("extractions") / f"{source_id}.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(
        episode.model_dump_json(indent=2),
        encoding="utf-8",
    )
    console.print(f"[dim]Saved extraction to {output_path}[/dim]")


async def _ingest_conversation(text: str, source_id: str, args: argparse.Namespace):
    """Ingest a conversation transcript."""
    console.print(Panel(
        f"Ingesting conversation [bold]{source_id}[/bold]\n"
        f"Text length: {len(text)} chars",
        title="Conversation Extraction",
    ))

    episode = await extract_conversation(
        text=text,
        source_id=source_id,
        source_timestamp=args.timestamp,
        min_confidence=args.min_confidence,
        extract_tactics=not args.skip_tactics,
    )

    # Display messages summary
    msg_table = Table(title=f"Parsed Messages ({len(episode.messages)} total)")
    msg_table.add_column("#", justify="right", style="bold")
    msg_table.add_column("Speaker", style="green")
    msg_table.add_column("Preview", max_width=60)
    msg_table.add_column("Entities", justify="right")
    msg_table.add_column("Tactics", justify="right")

    for msg in episode.messages:
        preview = msg.content[:60].replace("\n", " ")
        if len(msg.content) > 60:
            preview += "..."
        msg_table.add_row(
            msg.id,
            msg.speaker.value,
            preview,
            str(len(msg.entities_mentioned)),
            str(len(msg.tactical_moves)),
        )

    console.print(msg_table)

    # Display entities
    if episode.entities:
        ent_table = Table(title="Extracted Entities")
        ent_table.add_column("Name", style="cyan")
        ent_table.add_column("Label", style="green")
        ent_table.add_column("Relations", justify="right")

        for entity in episode.entities:
            ent_table.add_row(
                entity.name,
                entity.label.value,
                str(len(entity.relations)),
            )
        console.print(ent_table)

    # Display tactical moves
    tactics_found = [
        (msg.id, msg.speaker.value, tactic)
        for msg in episode.messages
        for tactic in msg.tactical_moves
    ]
    if tactics_found:
        tac_table = Table(title=f"Tactical Moves ({len(tactics_found)} found)")
        tac_table.add_column("Message", style="bold")
        tac_table.add_column("Speaker", style="green")
        tac_table.add_column("Tactic", style="red")
        tac_table.add_column("Evidence", max_width=50)
        tac_table.add_column("Conf", justify="right")

        for msg_id, speaker, tactic in tactics_found:
            evidence_preview = tactic.evidence[:50]
            if len(tactic.evidence) > 50:
                evidence_preview += "..."
            tac_table.add_row(
                msg_id,
                speaker,
                tactic.move_type.value,
                evidence_preview,
                f"{tactic.confidence:.2f}",
            )
        console.print(tac_table)

    # Load into Neo4j
    if not args.dry_run:
        store = GraphStore()
        await store.setup_indexes()
        stats = await store.ingest_conversation(episode)
        await store.close()

        console.print(Panel(
            f"Messages created: {stats['messages_created']}\n"
            f"Nodes created:    {stats['nodes_created']}\n"
            f"Nodes merged:     {stats['nodes_merged']}\n"
            f"Rels created:     {stats['rels_created']}\n"
            f"Tactics stored:   {stats['tactics_stored']}",
            title="Loaded into Neo4j",
        ))
    else:
        console.print("[dim]Dry run -- skipped Neo4j ingestion[/dim]")

    # Save extraction as JSON
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


async def cmd_tag_message(args: argparse.Namespace):
    """Tag a conversation message with feedback (e.g., a tactical classification)."""
    # Validate the feedback type
    valid_tactics = {t.value for t in TacticalMoveType}
    feedback_type = args.type.upper()

    if feedback_type not in valid_tactics:
        # Also accept general feedback types
        try:
            FeedbackType(feedback_type)
        except ValueError:
            all_valid = sorted(valid_tactics | {ft.value for ft in FeedbackType})
            console.print(f"[red]Invalid type. Valid: {', '.join(all_valid)}[/red]")
            return

    store = GraphStore()
    await store.tag_message_feedback(
        message_id=args.message_id,
        conversation_id=args.conversation,
        feedback_type=feedback_type,
        instruction=args.instruction,
    )
    await store.close()

    console.print(Panel(
        f"Message:     {args.message_id}\n"
        f"Conversation: {args.conversation}\n"
        f"Type:        {feedback_type}\n"
        f"Instruction: {args.instruction}",
        title="Message Tagged",
    ))


async def cmd_outline_conversation(args: argparse.Namespace):
    """Run community detection on a conversation's message graph."""
    conversation_id = args.conversation_id

    store = GraphStore()

    console.print(Panel(
        f"Running conversation community detection on [bold]{conversation_id}[/bold]\n"
        f"Algorithm: {args.algorithm}",
        title="Conversation Outline Discovery",
    ))

    communities = await store.detect_conversation_communities(
        conversation_id=conversation_id,
        algorithm=args.algorithm,
    )

    if not communities:
        console.print(
            "[yellow]No communities found. Ensure the conversation has been "
            "ingested first.[/yellow]"
        )
        await store.close()
        return

    # Build the outline data structure
    outline = {
        "conversation_id": conversation_id,
        "algorithm": args.algorithm,
        "sections": [],
    }

    table = Table(title=f"Conversation Outline ({len(communities)} sections)")
    table.add_column("Section", justify="right", style="bold")
    table.add_column("Community", justify="right")
    table.add_column("Messages", style="cyan")
    table.add_column("Count", justify="right")

    for i, (comm_id, message_ids) in enumerate(communities.items()):
        outline["sections"].append({
            "section_id": i,
            "community_id": comm_id,
            "message_ids": message_ids,
            "size": len(message_ids),
        })
        table.add_row(
            str(i),
            str(comm_id),
            ", ".join(message_ids[:6])
            + (f" (+{len(message_ids) - 6} more)" if len(message_ids) > 6 else ""),
            str(len(message_ids)),
        )

    console.print(table)

    # Save the outline
    outline_path = Path("outline_conversation.json")
    outline_path.write_text(
        json.dumps(outline, indent=2),
        encoding="utf-8",
    )
    console.print(f"\n[dim]Saved conversation outline to {outline_path}[/dim]")
    console.print(
        "[dim]Use 'draft-conversation --section <N> --conversation "
        f"{conversation_id}' to draft each section.[/dim]"
    )

    await store.close()


async def cmd_draft_conversation(args: argparse.Namespace):
    """Draft an analytical essay section from a conversation community."""
    # Try conversation outline first, fall back to entity outline
    conv_outline_path = Path("outline_conversation.json")
    entity_outline_path = Path("outline.json")

    use_conversation_outline = False
    section_message_ids = None

    if conv_outline_path.exists():
        conv_outline = json.loads(conv_outline_path.read_text())
        # Check if this outline matches the requested conversation
        if conv_outline.get("conversation_id") == args.conversation:
            sections = conv_outline.get("sections", [])
            if args.section < len(sections):
                section = sections[args.section]
                section_message_ids = section["message_ids"]
                use_conversation_outline = True
                console.print(Panel(
                    f"Drafting conversation section {args.section}\n"
                    f"Messages: {', '.join(section_message_ids[:5])}"
                    + ("..." if len(section_message_ids) > 5 else ""),
                    title="Conversation Drafting (message-based outline)",
                ))
            else:
                console.print(
                    f"[red]Section {args.section} doesn't exist in conversation outline "
                    f"(max: {len(sections) - 1})[/red]"
                )
                return

    if not use_conversation_outline:
        # Fall back to entity-based outline
        if not entity_outline_path.exists():
            console.print(
                "[red]No outline found. Run 'outline-conversation' or 'outline' first.[/red]"
            )
            return

        communities = json.loads(entity_outline_path.read_text())

        if args.section >= len(communities):
            console.print(
                f"[red]Section {args.section} doesn't exist (max: {len(communities) - 1})[/red]"
            )
            return

        community = communities[args.section]
        console.print(Panel(
            f"Drafting conversation section {args.section}\n"
            f"Entities: {', '.join(community['members'][:5])}...",
            title="Conversation Drafting (entity-based outline)",
        ))

    store = GraphStore()

    if use_conversation_outline:
        # Use message-ID-based section data
        section_data = await store.get_conversation_community_section_data(
            conversation_id=args.conversation,
            message_ids=section_message_ids,
        )
    else:
        # Use entity-based section data (original behavior)
        section_data = await store.get_conversation_section_data(
            conversation_id=args.conversation,
            community_members=community["members"],
        )

    # Also get general feedback
    feedback = await store.get_active_feedback()
    await store.close()

    messages = section_data["messages"]
    entity_relations = section_data["entity_relations"]

    if not messages:
        # Fall back: get all messages in the conversation for this section
        store = GraphStore()
        all_messages = await store.get_conversation_thread(args.conversation)
        await store.close()

        if not all_messages:
            console.print("[yellow]No messages found for this conversation/section.[/yellow]")
            return

        messages = all_messages
        console.print(
            f"[dim]No messages found for this section; using all {len(messages)} "
            f"conversation messages.[/dim]"
        )

    # Build the messages block
    messages_block = ""
    for msg in messages:
        speaker = msg.get("speaker", "unknown").upper()
        content = msg.get("content", "")
        msg_id = msg.get("id", "")
        messages_block += f"[{msg_id}] {speaker}: {content}\n\n"

    # Build the tactics block
    tactics_lines = []
    for msg in messages:
        for tactic in msg.get("tactics", []):
            move_type = tactic.get("move_type", "UNKNOWN")
            evidence = tactic.get("evidence", "")
            msg_id = msg.get("id", "")
            tactics_lines.append(f"- [{msg_id}] {move_type}: {evidence}")
    tactics_block = "\n".join(tactics_lines) if tactics_lines else "(none identified)"

    # Build the facts block from entity relations
    facts_block = "\n".join(
        f"- {r['source']} --[{r['type']}]--> {r['target']}: {r.get('context', '')}"
        for r in entity_relations
    ) if entity_relations else "(no entity relationships for this section)"

    # Build constraints block
    constraints_block = ""
    if feedback:
        constraints_block = "\n".join(
            f"- [{f['type']}] {f['instruction']}" for f in feedback
        )
    # Include message-level feedback
    for msg in messages:
        for fb in msg.get("feedback", []):
            constraints_block += f"\n- [{fb.get('feedback_type', '')}] (msg {msg.get('id', '')}): {fb.get('instruction', '')}"
    if not constraints_block.strip():
        constraints_block = "(none)"

    # Load the prompt template
    template_path = Path(args.template) if args.template else Path("prompts/conversation_draft.txt")
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        # Fallback inline template
        template = (
            "Write an analytical essay section about this conversation.\n\n"
            "MESSAGES:\n{messages_block}\n\n"
            "TACTICS:\n{tactics_block}\n\n"
            "FACTS:\n{facts_block}\n\n"
            "CONSTRAINTS:\n{constraints_block}\n\n"
            "Write 3-5 paragraphs of analytical prose:"
        )

    drafting_prompt = template.format(
        messages_block=messages_block,
        tactics_block=tactics_block,
        facts_block=facts_block,
        constraints_block=constraints_block,
    )

    console.print(Panel(drafting_prompt[:500] + "...", title="Prompt Preview"))
    console.print(
        f"\n[bold]Section stats:[/bold] {len(messages)} messages, "
        f"{len(tactics_lines)} tactical moves, "
        f"{len(entity_relations)} entity relations"
    )

    # Save the prompt
    draft_dir = Path("drafts")
    draft_dir.mkdir(exist_ok=True)
    prompt_path = draft_dir / f"conversation_section_{args.section}_prompt.txt"
    prompt_path.write_text(drafting_prompt, encoding="utf-8")
    console.print(f"\n[dim]Prompt saved to {prompt_path}[/dim]")
    console.print("[dim]Send this prompt to Ollama to generate the draft.[/dim]")


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
    p_ingest.add_argument(
        "--conversation", action="store_true",
        help="Treat source as a conversation transcript (chat export)",
    )
    p_ingest.add_argument(
        "--skip-tactics", action="store_true",
        help="Skip tactical move detection during conversation ingestion",
    )

    # outline
    p_outline = subparsers.add_parser("outline", help="Discover outline via community detection")
    p_outline.add_argument("--algorithm", default="louvain", choices=["louvain", "leiden"])

    # outline-conversation
    p_outline_conv = subparsers.add_parser(
        "outline-conversation",
        help="Discover conversation outline via message community detection",
    )
    p_outline_conv.add_argument(
        "conversation_id",
        help="Conversation source ID (as used during ingestion)",
    )
    p_outline_conv.add_argument(
        "--algorithm", default="leiden", choices=["louvain", "leiden"],
        help="Community detection algorithm (default: leiden)",
    )

    # draft
    p_draft = subparsers.add_parser("draft", help="Generate a draft for a section")
    p_draft.add_argument("--section", type=int, required=True, help="Section index from outline")

    # feedback
    p_fb = subparsers.add_parser("feedback", help="Store editorial feedback")
    p_fb.add_argument("--type", required=True, help="Feedback type")
    p_fb.add_argument("--entity", default=None, help="Target entity name")
    p_fb.add_argument("--community", type=int, default=None, help="Target community ID")
    p_fb.add_argument("--instruction", required=True, help="Feedback directive")

    # tag-message
    p_tag = subparsers.add_parser(
        "tag-message", help="Tag a conversation message with feedback",
    )
    p_tag.add_argument("message_id", help="Message ID (e.g., msg_003)")
    p_tag.add_argument(
        "--conversation", required=True,
        help="Conversation source ID",
    )
    p_tag.add_argument(
        "--type", required=True,
        help="Feedback type (e.g., EVASION, PREMATURE_PLURALISM, PALTERING)",
    )
    p_tag.add_argument(
        "--instruction", required=True,
        help="Explanation of why this tag applies",
    )

    # draft-conversation
    p_draftconv = subparsers.add_parser(
        "draft-conversation",
        help="Draft an analytical essay section from a conversation",
    )
    p_draftconv.add_argument(
        "--section", type=int, required=True,
        help="Section index from outline",
    )
    p_draftconv.add_argument(
        "--conversation", required=True,
        help="Conversation source ID",
    )
    p_draftconv.add_argument(
        "--template", default=None,
        help="Path to custom prompt template file",
    )

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
        "outline-conversation": cmd_outline_conversation,
        "draft": cmd_draft,
        "feedback": cmd_feedback,
        "density": cmd_density,
        "tag-message": cmd_tag_message,
        "draft-conversation": cmd_draft_conversation,
    }

    asyncio.run(cmd_map[args.command](args))


if __name__ == "__main__":
    main()
