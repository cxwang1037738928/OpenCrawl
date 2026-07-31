"""
prune_authors.py — drop author nodes from a knowledge graph

kg_graph.py extracts entities chunk by chunk, so any reference list that
survives docling's section split leaks author surnames into the graph as
first-class nodes. This module removes them using the relations that name a
person outright, then drops every relation left touching a removed node.

Which endpoint of a relation is the author depends on the predicate, so the
rule is per-predicate: "X authored Y" names a person on the subject side only
(Y is the paper/tool/journal and stays), while "X co-authored with Y" names
one on both.

Contents:
  Rules      AUTHOR_ENDPOINTS
  Pruning    author_nodes, prune_author_nodes
  CLI        __main__ (graph.json in -> pruned graph.json + kg_view.html out)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_merge import aligned_sources, sources_by_relation

# Predicate -> the endpoints of that relation that name an author. Widen this
# to catch more of the model's phrasings ("co-authored", "is an author of").
AUTHOR_ENDPOINTS = {
    "authored":         ("subject",),
    "co-authored with": ("subject", "object"),
}


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def author_nodes(relations, author_endpoints=AUTHOR_ENDPOINTS) -> set[str]:
    """Collect every node an author predicate names as a person."""
    authors = set()
    for subject, predicate, obj in relations:
        endpoints = author_endpoints.get(predicate)
        if not endpoints:
            continue
        if "subject" in endpoints:
            authors.add(subject)
        if "object" in endpoints:
            authors.add(obj)
    return authors


def prune_author_nodes(graph: dict, author_endpoints=AUTHOR_ENDPOINTS) -> tuple[dict, dict]:
    """Drop author nodes from a graph.json payload.

    Returns (pruned, stats). `pruned` is a copy with entities / relations /
    edges replaced; every other field (token counts, model, doc ids) carries
    over untouched, since pruning does not invalidate them.
    """
    relations = [tuple(relation) for relation in graph["relations"]]
    authors = author_nodes(relations, author_endpoints)

    kept_relations = [relation for relation in relations
                      if relation[0] not in authors and relation[2] not in authors]
    kept_entities = [entity for entity in graph["entities"] if entity not in authors]
    # edges is the set of distinct predicates, so it is recomputed rather than
    # filtered — a predicate survives only if some kept relation still uses it.
    kept_edges = sorted({relation[1] for relation in kept_relations})

    connected = {relation[0] for relation in kept_relations}
    connected |= {relation[2] for relation in kept_relations}

    pruned = dict(graph)
    pruned["entities"] = sorted(kept_entities)
    pruned["relations"] = sorted(list(relation) for relation in kept_relations)
    pruned["edges"] = kept_edges
    # Provenance is index-aligned with relations, so dropping a relation has to
    # drop its sources in the same step or every later lookup is off by one.
    if graph.get("relationDocIds") is not None:
        pruned["relationDocIds"] = aligned_sources(pruned["relations"],
                                                   sources_by_relation(graph))

    stats = {
        "authorNodes": len(authors),
        "droppedEntities": len(graph["entities"]) - len(kept_entities),
        "droppedRelations": len(relations) - len(kept_relations),
        "entities": len(kept_entities),
        "relations": len(kept_relations),
        "edges": len(kept_edges),
        "isolated": len(kept_entities) - len(connected & set(kept_entities)),
    }
    return pruned, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _render_html(pruned: dict, html_path: Path) -> None:
    """Re-render the standalone view. The HTML bakes in the nodes, so a pruned
    graph needs a fresh render — editing the JSON alone leaves the view stale."""
    from kg_gen.models import Graph
    from kg_gen.utils.visualize_kg import visualize

    visualize(Graph(entities=set(pruned["entities"]),
                    relations={tuple(relation) for relation in pruned["relations"]},
                    edges=set(pruned["edges"])),
              str(html_path), open_in_browser=False)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: prune_authors.py <graph.json> <pruned.json> [--no-html]", file=sys.stderr)
        sys.exit(1)

    source_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])
    graph = json.loads(source_path.read_text(encoding="utf-8"))
    pruned, stats = prune_author_nodes(graph)

    output_path.write_text(json.dumps(pruned, indent=2), encoding="utf-8")
    print(f"[prune_authors] {source_path.name}: dropped {stats['authorNodes']} author node(s) "
          f"and {stats['droppedRelations']} relation(s)")
    print(f"[prune_authors] {stats['entities']} entities, {stats['relations']} relations, "
          f"{stats['edges']} relation type(s) -> {output_path}")

    if "--no-html" not in sys.argv:
        html_path = output_path.with_suffix(".html")
        _render_html(pruned, html_path)
        print(f"[prune_authors] re-rendered view -> {html_path}")
