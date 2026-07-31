"""
graph_merge.py — apply a merge map to a knowledge graph payload

Shared by entity_resolution.py (lexical keys) and predicate_normalization.py
(embeddings): both decide WHICH names collapse together, then hand the groups
here to be applied. Crawler-agnostic — the payload shape is the only contract,
so sapphire, ruby and topaz graphs all go through the same rewrite.

A merge map is {original name: representative}. Applying it rewrites every
relation onto representatives, drops the triples that collapse onto each other,
and recomputes the edge set from the survivors.

Provenance rides alongside as relationDocIds — a list index-aligned with
relations, naming the document(s) each triple was extracted from. It is kept
aligned here rather than by each caller, because every rewrite that reorders or
collapses relations has to carry it.

Contents:
  Counting     name_counts
  Grouping     choose_representative, merge_map_from_groups, clusters_from_map
  Recording    fold_clusters
  Provenance   sources_by_relation, aligned_sources
  Applying     apply_merge_map
"""

import collections


def name_counts(relations, position: str) -> collections.Counter:
    """How often each name is used, to decide which surface form wins.

    position is "entity" (counts subjects and objects) or "predicate".
    """
    counts = collections.Counter()
    for subject, predicate, obj in relations:
        if position == "predicate":
            counts[predicate] += 1
        else:
            counts[subject] += 1
            counts[obj] += 1
    return counts


def choose_representative(members, counts) -> str:
    """The form a group collapses onto: most used in relations, then shortest,
    then alphabetical so the choice never depends on iteration order."""
    return min(members, key=lambda name: (-counts[name], len(name), name))


def merge_map_from_groups(groups, counts) -> dict[str, str]:
    """{name: representative} for every group with something to merge."""
    merge_map = {}
    for members in groups:
        if len(members) < 2:
            continue
        representative = choose_representative(members, counts)
        for name in members:
            merge_map[name] = representative
    return merge_map


def clusters_from_map(merge_map: dict[str, str]) -> dict[str, list[str]]:
    """Invert a merge map into {representative: members}, the shape kg-gen uses
    for its cluster output. Kept in the payload so a collapse can be audited."""
    clusters = collections.defaultdict(list)
    for name, representative in merge_map.items():
        clusters[representative].append(name)
    return {representative: sorted(members) for representative, members in clusters.items()}


def fold_clusters(existing: dict[str, list[str]],
                  new: dict[str, list[str]]) -> dict[str, list[str]]:
    """Merge `new` clusters into `existing`, following names that were themselves
    representatives.

    A pass running after another one can merge away a name the earlier pass had
    already made a representative: abbreviations fold 'AI' into 'artificial
    intelligence', but 'AI' was already representing {A.I., AI}. Appending alone
    would leave 'AI' as a key naming a node that no longer exists, and strand
    'A.I.' under it — so the group is absorbed and the dead key dropped, keeping
    every key a live entity and every lineage in one place.

    Absorption is transitive: two passes can only chain two deep, but a third
    entity pass would compound, and following the chain costs nothing.
    """
    folded = dict(existing)
    for representative, members in new.items():
        group = set(folded.get(representative, ())) | set(members)
        pending = [member for member in group if member != representative]
        while pending:
            member = pending.pop()
            if member == representative or member not in folded:
                continue
            absorbed = set(folded.pop(member))
            pending.extend(absorbed - group)
            group |= absorbed
        folded[representative] = sorted(group)
    return folded


def sources_by_relation(graph: dict) -> dict[tuple, set[str]]:
    """{(subject, predicate, object): {docId}} from the payload's aligned lists.

    Empty when the graph predates provenance, so every caller degrades to a
    no-op rather than failing on an older graph.json.
    """
    doc_ids = graph.get("relationDocIds") or []
    return {tuple(relation): set(ids)
            for relation, ids in zip(graph["relations"], doc_ids)}


def aligned_sources(relations, sources: dict[tuple, set[str]]) -> list[list[str]]:
    """relationDocIds for `relations`, in the same order, sorted for a stable file."""
    return [sorted(sources.get(tuple(relation), ())) for relation in relations]


def apply_merge_map(graph: dict, entity_map: dict = None,
                    predicate_map: dict = None) -> tuple[dict, dict]:
    """Rewrite a graph payload onto its representatives.

    Returns (merged, stats). `merged` is a copy with entities / relations /
    edges replaced; every other field carries over. Relations are a set, so
    triples that become identical after merging collapse into one.
    """
    entity_map = entity_map or {}
    predicate_map = predicate_map or {}

    sources = sources_by_relation(graph)
    relations = set()
    merged_sources: dict[tuple, set[str]] = {}
    for relation in graph["relations"]:
        subject, predicate, obj = relation
        merged_relation = (entity_map.get(subject, subject),
                           predicate_map.get(predicate, predicate),
                           entity_map.get(obj, obj))
        relations.add(merged_relation)
        # Two triples collapsing into one are still attested by both their
        # documents, so the sources union rather than overwrite.
        origin = sources.get(tuple(relation))
        if origin:
            merged_sources.setdefault(merged_relation, set()).update(origin)
    entities = {entity_map.get(entity, entity) for entity in graph["entities"]}

    merged = dict(graph)
    merged["entities"] = sorted(entities)
    merged["relations"] = sorted(list(relation) for relation in relations)
    merged["edges"] = sorted({relation[1] for relation in relations})
    if graph.get("relationDocIds") is not None:
        merged["relationDocIds"] = aligned_sources(merged["relations"], merged_sources)

    stats = {
        "entitiesMerged":     len(graph["entities"]) - len(entities),
        "predicatesMerged":   len(graph["edges"]) - len(merged["edges"]),
        "relationsCollapsed": len(graph["relations"]) - len(relations),
        "entities":           len(entities),
        "relations":          len(relations),
        "edges":              len(merged["edges"]),
    }
    return merged, stats
