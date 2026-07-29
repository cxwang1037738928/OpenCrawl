"""
entity_resolution.py — lexical entity resolution for a knowledge graph

Entities arrive as raw model output, so one concept turns up under many surface
forms: 'Random Forest', 'random forest', 'RandomForest', 'random-forest'. This
module folds those onto a normalized key and rewrites the relations onto one
representative per group. No model calls — the whole pass is string work.

Case folds only where it is safe. Chemical formulae carry meaning in their
capitalization, and so do the acronyms sitting next to them: CO (carbon
monoxide) is not Co (cobalt), CaS is not CAS, Ga is not GA, Ea is not EA.
Measured on a 192-document materials corpus, 67 of 900 case-collisions were
distinct things, so short and formula-shaped names keep their case and only
longer prose-shaped names fold.

Crawler-agnostic: the graph payload is the only contract.

Contents:
  Keys        resolution_key (+ CASE_FOLD_MIN_CHARS, _FORMULA)
  Resolution  resolve_entities
"""

import collections
import re

from graph_merge import (apply_merge_map, clusters_from_map, merge_map_from_groups,
                         name_counts)

# At or below this length a name never case-folds: two-to-four character strings
# are exactly where element symbols and acronyms collide.
CASE_FOLD_MIN_CHARS = 4

# Element-symbol runs with counts and grouping punctuation — BaTiO3, Fe2O3,
# (NH4)2HPO4. Case is meaningful at any length here, so these are exempt from
# folding even when long. A digit is required: without one, every capital is a
# valid element symbol and an all-caps word like SCIKIT-LEARN reads as a
# formula. Digit-free compounds are short enough for the length guard anyway.
_FORMULA = re.compile(r"^(?=.*\d)(?:[A-Z][a-z]?\d*|[()\[\]·.+\-]|\d)+$")
_JOINERS = re.compile(r"[\s\-_]+")
_PUNCTUATION = re.compile(r"[^\w]")


def resolution_key(entity: str) -> str:
    """The key two entities must share to be treated as the same thing.

    Whitespace, hyphens, underscores and punctuation always fold; case folds
    only for names long enough, and shaped unlike a formula, to be safe.
    """
    stripped = entity.strip()
    key = _PUNCTUATION.sub("", _JOINERS.sub("", stripped))
    if len(stripped) > CASE_FOLD_MIN_CHARS and not _FORMULA.match(stripped):
        key = key.lower()
    return key


def resolve_entities(graph: dict) -> tuple[dict, dict]:
    """Fold surface variants of one entity onto a single representative.

    Returns (resolved, stats). The merge map lands in the payload as
    entityClusters, so which forms collapsed is recoverable after the fact.
    """
    groups = collections.defaultdict(list)
    for entity in graph["entities"]:
        groups[resolution_key(entity)].append(entity)

    counts = name_counts(graph["relations"], "entity")
    merge_map = merge_map_from_groups(groups.values(), counts)
    resolved, stats = apply_merge_map(graph, entity_map=merge_map)

    resolved["entityClusters"] = clusters_from_map(merge_map)
    stats["groups"] = len(resolved["entityClusters"])
    return resolved, stats
