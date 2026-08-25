---
name: grill-me
description: Quiz the user with hard questions about their own code — the current branch's changes by default, or a file, subsystem, or topic passed as an argument. Use when the user asks to be grilled, quizzed, tested, or interviewed on this codebase, or wants to check they can defend a design decision.
---

# Grill Me

Interrogate the user about their own code, one question at a time, until they can defend it.

## Scope

`$ARGUMENTS` sets the scope: a file, directory, subsystem, or topic (e.g. `backend/retrieval`,
`the KG merge passes`, `citation verification`). With no arguments, grill on the current branch's
changes — `git diff main...HEAD` — falling back to the last few commits if that diff is empty.

## Before asking anything

Read the code in scope first. Never ask a question you cannot grade yourself; the answer has to
be in the source you just read, not in your guess about it.

## The loop

1. Ask **one** question. No preamble, no hints, no multiple choice.
2. Stop and wait. Do not answer your own question.
3. Grade the answer out loud: **Correct / Partial / Wrong**, one line of why, citing `file:line`.
4. If it was partial or wrong, follow up on that same gap before moving on.
5. Continue until roughly 8 questions, or the user says stop.

## What counts as a good question

- Ask *why*, not *what*. "Why does the reranker run before dedup instead of after?" beats
  "What does the reranker do?"
- Target decisions with a real alternative — the trade-off should be arguable.
- Ask about failure modes: empty input, the model or DB unreachable, two runs at once, a
  10x larger corpus.
- Include at least one question about something genuinely shaky in the code — a swallowed
  error, an unchecked assumption, a value that only works by luck.
- Skip trivia. Function names and argument order prove nothing.

## Wrap-up

Close with three short lists: what they defended well, where they were shaky (with `file:line`
for each), and what to go read next.

## Rules

- Never soften a wrong answer. Say it's wrong, give the real answer, move on.
- "I don't know" is a fine answer — give the answer immediately and continue.
- Do not edit code during a grilling. Note anything worth fixing for the wrap-up.
