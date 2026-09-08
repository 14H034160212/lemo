"""
zebra_solver.py
A from-scratch NL-clue constraint solver for ZebraLogicBench (allenai/ZebraLogicBench,
mc_mode). The official public test split withholds ground-truth answers (to prevent
leaderboard contamination), so we derive our own gold labels by solving each puzzle
with a backtracking CSP solver over the *actual* category structure declared in the
puzzle preamble, and cross-checking every clue against the solution.

Correctness-first design:
  - The preamble ("Each person has a unique NAME: `A`, `B`, ...") is parsed to recover
    every category and its exhaustive member list. Each category is solved as a
    permutation (bijection) of its members onto houses 1..N -- this is what actually
    gives puzzles a unique solution (a free-variable model without this is
    under-constrained and was rejected after testing: see git history / conversation).
  - Every descriptor phrase used in a clue ("Eric", "the person who loves baseball",
    "the Dane") is resolved to a specific (category, member) pair via whole-word
    matching against the category member lists recovered above. If a descriptor
    cannot be resolved unambiguously, the puzzle is skipped.
  - Every clue is matched against a small, closed set of templates (identity/same-house,
    adjacency, strict left/right, "between", fixed house, negated fixed house). Any
    clue that does not match a known template, or any descriptor that fails to
    resolve, causes the WHOLE puzzle to be skipped (never guessed).
  - The search finds a satisfying joint assignment across all categories with
    incremental constraint checking, then re-validates the *complete* solution against
    every parsed clue from scratch. If zero or more-than-one satisfying assignment is
    found, the puzzle is skipped rather than answered.
  - Skip counts are reported so coverage is transparent, not silently inflated.
"""

import re
from itertools import permutations

ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9,
}
BETWEEN_WORDS = {"one": 1, "two": 2, "three": 3}


def parse_categories(puzzle_text):
    """Recover (category_label, [members...]) from preamble bullet lines."""
    categories = []
    for line in puzzle_text.split("\n"):
        line = line.strip()
        if not line.startswith("-"):
            continue
        if ":" not in line:
            continue
        label, rest = line.split(":", 1)
        members = re.findall(r"`([^`]+)`", rest)
        if len(members) >= 2:
            categories.append((label.strip("- ").strip(), members))
    return categories


def resolve_descriptor(text, categories):
    """Map a descriptor phrase to (cat_idx, member) via whole-word substring match.
    Prefers the longest matching member string; returns None if no clean match."""
    text_l = text.lower()
    best = None  # (len(member), cat_idx, member)
    for cat_idx, (_, members) in enumerate(categories):
        for member in members:
            pattern = r"\b" + re.escape(member.lower()) + r"\b"
            if re.search(pattern, text_l):
                cand = (len(member), cat_idx, member)
                if best is None or cand[0] > best[0]:
                    best = cand
    if best is None:
        return None
    return (best[1], best[2])


def parse_clue(line, categories, constraints):
    line = line.strip().rstrip(".")
    line = re.sub(r"^\d+\.\s*", "", line)

    if "person's child is named" in line or "the mother of" in line.lower():
        return False  # known-buggy phrasing; see module docstring

    def resolve(txt):
        return resolve_descriptor(txt, categories)

    m = re.match(r"There (?:is|are) (\w+) houses? between (.+) and (.+)$", line, re.I)
    if m:
        n_word, a, b = m.groups()
        if n_word.lower() not in BETWEEN_WORDS:
            return False
        ra, rb = resolve(a), resolve(b)
        if not ra or not rb:
            return False
        constraints.append(("DIST", ra, rb, BETWEEN_WORDS[n_word.lower()] + 1))
        return True

    m = re.match(r"(.+) and (.+) are next to each other$", line, re.I)
    if m:
        a, b = m.groups()
        ra, rb = resolve(a), resolve(b)
        if not ra or not rb:
            return False
        constraints.append(("DIST", ra, rb, 1))
        return True

    m = re.match(r"(.+) is directly left of (.+)$", line, re.I)
    if m:
        a, b = m.groups()
        ra, rb = resolve(a), resolve(b)
        if not ra or not rb:
            return False
        constraints.append(("ADJ_LEFT", ra, rb))
        return True

    m = re.match(r"(.+) is directly right of (.+)$", line, re.I)
    if m:
        a, b = m.groups()
        ra, rb = resolve(a), resolve(b)
        if not ra or not rb:
            return False
        constraints.append(("ADJ_LEFT", rb, ra))
        return True

    m = re.match(r"(.+) is somewhere to the left of (.+)$", line, re.I)
    if m:
        a, b = m.groups()
        ra, rb = resolve(a), resolve(b)
        if not ra or not rb:
            return False
        constraints.append(("LT", ra, rb))
        return True

    m = re.match(r"(.+) is somewhere to the right of (.+)$", line, re.I)
    if m:
        a, b = m.groups()
        ra, rb = resolve(a), resolve(b)
        if not ra or not rb:
            return False
        constraints.append(("LT", rb, ra))
        return True

    m = re.match(r"(.+) is not in the (\w+) house$", line, re.I)
    if m:
        a, ord_word = m.groups()
        ra = resolve(a)
        if not ra or ord_word.lower() not in ORDINALS:
            return False
        constraints.append(("NOT_FIXED", ra, ORDINALS[ord_word.lower()]))
        return True

    m = re.match(r"(.+) is in the (\w+) house$", line, re.I)
    if m:
        a, ord_word = m.groups()
        ra = resolve(a)
        if not ra or ord_word.lower() not in ORDINALS:
            return False
        constraints.append(("FIXED", ra, ORDINALS[ord_word.lower()]))
        return True

    # Generic same-house fallback: "X is Y" (also covers "X is the Y", "X is the
    # person who Z", "The person who Y is the person who Z", etc.)
    m = re.match(r"(.+?) is (.+)$", line, re.I)
    if m:
        a, b = m.groups()
        ra, rb = resolve(a), resolve(b)
        if not ra or not rb:
            return False
        if ra[0] == rb[0]:
            return False  # same category on both sides -> our resolution is wrong
        constraints.append(("EQ", ra, rb))
        return True

    return False


def parse_puzzle(puzzle_text):
    n_match = re.search(r"There are (\d+) houses", puzzle_text)
    if not n_match:
        return None
    n = int(n_match.group(1))
    categories = parse_categories(puzzle_text)
    if not categories or any(len(members) != n for _, members in categories):
        return None

    clue_lines = [l for l in puzzle_text.split("\n") if re.match(r"^\s*\d+\.", l)]
    constraints = []
    for line in clue_lines:
        if not parse_clue(line, categories, constraints):
            return None
    return {"n": n, "categories": categories, "constraints": constraints}


def _constraint_ready(c, assigned_cats):
    refs = [r[0] for r in c[1:] if isinstance(r, tuple) and len(r) == 2]
    return all(ci in assigned_cats for ci in refs)


def _constraint_holds(c, pos):
    kind = c[0]
    if kind == "FIXED":
        (ci, m), k = c[1], c[2]
        return pos[(ci, m)] == k
    if kind == "NOT_FIXED":
        (ci, m), k = c[1], c[2]
        return pos[(ci, m)] != k
    if kind == "EQ":
        a, b = c[1], c[2]
        return pos[a] == pos[b]
    if kind == "ADJ_LEFT":
        a, b = c[1], c[2]
        return pos[a] == pos[b] - 1
    if kind == "LT":
        a, b = c[1], c[2]
        return pos[a] < pos[b]
    if kind == "DIST":
        a, b, d = c[1], c[2], c[3]
        return abs(pos[a] - pos[b]) == d
    raise ValueError(kind)


def solve(parsed, max_solutions=2):
    n = parsed["n"]
    categories = parsed["categories"]
    constraints = parsed["constraints"]

    # Pre-filter each category's candidate permutations against constraints that
    # reference only that single category (cheap local pruning before the search).
    cat_local_constraints = []
    for ci in range(len(categories)):
        local = [c for c in constraints
                 if all((r[0] == ci) for r in c[1:] if isinstance(r, tuple) and len(r) == 2)]
        cat_local_constraints.append(local)

    def local_ok(ci, perm):
        members = categories[ci][1]
        pos = {(ci, m): p for m, p in zip(members, perm)}
        return all(_constraint_holds(c, pos) for c in cat_local_constraints[ci])

    cat_candidates = []
    for ci, (_, members) in enumerate(categories):
        cands = [perm for perm in permutations(range(1, n + 1)) if local_ok(ci, perm)]
        cat_candidates.append(cands)
        if not cands:
            return []

    solutions = []
    assigned_pos = {}

    def backtrack(idx, assigned_cats):
        if len(solutions) >= max_solutions:
            return
        if idx == len(categories):
            solutions.append(dict(assigned_pos))
            return
        members = categories[idx][1]
        for perm in cat_candidates[idx]:
            for m, p in zip(members, perm):
                assigned_pos[(idx, m)] = p
            new_assigned = assigned_cats | {idx}
            ok = True
            for c in constraints:
                if _constraint_ready(c, new_assigned) and (idx in [r[0] for r in c[1:] if isinstance(r, tuple) and len(r) == 2]):
                    if not _constraint_holds(c, assigned_pos):
                        ok = False
                        break
            if ok:
                backtrack(idx + 1, new_assigned)
            for m, _ in zip(members, perm):
                del assigned_pos[(idx, m)]
            if len(solutions) >= max_solutions:
                return

    backtrack(0, set())
    return solutions


def solve_and_answer(example):
    parsed = parse_puzzle(example["puzzle"])
    if parsed is None:
        return None
    solutions = solve(parsed, max_solutions=2)
    if len(solutions) != 1:
        return None
    pos = solutions[0]

    # Full re-validation from scratch (defence in depth against solver bugs).
    for c in parsed["constraints"]:
        if not _constraint_holds(c, pos):
            return None

    q_match = re.search(r"House (\d+)", example["question"])
    if not q_match:
        return None
    target_house = int(q_match.group(1))

    answer = None
    for choice in example["choices"]:
        r = resolve_descriptor(choice, parsed["categories"])
        if r is None:
            return None
        if pos[r] == target_house:
            if answer is not None:
                return None
            answer = choice
    return answer


if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("allenai/ZebraLogicBench", "mc_mode", split="test")
    n_total = 0
    n_solved = 0
    solved_examples = []
    for ex in ds.select(range(300)):
        n_total += 1
        ans = solve_and_answer(ex)
        if ans is not None:
            n_solved += 1
            solved_examples.append((ex["id"], ans))
    print(f"Solved {n_solved}/{n_total} sampled puzzles (rest skipped as unparseable/non-unique).")
    for ex_id, ans in solved_examples[:10]:
        print(" ", ex_id, "->", ans)
