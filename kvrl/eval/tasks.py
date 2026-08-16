"""Long-context task generators with programmatically known answers.

Every task produces :class:`TaskInstance` objects whose ``prompt`` is the user
message (context + question), ``answers`` the acceptable answers, and
``critical_spans`` the character spans of the prompt that are *necessary* to answer
correctly (used to label catastrophically-important tokens vs forgettable ones).

Length control: generators accept ``count_tokens`` (a callable ``str -> int``) so
contexts hit a target token length for the actual tokenizer in use.

Tasks
-----
- ``needle``      needle-in-a-haystack at controlled depth (begin / middle / end)
- ``kv``          key→value retrieval among many distractor pairs
- ``multihop``    two/three-hop entity chains scattered through prose
- ``dependency``  synthetic variable-assignment chains (known causal relevance)
- ``code``        this repository's own source as context; AST-derived questions
- ``lm``          natural-text continuation (NLL of the true continuation)
"""

from __future__ import annotations

import ast
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import Filler

CountFn = Callable[[str], int]


def approx_count_tokens(text: str) -> int:
    """Fallback token estimate (~4 chars/token for English prose)."""
    return max(1, len(text) // 4)


@dataclass
class TaskInstance:
    task: str
    prompt: str
    answers: list[str]
    critical_spans: list[tuple[int, int]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    max_new_tokens: int = 16
    continuation: str | None = None  # for lm task: text whose NLL we score
    system: str = "You are a precise assistant. Answer using only the provided context."

    def critical_mask_chars(self) -> list[bool]:
        m = [False] * len(self.prompt)
        for a, b in self.critical_spans:
            for i in range(max(0, a), min(len(self.prompt), b)):
                m[i] = True
        return m


# --------------------------------------------------------------------------- helpers
_FIRST = [
    "Nora",
    "Tomas",
    "Ingrid",
    "Baptiste",
    "Kenji",
    "Amara",
    "Sofia",
    "Declan",
    "Priya",
    "Mateo",
    "Helga",
    "Yusuf",
    "Lena",
    "Rafael",
    "Zainab",
    "Oskar",
    "Mira",
    "Farid",
    "Elin",
    "Jonah",
]
_LAST = [
    "Ellison",
    "Wren",
    "Halvorsen",
    "Marchetti",
    "Okafor",
    "Lindqvist",
    "Sato",
    "Varga",
    "Brennan",
    "Castellano",
    "Adeyemi",
    "Novak",
    "Fairweather",
    "Ibarra",
    "Kowalski",
    "Thorne",
    "Mensah",
    "Petrov",
    "Quill",
    "Roux",
]
_CITIES = [
    "Kestrel Bay",
    "Harrowgate",
    "Port Elm",
    "Vellmoor",
    "Silvermere",
    "Dunmarrow",
    "Ashford Cross",
    "Lantern Hill",
    "Greywater",
    "Corvane",
    "Brightholm",
    "Oxmere",
]
_TOPICS = [
    "the vault",
    "the north gate",
    "the archive room",
    "the observatory",
    "the greenhouse",
    "the harbour office",
    "the west tower",
    "the reading room",
    "the boiler room",
    "the map cabinet",
]


def _name(rng: random.Random, used: set[str]) -> str:
    while True:
        n = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
        if n not in used:
            used.add(n)
            return n


def _code(rng: random.Random) -> str:
    suffix = rng.choice(["alpha", "bravo", "delta", "echo", "kilo", "sierra", "tango", "zulu"])
    return f"{rng.randint(1000, 9999)}-{suffix}"


def _fit_filler(
    filler: Filler,
    target_tokens: int,
    count_tokens: CountFn,
    reserve_tokens: int,
    max_iter: int = 4,
) -> str:
    """Sample filler whose token count ≈ target_tokens - reserve_tokens (within ~2%)."""
    want = max(16, target_tokens - reserve_tokens)
    cpt = 4.0
    text = filler.sample(int(want * cpt))
    for _ in range(max_iter):
        n = count_tokens(text)
        if abs(n - want) <= max(4, int(0.02 * want)):
            break
        cpt *= want / max(1, n)
        text = filler.sample(int(want * cpt))
    # trim by paragraphs if still too long
    while count_tokens(text) > want * 1.03 and "\n\n" in text:
        text = text.rsplit("\n\n", 1)[0]
    return text


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _units(haystack: str) -> list[str]:
    """Split text into sentence-ish units (keeps paragraph breaks attached to units)."""
    units: list[str] = []
    for para in haystack.split("\n\n"):
        sents = [x for x in _SENT_SPLIT.split(para) if x]
        if not sents:
            continue
        sents[-1] = sents[-1] + "\n\n"
        units.extend(sents)
    if units:
        units[-1] = units[-1].rstrip("\n")
    return units


def _join_units(units: list[str]) -> str:
    out = []
    for u in units:
        out.append(u)
        if not u.endswith("\n\n"):
            out.append(" ")
    return "".join(out).rstrip(" ")


def _insert_many(
    haystack: str, needles: list[tuple[str, float]]
) -> tuple[str, list[tuple[int, int]]]:
    """Insert several (needle, depth∈[0,1]) at sentence boundaries; returns text and spans."""
    units = _units(haystack)
    n = len(units)
    # map depth (fraction of characters) -> unit boundary index
    cum = [0]
    for u in units:
        cum.append(cum[-1] + len(u) + 1)
    total = max(1, cum[-1])
    positions = []
    for _, d in needles:
        target = min(max(d, 0.0), 1.0) * total
        idx = min(range(n + 1), key=lambda k: abs(cum[k] - target))
        positions.append(idx)
    order = sorted(range(len(needles)), key=lambda i: positions[i])
    out: list[str] = []
    marker: dict[int, str] = {}
    oi = 0
    for u_i in range(n + 1):
        while oi < len(order) and positions[order[oi]] == u_i:
            i = order[oi]
            marker[i] = f"\x00NEEDLE{i}\x00"
            out.append(marker[i])
            oi += 1
        if u_i < n:
            out.append(units[u_i])
    text = _join_units(out)
    for i, (needle, _) in enumerate(needles):
        text = text.replace(marker[i], needle)
    spans: list[tuple[int, int]] = []
    for needle, _ in needles:
        st = text.index(needle)
        spans.append((st, st + len(needle)))
    return text, spans


def _insert_at_depth(haystack: str, needle: str, depth: float) -> tuple[str, tuple[int, int]]:
    text, spans = _insert_many(haystack, [(needle, depth)])
    return text, spans[0]


# --------------------------------------------------------------------------- tasks
def gen_needle(
    n: int,
    target_tokens: int,
    seed: int,
    filler: Filler,
    count_tokens: CountFn,
    depths: list[float] | None = None,
) -> list[TaskInstance]:
    rng = random.Random(seed)
    depths = depths or [0.05, 0.25, 0.5, 0.75, 0.95]
    out = []
    for i in range(n):
        depth = depths[i % len(depths)]
        topic = rng.choice(_TOPICS)
        code = _code(rng)
        needle = f"Reminder: the access code for {topic} is {code}. Keep it safe."
        question = (
            f"\n\nQuestion: According to the text above, what is the access code for "
            f"{topic}? Answer with just the code."
        )
        hay = _fit_filler(
            filler, target_tokens, count_tokens, reserve_tokens=count_tokens(needle + question) + 8
        )
        ctx, span = _insert_at_depth(hay, needle, depth)
        prompt = ctx + question
        out.append(
            TaskInstance(
                "needle",
                prompt,
                [code],
                [span],
                {
                    "depth": depth,
                    "target_tokens": target_tokens,
                    "seed": seed,
                    "filler": filler.source,
                    "topic": topic,
                },
                max_new_tokens=12,
            )
        )
    return out


_KV_WORDS = [
    "harbor",
    "lantern",
    "meadow",
    "copper",
    "willow",
    "falcon",
    "marble",
    "cinder",
    "orchid",
    "quartz",
    "saffron",
    "thistle",
    "velvet",
    "walnut",
    "amber",
    "birch",
    "cobalt",
    "dune",
    "ember",
    "fjord",
    "garnet",
    "heather",
    "ivory",
    "juniper",
    "kestrel",
    "lichen",
    "maple",
    "nettle",
    "onyx",
    "pebble",
    "raven",
    "sable",
    "tundra",
    "umber",
    "violet",
    "wren",
]


def gen_kv(
    n: int,
    target_tokens: int,
    seed: int,
    filler: Filler,
    count_tokens: CountFn,
    n_pairs: int | None = None,
) -> list[TaskInstance]:
    """Key→value retrieval: a block of word-keyed records embedded in prose.

    v1 used ~200 numeric keys with no prose; the full-cache 0.5B model solved 1/4 (E-007), so
    the task could not discriminate controllers. Now: ``n_pairs`` (default 40) records with
    two-word keys, prose before and after, the question at the end. The record line is critical.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pairs = n_pairs or 40
        keys: list[str] = []
        while len(keys) < pairs:
            k = f"{rng.choice(_KV_WORDS)}-{rng.choice(_KV_WORDS)}"
            if k not in keys:
                keys.append(k)
        vals = [_code(rng) for _ in range(pairs)]
        lines = [f"record {k}: {v}" for k, v in zip(keys, vals)]
        target_idx = rng.randrange(pairs)
        block = "Records:\n" + "\n".join(lines) + "\n"
        q = (
            f"\n\nQuestion: In the records above, what is the value of record {keys[target_idx]}? "
            f"Answer with just the value."
        )
        reserve = count_tokens(block + q) + 8
        hay = _fit_filler(filler, target_tokens, count_tokens, reserve)
        depth = rng.uniform(0.1, 0.9)
        ctx, _span = _insert_at_depth(hay, block, depth)
        prompt = ctx + q
        line = lines[target_idx]
        st = prompt.index(line)
        out.append(
            TaskInstance(
                "kv",
                prompt,
                [vals[target_idx]],
                [(st, st + len(line))],
                {
                    "depth": depth,
                    "n_pairs": pairs,
                    "target_tokens": target_tokens,
                    "seed": seed,
                    "filler": filler.source,
                },
                max_new_tokens=12,
            )
        )
    return out


def gen_multihop(
    n: int, target_tokens: int, seed: int, filler: Filler, count_tokens: CountFn, hops: int = 2
) -> list[TaskInstance]:
    rng = random.Random(seed)
    out = []
    for _i in range(n):
        used: set[str] = set()
        person = _name(rng, used)
        city = rng.choice(_CITIES)
        mayor = _name(rng, used)
        facts = [
            f"{person} has lived in {city} for many years.",
            f"The mayor of {city} is {mayor}.",
        ]
        answer = mayor
        question = (
            f"\n\nQuestion: Who is the mayor of the city where {person} lives? "
            f"Answer with just the full name."
        )
        if hops == 3:
            pet = rng.choice(["a grey parrot", "an old tortoise", "a red setter", "two goats"])
            facts.append(f"{mayor} is known for keeping {pet}.")
            answer = pet
            question = (
                f"\n\nQuestion: What animal does the mayor of the city where {person} "
                f"lives keep? Answer briefly."
            )
        # distractor facts about other people/cities
        for _ in range(3):
            facts_d = f"{_name(rng, used)} once visited {rng.choice(_CITIES)} in the spring."
            facts.append(facts_d)
        reserve = count_tokens(" ".join(facts) + question) + 8
        hay = _fit_filler(filler, target_tokens, count_tokens, reserve)
        depths = sorted(rng.uniform(0.05, 0.95) for _ in facts)
        ctx, spans = _insert_many(hay, list(zip(facts, depths)))
        prompt = ctx + question
        crit = spans[:hops]
        out.append(
            TaskInstance(
                "multihop",
                prompt,
                [answer],
                crit,
                {
                    "hops": hops,
                    "depths": depths[:hops],
                    "target_tokens": target_tokens,
                    "seed": seed,
                    "filler": filler.source,
                },
                max_new_tokens=16,
            )
        )
    return out


def gen_dependency(
    n: int,
    target_tokens: int,
    seed: int,
    filler: Filler,
    count_tokens: CountFn,
    chain_len: int = 3,
    n_distractors: int = 6,
    arithmetic: bool = False,
) -> list[TaskInstance]:
    """Variable chains: x0 = c; x1 = x0 (+ d1 if arithmetic); ... ; question asks the final value.

    Every chain sentence is critical; distractor assignments are not. The default is a pure
    copy chain ("Set beta to alpha.") because the 0.5B model cannot do arithmetic chains even
    with a full cache (E-007: 0/4) — a task at the model's ceiling cannot discriminate
    controllers. ``arithmetic=True`` restores the harder variant.
    """
    rng = random.Random(seed)
    names = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "rho",
        "sigma",
        "tau",
    ]
    out = []
    for _i in range(n):
        vs = rng.sample(names, chain_len + n_distractors)
        chain, distract = vs[:chain_len], vs[chain_len:]
        val = rng.randint(3, 40)
        sents = [f"Set {chain[0]} to {val}."]
        for j in range(1, chain_len):
            if arithmetic:
                d = rng.randint(1, 9)
                op = rng.choice(["plus", "minus"])
                val = val + d if op == "plus" else val - d
                sents.append(f"Set {chain[j]} to {chain[j - 1]} {op} {d}.")
            else:
                sents.append(f"Set {chain[j]} to the same value as {chain[j - 1]}.")
        dsents = [f"Set {v} to {rng.randint(1, 60)}." for v in distract]
        question = (
            f"\n\nQuestion: Following the assignments above, what is the value of "
            f"{chain[-1]}? Answer with just the number."
        )
        reserve = count_tokens(" ".join(sents + dsents) + question) + 8
        hay = _fit_filler(filler, target_tokens, count_tokens, reserve)
        all_s = sents + dsents
        depths = [rng.uniform(0.03, 0.97) for _ in all_s]
        ctx, spans = _insert_many(hay, list(zip(all_s, depths)))
        prompt = ctx + question
        out.append(
            TaskInstance(
                "dependency",
                prompt,
                [str(val)],
                spans[:chain_len],
                {
                    "chain_len": chain_len,
                    "target_tokens": target_tokens,
                    "seed": seed,
                    "filler": filler.source,
                },
                max_new_tokens=8,
            )
        )
    return out


def _repo_py_files(root: Path) -> list[Path]:
    files = sorted(
        p
        for p in root.rglob("*.py")
        if ".venv" not in p.parts
        and "node_modules" not in p.parts
        and not p.name.startswith("test_")
    )
    return files


def gen_code(
    n: int,
    target_tokens: int,
    seed: int,
    filler: Filler,
    count_tokens: CountFn,
    root: Path | None = None,
) -> list[TaskInstance]:
    """Use this repository's own source as a long code context; questions from the AST."""
    rng = random.Random(seed)
    root = root or Path(__file__).resolve().parents[2] / "kvrl"
    files = _repo_py_files(root)
    rng.shuffle(files)
    out = []
    for i in range(n):
        # build a context from a random rotation of files until target length: skip files
        # that do not fit and keep looking for smaller ones (never leave a tiny context)
        rot = files[i % len(files) :] + files[: i % len(files)]
        chunks: list[tuple[Path, str]] = []
        total = 0
        limit = target_tokens - 64
        for f in rot:
            src = f.read_text(errors="ignore")
            t = count_tokens(src)
            if total + t > limit:
                if not chunks and t > limit:
                    # single file larger than the budget: truncate by lines to fit
                    lines = src.splitlines(keepends=True)
                    keep_lines = max(10, int(len(lines) * limit / max(1, t)))
                    src = "".join(lines[:keep_lines])
                    chunks.append((f, src))
                    total += count_tokens(src)
                    break
                continue
            chunks.append((f, src))
            total += t
            if total >= limit * 0.9:
                break
        # candidate questions: functions with defaults / parameter counts
        cands = []
        for f, src in chunks:
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(
                    node, ast.FunctionDef | ast.AsyncFunctionDef
                ) and not node.name.startswith("_"):
                    args = node.args
                    nparams = len(args.args) + len(args.kwonlyargs)
                    cands.append((f, node, nparams))
        if not cands:
            continue
        f, node, nparams = rng.choice(cands)
        rel = f.relative_to(root.parent)
        kind = rng.choice(["file", "nparams"])
        if kind == "file":
            question = (
                f"\n\nQuestion: In the code above, which file defines the function "
                f"`{node.name}`? Answer with just the file path."
            )
            answers = [str(rel), rel.name]
        else:
            question = (
                f"\n\nQuestion: In the code above, how many parameters (positional and "
                f"keyword-only, excluding *args/**kwargs) does the function `{node.name}` "
                f"take? Answer with just the number."
            )
            answers = [str(nparams)]
        parts = []
        for ff, src in chunks:
            parts.append(f"# File: {ff.relative_to(root.parent)}\n{src}")
        ctx = "\n\n".join(parts)
        prompt = ctx + question
        # critical span: the def line region of the chosen function
        fsrc = dict(chunks)[f]
        lines = fsrc.splitlines(keepends=True)
        start_off = sum(len(l) for l in lines[: node.lineno - 1])
        end_off = sum(len(l) for l in lines[: min(len(lines), node.lineno + 2)])
        header = f"# File: {rel}\n"
        base = prompt.index(header) + len(header)
        span = (base + start_off, base + end_off)
        out.append(
            TaskInstance(
                "code",
                prompt,
                answers,
                [span],
                {
                    "kind": kind,
                    "function": node.name,
                    "file": str(rel),
                    "target_tokens": target_tokens,
                    "seed": seed,
                },
                max_new_tokens=16,
            )
        )
    return out


def gen_lm(
    n: int,
    target_tokens: int,
    seed: int,
    filler: Filler,
    count_tokens: CountFn,
    continuation_tokens: int = 128,
) -> list[TaskInstance]:
    """Natural text: context = passage, continuation = what actually follows in the book."""
    rng = random.Random(seed)
    out = []
    for _i in range(n):
        text = filler.sample(int((target_tokens + continuation_tokens) * 4.5))
        # split so that continuation is ~continuation_tokens
        words = text.split(" ")
        _lo, hi = 0, len(words)
        while count_tokens(" ".join(words[:hi])) > target_tokens + continuation_tokens and hi > 32:
            hi = int(hi * 0.9)
        cut = hi
        while cut > 8 and count_tokens(" ".join(words[cut:hi])) < continuation_tokens:
            cut -= 8
        ctx = " ".join(words[:cut])
        cont = " " + " ".join(words[cut:hi])
        out.append(
            TaskInstance(
                "lm",
                ctx,
                [],
                [],
                {
                    "target_tokens": target_tokens,
                    "seed": seed,
                    "filler": filler.source,
                    "rng": rng.random(),
                },
                max_new_tokens=0,
                continuation=cont,
                system="",
            )
        )
    return out


TASKS: dict[str, Callable[..., list[TaskInstance]]] = {
    "needle": gen_needle,
    "kv": gen_kv,
    "multihop": gen_multihop,
    "dependency": gen_dependency,
    "code": gen_code,
    "lm": gen_lm,
}


def generate(
    task: str,
    n: int,
    target_tokens: int,
    seed: int,
    filler: Filler,
    count_tokens: CountFn = approx_count_tokens,
    **kw,
) -> list[TaskInstance]:
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; known: {sorted(TASKS)}")
    return TASKS[task](n, target_tokens, seed, filler, count_tokens, **kw)


_WS = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\"'`.,;:!?()\[\]{}]", " ", s)
    return _WS.sub(" ", s).strip()


def is_correct(prediction: str, answers: list[str]) -> bool:
    """Lenient exact/contains match after normalisation (first line of the prediction)."""
    pred = normalize_answer(prediction.split("\n")[0])
    for a in answers:
        na = normalize_answer(a)
        if not na:
            continue
        if pred == na or na in pred:
            return True
    return False
