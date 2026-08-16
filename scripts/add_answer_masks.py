#!/usr/bin/env python
"""Add an ``answer_mask`` (bool[T]) to existing traces: the tokens of the task's answer string
inside the critical span(s). Needed because a needle's *frame* ("Reminder: the access code for
X is") and its *answer* ("4855-kilo") are equally "critical" in the old mask, but only the
answer tokens decide task success (E-007/E-014 analysis).

    python scripts/add_answer_masks.py data/raw/train data/raw/val
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer

from kvrl.traces.collector import answer_token_mask as answer_mask_for


def main() -> int:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    n_done = n_ans = 0
    for d in sys.argv[1:]:
        for p in sorted(Path(d).glob("*.npz")):
            z = dict(np.load(p, allow_pickle=False))
            meta = json.loads(str(z["meta"]))
            am = answer_mask_for(z["token_ids"], z["critical_mask"], meta.get("answers") or [], tok)
            z["answer_mask"] = am
            np.savez_compressed(p, **z)
            n_done += 1
            n_ans += int(am.any())
    print(f"[answer_mask] updated {n_done} traces ({n_ans} with answer tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
