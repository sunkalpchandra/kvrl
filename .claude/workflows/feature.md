# Workflow: feature

1. Read CLAUDE.md checklist files. Confirm the feature is in TODO.md (add if not).
2. If architectural: consult DECISIONS.md; if new decision needed, ask relevant agent
   roles for independent analysis; lead synthesises; append decision.
3. Implement in small units. Each unit: code + test + `ruff check` + `pytest -q -m "not slow"`.
4. Commit per unit with conventional prefix; push every few commits.
5. QA pass (role prompt) on the finished feature; fix; add regression tests.
6. Update STATUS.md / TODO.md; BUGS.md if anything was found.
