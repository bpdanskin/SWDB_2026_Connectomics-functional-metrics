---
name: user-handles-commits
description: Leave changes in the working tree — the user inspects the diff and commits themselves; do not commit unless asked for that specific commit.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0d22f3a0-6d85-4363-9e71-37a1e3b3d45c
  modified: 2026-08-15T22:19:06.560Z
---

**Do not run `git commit`.** Leave finished work as uncommitted changes in the working
tree and say what changed; the user reviews the diff and commits.

**Why:** they inspect diffs before committing, and that review is part of how they stay
oriented in a codebase an agent is changing quickly. Commits made for them skip the step.

**How to apply:** finish the work, summarise what changed and in which files, and stop.
If a commit is genuinely needed, ask. A one-off request — "help me with this merge" — is
permission for *that* commit only, not a standing grant; I made that mistake and committed
three more times off the back of one merge request.

Pushing is a separate question and the answer changed when the fork was created — see
[[v1dd-functional-metrics-fork]]. Pushing to `origin` is now normal; `upstream` is still
PR-only. Neither licenses committing on their behalf.
