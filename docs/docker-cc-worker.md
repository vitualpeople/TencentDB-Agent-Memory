# Docker Claude Code worker (`cc-worker`)

A container that runs the Claude Code CLI against this fork with a custom LLM
endpoint. A new container clones the repo from GitHub, checks out a task branch
from `origin/mesh-stable`, and starts the TUI; a restarted container reuses the
clone it already has. Claude does the task, runs the gate, pushes, and opens
the PR with `gh`. Other modes run the MemoryCore gate or a headless Orca
runtime the host pairs with.

This is a development worker only. The production image is
`MemoryCore/Dockerfile` (tdai-gateway, built from `MemoryCore/`); it and its
ignore file are unrelated and unchanged.

Files: `docker/Dockerfile.cc-worker`, `docker/Dockerfile.cc-worker.dockerignore`,
`docker/entrypoint.sh`, `docker-compose.cc.yml`, `.env.cc.example`.

## Base branch

The base is `mesh-stable`, the fork's GitHub default branch and the source of
the deployed gateway image. A fresh clone's `origin/HEAD` points at it, so
`git checkout -B <branch> origin/HEAD` and the entrypoint's
`origin/mesh-stable` give the same start. `main` and `feat/server_team` track
upstream; PRs from this worker go to `mesh-stable`.

## Why a clone and not a mount

An Orca/git worktree's `.git` is a file pointing at a `C:/...` gitdir that
Linux git cannot resolve, so the host checkout is never mounted. The container
clones over HTTPS instead; the token is read from `GH_TOKEN` by `gh` acting as
git's credential helper and is never written to `.git/config`, a remote URL, or
the image.

## Build

```sh
docker build -f docker/Dockerfile.cc-worker -t tencentdb-agent-memory-cc-worker .
# or: docker compose -f docker-compose.cc.yml build cc-worker
```

The build context is filtered by `docker/Dockerfile.cc-worker.dockerignore`,
an allow-list holding only `docker/entrypoint.sh`, because that is all this
build needs: `.env`, `.env.cc`, `node_modules` and the sources never reach the
build.

The image is `node:22-bookworm-slim` (Node 22, the major
`MemoryCore/Dockerfile` and upstream CI use) plus pnpm (pinned by the
`PNPM_VERSION` build arg), Claude Code (`CLAUDE_CODE_VERSION`), Orca
(`ORCA_VERSION`=1.4.220, the host's version; the deb is checked against
`ORCA_DEB_SHA256`, the digest GitHub publishes for the release asset), `git`
and `gh`. node, pnpm and claude live in `/usr/local/bin`, which Debian's
`/etc/profile` keeps, so login shells (Orca agent terminals) find them without
a profile fix.

No dependencies are baked in. `MemoryCore/` has no lockfile, so the gate
resolves the tree from the registry on every install, the same way upstream CI
does. No C/C++ toolchain is installed: pnpm 10 skips dependency build scripts
unless they are approved (the install lists them as "Ignored build scripts"),
so nothing compiles during install, and the gate passes without them.

The node image's `node` user (uid 1000) is replaced by `worker` (uid 1000).
The image bakes the answers to the onboarding, folder-trust and
bypass-permissions dialogs for `worker`, so
`claude --dangerously-skip-permissions` starts at an idle prompt.

### Rebuild rule

**Rebuild the image whenever anything under `docker/` changes on
`mesh-stable`**, or to move a pinned version, then `docker rm -f` the standing
worker container before starting it again (an existing container restarts on
its old image).

## Run

```sh
cp .env.cc.example .env.cc     # fill in values; .env.cc is gitignored

docker compose -f docker-compose.cc.yml run --rm cc-worker          # Claude TUI on $TASK_BRANCH
docker compose -f docker-compose.cc.yml run --rm cc-worker test     # MemoryCore gate in a fresh clone, no LLM call
docker compose -f docker-compose.cc.yml run --rm cc-worker bash     # shell in the prepared clone
docker compose -f docker-compose.cc.yml up -d cc-worker-serve       # headless Orca runtime on 127.0.0.1:17784
```

`.env.cc` is optional: the token can come from the shell instead, without
touching disk:

```sh
GH_TOKEN="$(gh auth token)" docker compose -f docker-compose.cc.yml run --rm -e GH_TOKEN -e TASK_BRANCH=mesh-stable cc-worker test
```

From Git Bash, prefix with `MSYS_NO_PATHCONV=1` when an argument is a
container path such as `/work`.

## Modes (entrypoint arguments)

| Argument | Runs (as `worker`, cwd `/work`) |
|---|---|
| none | `claude --dangerously-skip-permissions`; `TASK_BRANCH` required |
| `test` | `pnpm -C MemoryCore install && pnpm -C MemoryCore test && pnpm -C MemoryCore build` |
| `serve` | `orca-ide serve` (see below) |
| anything else | that command, unchanged |

The entrypoint starts as root, clones on the first start only, and runs
`git checkout -B $TASK_BRANCH origin/mesh-stable` only right after that clone.
It then drops to `worker` (uid 1000) with `setpriv` and empty capability sets.
Resetting `/work` between tasks is the host's job.

## Environment

| Variable | Used by | Notes |
|---|---|---|
| `GH_TOKEN` | clone, push, `gh pr create` | required |
| `TASK_BRANCH` | entrypoint | required for the Claude mode; applied on a first start only |
| `ANTHROPIC_BASE_URL` | Claude Code | no `/v1` suffix |
| `ANTHROPIC_AUTH_TOKEN` | Claude Code | not `ANTHROPIC_API_KEY`, which raises a blocking dialog |
| `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL` | Claude Code | model ids the endpoint serves |
| `GIT_AUTHOR_NAME`/`_EMAIL`, `GIT_COMMITTER_NAME`/`_EMAIL` | git | commit identity |
| `REPO_URL` | entrypoint | defaults to `https://github.com/vitualpeople/TencentDB-Agent-Memory.git` |
| `ORCA_SERVE_PORT` | entrypoint (`serve`) | port the runtime binds in the container; default `17777` |
| `ORCA_PAIRING_ADDRESS` | entrypoint (`serve`) | endpoint written into the pairing code; default `ws://localhost:17777` |

The gateway's own variables (LLM and embedding keys, TencentDB VDB, COS) are
not set in the image and do not belong in `.env.cc`. The test suite needs none
of them. Do not put production values there.

## Gate inside the container

```sh
pnpm -C MemoryCore install && pnpm -C MemoryCore test && pnpm -C MemoryCore build   # entrypoint arg `test`
```

Run from the repo root. PRs into `mesh-stable` get no GitHub CI
(`.github/workflows/pr-ci.yml` only runs for PRs into `main`), so this is the
only gate. The baseline (test counts, build result, and the commit they were
measured at) lives in the ORCA project registry, not here, because it changes
with every merge. A task's gate is no new test failures and no new build
errors against that baseline.

The install writes `MemoryCore/pnpm-lock.yaml`; `.gitignore` covers it, and it
is never committed.

## Serve mode (headless Orca runtime)

`serve` clones (or reuses the clone) like the other modes, then runs, as
`worker` with cwd `/work`:

```sh
orca-ide serve --port "${ORCA_SERVE_PORT:-17777}" --pairing-address "${ORCA_PAIRING_ADDRESS:-ws://localhost:17777}" --json
```

Compose publishes it on `127.0.0.1:17784` (17777-17783 are the other repos'
workers). The container's stdout is one JSON object,
`"type": "orca_server_ready"`, with `advertisedEndpoint` and `pairing.url`;
the runtime's stderr goes to `/tmp/orca-serve.log` inside the container so
`docker logs` stays parseable.

**The pairing code is a secret.** Both `pairing.url` and `pairing.webClientUrl`
carry it. Read it into a variable, never echo it, log it, or paste it into an
issue or PR. A new container is a new runtime: remove the old Orca environment
and pair again with the new code.

Electron refuses root and Chromium's sandbox needs user namespaces that
Docker's default seccomp profile blocks, so the entrypoint sets
`ELECTRON_DISABLE_SANDBOX=1`. The isolation that remains is the container
boundary (non-root `worker`, no capabilities, `no-new-privileges`, loopback-only
port).

## How the seat drives it

The coordinator seat (obsidian-second-brain) drives this image with its
`scripts/docker_worker_dispatch.py --image tencentdb-agent-memory-cc-worker`
(added to every `--up`/`--attach` call): it starts the `serve` mode, pairs
with the runtime, resets `/work` to the task branch from `origin/HEAD`
(`mesh-stable`), and starts a Claude worker inside the container that reports
back with `orca orchestration send ... worker_done`. In the TUI and `test`
modes there is no Orca runtime, so the in-container Claude cannot report back;
use `serve` for supervised work.
