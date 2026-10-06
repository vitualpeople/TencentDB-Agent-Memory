#!/bin/sh
# Runs as root: clone the repo on the first start, then drop to the `worker`
# user for everything else.
#
#   (no args)      claude --dangerously-skip-permissions on $TASK_BRANCH
#   test           the MemoryCore gate in the clone: pnpm -C MemoryCore install
#                  && pnpm -C MemoryCore test && pnpm -C MemoryCore build; no
#                  Claude, no LLM call. The install resolves the dependency
#                  tree from the registry (MemoryCore has no lockfile) and
#                  writes MemoryCore/pnpm-lock.yaml, which .gitignore covers.
#   serve          headless Orca runtime on $ORCA_SERVE_PORT for the host to pair
#                  with; stdout is its one JSON ready line, stderr goes to
#                  /tmp/orca-serve.log. Everything it spawns runs as worker.
#   <cmd> [args]   run <cmd> as worker in the prepared clone (debugging, checks)
#
# A container filesystem survives `docker restart`, so this script can run
# again on the /work a previous start already cloned into. The clone and the
# task-branch checkout are first-start work only: cloning again is fatal
# (/work is not empty), and `checkout -B origin/mesh-stable` would reset the
# branch over whatever the last task left in the tree. Resetting /work between
# tasks is the host's job, not this script's.
#
# The base is mesh-stable, the fork's GitHub default branch (the deployed
# gateway); main and feat/server_team track upstream.
#
# The token is only ever read from the GH_TOKEN environment variable, by gh
# acting as git's credential helper; it is never written to .git/config,
# a remote URL, or the image.
set -eu

WORK=/work
BASE_REF=origin/mesh-stable

as_worker() {
    setpriv --reuid=worker --regid=worker --init-groups \
            --inh-caps=-all --bounding-set=-all \
            env HOME=/home/worker USER=worker "$@"
}

: "${GH_TOKEN:?GH_TOKEN is required to clone the repo (pass it through the environment)}"
if [ "$#" -eq 0 ] && [ -z "${TASK_BRANCH:-}" ]; then
    echo "TASK_BRANCH is required to start Claude (set it in .env.cc or with -e)" >&2
    exit 2
fi

if [ -d "$WORK/.git" ]; then
    echo "cc-worker-entrypoint: /work already holds a clone; reusing it" >&2
else
    as_worker git clone --quiet "$REPO_URL" "$WORK"
    # Only a clone that was just made gets the branch reset onto
    # origin/mesh-stable; that is the one state where the tree is clean by
    # construction.
    if [ -n "${TASK_BRANCH:-}" ]; then
        as_worker git -C "$WORK" checkout --quiet -B "$TASK_BRANCH" "$BASE_REF"
    fi
fi

if [ "$#" -eq 0 ]; then
    set -- claude --dangerously-skip-permissions
elif [ "$1" = "test" ] && [ "$#" -eq 1 ]; then
    set -- sh -c 'pnpm -C MemoryCore install && pnpm -C MemoryCore test && pnpm -C MemoryCore build'
elif [ "$1" = "serve" ] && [ "$#" -eq 1 ]; then
    # Electron refuses root, and Chromium's sandbox needs user namespaces that
    # Docker's default seccomp profile blocks; `serve` rejects --no-sandbox,
    # so the sandbox is disabled through the environment instead. The pairing
    # url in the stdout line is a secret: the host reads it, nothing logs it.
    cd "$WORK"
    exec setpriv --reuid=worker --regid=worker --init-groups \
                 --inh-caps=-all --bounding-set=-all \
                 env HOME=/home/worker USER=worker ELECTRON_DISABLE_SANDBOX=1 \
                 orca-ide serve --port "${ORCA_SERVE_PORT:-17777}" \
                     --pairing-address "${ORCA_PAIRING_ADDRESS:-ws://localhost:17777}" --json \
                 2>/tmp/orca-serve.log
fi
cd "$WORK"
exec setpriv --reuid=worker --regid=worker --init-groups \
             --inh-caps=-all --bounding-set=-all \
             env HOME=/home/worker USER=worker "$@"
