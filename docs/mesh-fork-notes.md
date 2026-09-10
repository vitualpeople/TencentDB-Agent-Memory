# Mesh fork notes

This repository (`vitualpeople/TencentDB-Agent-Memory`) is a fork of
`TencentCloud/TencentDB-Agent-Memory` (MIT). It exists for one reason: the
`distributed-agent-mesh` project runs the `MemoryCore/` gateway (`tdai-gateway`,
the L0/L1/L2 memory pipeline) from a private image, and that image needs two
small fixes to the L1 idle-timer path that upstream has not made. Everything
else in the repository is unchanged upstream code.

Only `MemoryCore/` matters to the mesh. `MemoryPanel/`, `MemoryProxy/`,
`MemoryKnowledge/`, `sdk/` and `deploy/` are carried along untouched.

## Branches

- `mesh-stable` is the base. It is upstream commit `97f9465`
  (`v2.0.1-beta.2` plus one docs commit), verified byte-identical to the
  image the mesh was running before this fork existed. Fix branches are cut
  from it and merged back into it.
- `main` and `feat/server_team` track upstream and are ahead of
  `mesh-stable`. Do not branch fixes from them: the mesh adapter
  (`distributed-agent-mesh/memory-hub`) is written against the exact `/v2`
  routes and response shapes of the `mesh-stable` build.

Remotes: `origin` is this fork, `upstream` is TencentCloud. Pull requests are
opened in the fork against `mesh-stable`, never against upstream.

## The two patches

Both were measured on the running gateway. The full root-cause analysis lives
with the mesh project; the short version:

1. **Timer member scope** (`MemoryCore/src/core/state/timer-member.ts`).
   `buildPipelineTimerMember` used to scope the L1 idle-timer member only when
   both `teamId` and `agentId` were set. `resolveIsolation()` defaults
   `agentId` to `"default"` but leaves `teamId` unset for any client that
   sends no `team_id`, so the member fell back to the legacy unscoped form and
   the timer-fired task parsed both ids back as `undefined`. The executor and
   `cascadeSchedule` then read and reset the state key
   `{svc}:_:_:{session}` while `captureAtomic` counted under
   `{svc}:_:default:{session}`. After the first reset created the `_:_` entry
   with count 0, every timer-fired L1 was skipped as "already processed".
   The member is now scoped when either id is set; the missing half is
   written as an empty field and parses back to `undefined`.

2. **Count released, not zeroed** (`MemoryCore/src/services/pipeline-worker.ts`,
   `MemoryCore/src/gateway/server.ts`). `cascadeSchedule` used to set
   `conversation_count: 0` after every L1 task, including tasks the executor
   had skipped. A message counted while an L1 was running was therefore wiped
   when that L1 completed, and the message's own timer fire was then
   deduplicated against the false zero. Now the executor's timer dedup
   (`claimTimerL1Count`) marks a skipped task so `cascadeSchedule` leaves the
   state alone, and records the count a timer-fired run actually consumed so
   `cascadeSchedule` subtracts that instead of assigning 0. Threshold-fired
   tasks need no reset at all: `captureAtomic` already zeroes the count when
   it enqueues them. `flush` tasks keep the old reset.

Tests for both live in `MemoryCore/__tests__/`. They fail on `mesh-stable` and
pass on the fix branch.

## Image and tags

`.github/workflows/docker-publish.yml` builds `MemoryCore/Dockerfile` and pushes
`chauit/tencentdb-agent-memory:<tag>` when a `v*` tag is pushed, or on manual
dispatch. It needs the repository secrets `DOCKERHUB_USERNAME` and
`DOCKERHUB_TOKEN` (the same names `distributed-agent-mesh` uses) and fails
before building when they are missing.

Tag scheme: `v<upstream version>-mesh.<n>`, for example `v2.0.1-mesh.1` for the
first image built from `mesh-stable` plus these two patches. The `-mesh.`
suffix keeps fork tags distinct from upstream tags (`v2.0.1`,
`v2.0.2-beta.1`), and the workflow refuses to publish any tag without it, so an
upstream tag synced into the fork can never overwrite the mesh image.

The mesh does not pull by tag. `distributed-agent-mesh/stacks/agent-state-memory.yml`
pins `chauit/tencentdb-agent-memory@sha256:<digest>`; the workflow prints the
digest of each published image in its step summary, and bumping that pin is a
change in the mesh repository, not here.

## Contract

The mesh adapter is written against this build's `/v2` routes. A change to any
route, request shape or response shape is a cross-repository contract change
and is coordinated with `distributed-agent-mesh` first; it is never made here
in isolation.
