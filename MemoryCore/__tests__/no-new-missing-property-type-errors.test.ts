/**
 * Guard against calling a method or reading a property that does not exist on
 * its type (TS2339). Vitest strips types without checking them, so such a call
 * only fails at runtime: `checkpoint.incrementMemoriesSincePersona` shipped this
 * way and broke every L2 run.
 *
 * MemoryCore does not type-check cleanly yet (`pnpm typecheck` reports other,
 * older errors), so the whole of `tsc` cannot gate CI. This test runs the
 * compiler over the same files as `tsconfig.json` and fails on any TS2339 not
 * already listed below. Keys are `file|property` with an occurrence count, so
 * line moves and dependency-version text in messages do not matter.
 *
 * Fixing one of the listed errors lowers its count: update the list. Never
 * raise a count or add a key to make this pass; fix the call instead.
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import { describe, expect, it } from "vitest";

const KNOWN_TS2339: Record<string, number> = {
  "src/adapters/standalone/llm-runner.ts|completionTokens": 2,
  "src/adapters/standalone/llm-runner.ts|promptTokens": 2,
  "src/core/report/otlp-backend.ts|Resource": 1,
  "src/core/skill/skill-extractor.ts|confidence": 1,
  "src/core/skill/skill-extractor.ts|file_path": 1,
  "src/core/skill/skill-extractor.ts|file_type": 1,
  "src/core/skill/skill-extractor.ts|reason": 1,
  "src/core/tdai-core.ts|cos": 3,
  "src/core/tdai-core.ts|jevGateId": 1,
  "src/core/tdai-core.ts|sceneDiff": 1,
  "src/core/tdai-core.ts|skipped": 1,
  "src/core/tdai-core.ts|stream": 1,
  "src/gateway/skill-handlers.ts|envelope": 28,
  "src/gateway/skill-handlers.ts|error": 2,
  "src/utils/clean-context-runner.ts|runEmbeddedPiAgent": 1,
  "src/utils/pipeline-manager.ts|latestCursor": 1,
  "src/utils/pipeline-manager.ts|profileScopes": 1,
  "src/utils/pipeline-manager.ts|skipped": 1,
};

const memoryCoreRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function collectMissingPropertyErrors(): { counts: Map<string, number>; locations: Map<string, string[]> } {
  const configPath = path.join(memoryCoreRoot, "tsconfig.json");
  const config = ts.readConfigFile(configPath, ts.sys.readFile);
  if (config.error) throw new Error(ts.flattenDiagnosticMessageText(config.error.messageText, "\n"));
  const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, memoryCoreRoot);
  const program = ts.createProgram(parsed.fileNames, parsed.options);

  const counts = new Map<string, number>();
  const locations = new Map<string, string[]>();
  for (const diagnostic of ts.getPreEmitDiagnostics(program)) {
    if (diagnostic.code !== 2339 || !diagnostic.file || diagnostic.start === undefined) continue;
    const file = path.relative(memoryCoreRoot, diagnostic.file.fileName).split(path.sep).join("/");
    const message = ts.flattenDiagnosticMessageText(diagnostic.messageText, " ");
    const property = /Property '([^']+)'/.exec(message)?.[1] ?? message;
    const key = `${file}|${property}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
    const { line, character } = diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start);
    locations.set(key, [...(locations.get(key) ?? []), `${file}(${line + 1},${character + 1}): ${message}`]);
  }
  return { counts, locations };
}

describe("type check", () => {
  it("adds no new 'property does not exist on type' (TS2339) errors", () => {
    const { counts, locations } = collectMissingPropertyErrors();
    const added: string[] = [];
    for (const [key, count] of counts) {
      const known = KNOWN_TS2339[key] ?? 0;
      if (count > known) {
        added.push(`${key}: ${count} (known ${known})\n    ${(locations.get(key) ?? []).join("\n    ")}`);
      }
    }
    expect(added, `New TS2339 errors:\n${added.join("\n")}`).toEqual([]);
  });
});
