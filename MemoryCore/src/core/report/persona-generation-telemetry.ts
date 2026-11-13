/**
 * Process-wide counters for L3 persona generations, read by /health.
 *
 * Same shape of reasoning as ProviderHealthSink: the L3 runner is built per
 * task, so a counter on one runner would lose the others. createL3Runner
 * records into the default sink below unless a test injects its own, and it
 * records at exactly one point per generation (its `finally`).
 *
 * A "generation" is one that called the LLM runner at least once; the early
 * "no changed scene" skip never reaches the runner and is not counted.
 */

/** Counters as /health reports them. */
export interface PersonaGenerationSnapshot {
  /** Generations that called the LLM runner at least once. */
  personaGenerations: number;
  /** Generations that sent the second, "you must call write/edit" call. */
  personaWriteNudges: number;
  /** Nudged generations whose second call wrote persona.md and that then succeeded. */
  personaWriteNudgeSuccesses: number;
  /** Successful generations whose persona body is still over the configured limit. */
  personaOverLimit: number;
}

/** What one generation contributes to the counters. */
export interface PersonaGenerationRecord {
  invokedRunner: boolean;
  nudged: boolean;
  nudgeSucceeded: boolean;
  overLimit: boolean;
}

export class PersonaGenerationSink {
  private counters: PersonaGenerationSnapshot = {
    personaGenerations: 0,
    personaWriteNudges: 0,
    personaWriteNudgeSuccesses: 0,
    personaOverLimit: 0,
  };

  record(r: PersonaGenerationRecord): void {
    if (!r.invokedRunner) return;
    this.counters.personaGenerations++;
    if (r.nudged) this.counters.personaWriteNudges++;
    if (r.nudged && r.nudgeSucceeded) this.counters.personaWriteNudgeSuccesses++;
    if (r.overLimit) this.counters.personaOverLimit++;
  }

  snapshot(): PersonaGenerationSnapshot {
    return { ...this.counters };
  }
}

/** Default sink for every L3 runner in this process; /health reads it. */
export const personaGenerationTelemetry = new PersonaGenerationSink();
