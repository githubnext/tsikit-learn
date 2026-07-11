/**
 * Parallel execution utilities.
 * Mirrors sklearn.utils.parallel — Parallel, delayed, cpu_count.
 */

export type DelayedCall<T> = {
  fn: (...args: unknown[]) => T;
  args: unknown[];
};

/**
 * Wraps a function for deferred execution in Parallel.
 * Usage: delayed(fn)(arg1, arg2)
 */
export function delayed<T>(
  fn: (...args: unknown[]) => T,
): (...args: unknown[]) => DelayedCall<T> {
  return (...args: unknown[]): DelayedCall<T> => ({ fn, args });
}

export interface ParallelOptions {
  nJobs?: number;
  prefer?: "threads" | "processes";
  verbose?: number;
  returnAs?: "list" | "generator";
}

/**
 * Simple sequential parallel executor (TypeScript single-threaded model).
 * Runs all delayed calls synchronously, matching sklearn's Parallel interface.
 */
export class Parallel<T> {
  private nJobs: number;
  private verbose: number;

  constructor(options: ParallelOptions = {}) {
    this.nJobs = options.nJobs ?? 1;
    this.verbose = options.verbose ?? 0;
  }

  /**
   * Execute all delayed calls and return results.
   */
  call(calls: DelayedCall<T>[]): T[] {
    if (this.verbose > 0) {
      console.log(
        `Parallel(n_jobs=${this.nJobs}): processing ${calls.length} tasks`,
      );
    }
    return calls.map((c) => c.fn(...c.args));
  }
}

/**
 * Returns the number of logical CPUs available (always 1 in browser/Bun).
 */
export function cpuCount(): number {
  if (typeof navigator !== "undefined" && navigator.hardwareConcurrency) {
    return navigator.hardwareConcurrency;
  }
  return 1;
}

/**
 * Effective number of jobs: -1 → cpuCount(), 0 → 1, n > 0 → n
 */
export function effectiveNJobs(nJobs: number): number {
  if (nJobs === -1) return cpuCount();
  if (nJobs <= 0) return 1;
  return nJobs;
}

/**
 * Parallel map: apply fn to each item in items, using nJobs workers.
 */
export function parallelMap<In, Out>(
  items: In[],
  fn: (item: In) => Out,
  _nJobs = 1,
): Out[] {
  return items.map(fn);
}
