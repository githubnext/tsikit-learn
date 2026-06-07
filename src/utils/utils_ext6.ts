/**
 * Caching and memoization utilities for sklearn computations.
 */

export class LRUCache<K, V> {
  private capacity: number;
  private cache: Map<K, V>;

  constructor(capacity = 128) {
    this.capacity = capacity;
    this.cache = new Map();
  }

  get(key: K): V | undefined {
    if (!this.cache.has(key)) return undefined;
    const value = this.cache.get(key) as V;
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }

  set(key: K, value: V): void {
    if (this.cache.has(key)) this.cache.delete(key);
    else if (this.cache.size >= this.capacity) {
      const firstKey = this.cache.keys().next().value as K;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }

  has(key: K): boolean {
    return this.cache.has(key);
  }

  clear(): void {
    this.cache.clear();
  }

  get size(): number {
    return this.cache.size;
  }
}

export function memoize<T extends (...args: unknown[]) => unknown>(fn: T, maxSize = 128): T {
  const cache = new LRUCache<string, ReturnType<T>>(maxSize);
  return ((...args: unknown[]) => {
    const key = JSON.stringify(args);
    if (cache.has(key)) return cache.get(key);
    const result = fn(...args) as ReturnType<T>;
    cache.set(key, result);
    return result;
  }) as T;
}

export class ComputationCache {
  private store: Map<string, { value: unknown; computedAt: number; expiresIn: number }>;
  private maxSize: number;

  constructor(maxSize = 256) {
    this.store = new Map();
    this.maxSize = maxSize;
  }

  set(key: string, value: unknown, expiresIn = Number.POSITIVE_INFINITY): void {
    if (this.store.size >= this.maxSize) {
      const oldest = Array.from(this.store.entries()).reduce((best, [k, v]) =>
        !best || v.computedAt < best[1].computedAt ? [k, v] as [string, typeof v] : best
      );
      if (oldest) this.store.delete(oldest[0]);
    }
    this.store.set(key, { value, computedAt: Date.now(), expiresIn });
  }

  get(key: string): unknown | undefined {
    const entry = this.store.get(key);
    if (!entry) return undefined;
    if (Date.now() - entry.computedAt > entry.expiresIn) {
      this.store.delete(key);
      return undefined;
    }
    return entry.value;
  }

  has(key: string): boolean {
    return this.get(key) !== undefined;
  }

  delete(key: string): void {
    this.store.delete(key);
  }

  clear(): void {
    this.store.clear();
  }

  get size(): number {
    return this.store.size;
  }
}

export class FittedModelCache {
  private cache: Map<string, { model: unknown; metrics: Record<string, number> }>;

  constructor() {
    this.cache = new Map();
  }

  store(key: string, model: unknown, metrics: Record<string, number> = {}): void {
    this.cache.set(key, { model, metrics });
  }

  retrieve(key: string): { model: unknown; metrics: Record<string, number> } | undefined {
    return this.cache.get(key);
  }

  listKeys(): string[] {
    return Array.from(this.cache.keys());
  }

  getBestModel(metric: string, higher = true): { key: string; model: unknown; score: number } | null {
    let best: { key: string; model: unknown; score: number } | null = null;
    for (const [key, { model, metrics }] of this.cache) {
      const score = metrics[metric] ?? (higher ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY);
      if (!best || (higher ? score > best.score : score < best.score)) {
        best = { key, model, score };
      }
    }
    return best;
  }

  clear(): void {
    this.cache.clear();
  }
}

export class PersistentHash {
  private data: Map<string, string>;

  constructor() {
    this.data = new Map();
  }

  hash(obj: unknown): string {
    const str = JSON.stringify(obj);
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) + hash) + str.charCodeAt(i);
      hash = hash & hash;
    }
    return Math.abs(hash).toString(36);
  }

  store(obj: unknown, value: string): void {
    this.data.set(this.hash(obj), value);
  }

  lookup(obj: unknown): string | undefined {
    return this.data.get(this.hash(obj));
  }
}
