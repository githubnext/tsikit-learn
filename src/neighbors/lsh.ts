/**
 * Locality-Sensitive Hashing and MinHash for nearest neighbor search.
 */

export class MinHash {
  private hashFunctions: Array<{ a: number; b: number }>;
  private readonly prime = 2147483647;

  constructor(private readonly nHashFunctions = 128) {
    const rng = this._seededRng(42);
    this.hashFunctions = Array.from({ length: nHashFunctions }, () => ({
      a: Math.floor(rng() * (this.prime - 1)) + 1,
      b: Math.floor(rng() * (this.prime - 1)),
    }));
  }

  signature(set: number[]): Int32Array {
    const sig = new Int32Array(this.nHashFunctions).fill(2147483647);
    for (const elem of set) {
      for (let j = 0; j < this.nHashFunctions; j++) {
        const { a, b } = this.hashFunctions[j]!;
        const h = (a * elem + b) % this.prime;
        if (h < (sig[j] ?? 2147483647)) sig[j] = h;
      }
    }
    return sig;
  }

  jaccardEstimate(sig1: Int32Array, sig2: Int32Array): number {
    let matches = 0;
    const n = Math.min(sig1.length, sig2.length);
    for (let i = 0; i < n; i++) if (sig1[i] === sig2[i]) matches++;
    return matches / Math.max(n, 1);
  }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    };
  }
}

export class LSHIndex {
  private tables: Map<string, number[]>[] = [];
  private signatures: Int32Array[] = [];

  constructor(
    private readonly nHashFunctions = 128,
    private readonly nBands = 16,
  ) {}

  fit(X: Float64Array[]): this {
    const n = X.length;
    const nF = X[0]?.length ?? 1;
    const mh = new MinHash(this.nHashFunctions);
    // Convert float vectors to sets of (feature, bucket) pairs
    this.signatures = X.map((x) => {
      const set: number[] = [];
      for (let f = 0; f < nF; f++) {
        const bucket = Math.floor((x[f] ?? 0) * 10) + f * 10000;
        set.push(bucket);
      }
      return mh.signature(set);
    });
    // Build hash tables (banding technique)
    const rowsPerBand = Math.floor(this.nHashFunctions / this.nBands);
    this.tables = Array.from(
      { length: this.nBands },
      () => new Map<string, number[]>(),
    );
    for (let i = 0; i < n; i++) {
      for (let band = 0; band < this.nBands; band++) {
        const start = band * rowsPerBand;
        const bandSig = this.signatures[i]!.slice(start, start + rowsPerBand);
        const key = bandSig.join(",");
        const t = this.tables[band]!;
        const bucket = t.get(key) ?? [];
        bucket.push(i);
        t.set(key, bucket);
      }
    }
    return this;
  }

  queryCandidates(query: Float64Array): Set<number> {
    const nF = query.length;
    const mh = new MinHash(this.nHashFunctions);
    const set: number[] = [];
    for (let f = 0; f < nF; f++) {
      const bucket = Math.floor((query[f] ?? 0) * 10) + f * 10000;
      set.push(bucket);
    }
    const sig = mh.signature(set);
    const rowsPerBand = Math.floor(this.nHashFunctions / this.nBands);
    const candidates = new Set<number>();
    for (let band = 0; band < this.nBands; band++) {
      const bandSig = sig.slice(band * rowsPerBand, (band + 1) * rowsPerBand);
      const key = bandSig.join(",");
      for (const idx of this.tables[band]?.get(key) ?? []) candidates.add(idx);
    }
    return candidates;
  }
}

export class LSHNearestNeighbors {
  private index: LSHIndex | null = null;
  private X_: Float64Array[] = [];

  constructor(
    private readonly nNeighbors = 5,
    private readonly nHashFunctions = 128,
    private readonly nBands = 16,
  ) {}

  fit(X: Float64Array[]): this {
    this.X_ = X;
    this.index = new LSHIndex(this.nHashFunctions, this.nBands).fit(X);
    return this;
  }

  kneighbors(X: Float64Array[]): {
    indices: Int32Array[];
    distances: Float64Array[];
  } {
    if (!this.index) throw new Error("Not fitted");
    const indices: Int32Array[] = [];
    const distances: Float64Array[] = [];
    for (const query of X) {
      const candidates = this.index.queryCandidates(query);
      if (candidates.size === 0) {
        // Fall back to all points
        for (let i = 0; i < this.X_.length; i++) candidates.add(i);
      }
      const scored = [...candidates]
        .map((i) => {
          let d = 0;
          const xi = this.X_[i]!;
          for (let f = 0; f < query.length; f++)
            d += ((query[f] ?? 0) - (xi[f] ?? 0)) ** 2;
          return { i, d: Math.sqrt(d) };
        })
        .sort((a, b) => a.d - b.d)
        .slice(0, this.nNeighbors);
      indices.push(new Int32Array(scored.map((s) => s.i)));
      distances.push(new Float64Array(scored.map((s) => s.d)));
    }
    return { indices, distances };
  }
}
