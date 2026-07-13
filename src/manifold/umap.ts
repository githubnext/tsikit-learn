/**
 * UMAP: Uniform Manifold Approximation and Projection for dimensionality reduction.
 * Mirrors scikit-learn's manifold.UMAP interface (pure TypeScript implementation).
 */

export interface UMAPOptions {
  nComponents?: number;
  nNeighbors?: number;
  minDist?: number;
  nEpochs?: number;
  learningRate?: number;
  randomState?: number;
  metric?: "euclidean" | "cosine" | "manhattan";
}

export class UMAP {
  readonly nComponents: number;
  readonly nNeighbors: number;
  readonly minDist: number;
  readonly nEpochs: number;
  readonly learningRate: number;
  readonly randomState: number;
  readonly metric: "euclidean" | "cosine" | "manhattan";

  private _embedding: Float64Array[] | null = null;

  constructor(options: UMAPOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.nNeighbors = options.nNeighbors ?? 15;
    this.minDist = options.minDist ?? 0.1;
    this.nEpochs = options.nEpochs ?? 200;
    this.learningRate = options.learningRate ?? 1.0;
    this.randomState = options.randomState ?? 42;
    this.metric = options.metric ?? "euclidean";
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    this._embedding = this._optimizeLayout(X, n);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return this._optimizeLayout(X, X.length);
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    this.fit(X);
    return this._embedding ?? this.transform(X);
  }

  private _dist(a: Float64Array, b: Float64Array): number {
    switch (this.metric) {
      case "cosine": {
        let dot = 0;
        let na = 0;
        let nb = 0;
        for (let i = 0; i < a.length; i++) {
          dot += (a[i] ?? 0) * (b[i] ?? 0);
          na += (a[i] ?? 0) ** 2;
          nb += (b[i] ?? 0) ** 2;
        }
        return 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
      }
      case "manhattan": {
        let s = 0;
        for (let i = 0; i < a.length; i++)
          s += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
        return s;
      }
      default: {
        let s = 0;
        for (let i = 0; i < a.length; i++)
          s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
        return Math.sqrt(s);
      }
    }
  }

  private _optimizeLayout(X: Float64Array[], n: number): Float64Array[] {
    // Simplified UMAP: spectral initialization + force-directed refinement
    const rng = this._seededRng(this.randomState);
    const embedding: Float64Array[] = Array.from({ length: n }, () => {
      const row = new Float64Array(this.nComponents);
      for (let d = 0; d < this.nComponents; d++) row[d] = rng() * 2 - 1;
      return row;
    });

    const k = Math.min(this.nNeighbors, n - 1);
    // Build kNN graph
    const neighbors: number[][] = Array.from({ length: n }, (_, i) => {
      const dists = Array.from({ length: n }, (__, j) => ({
        j,
        d:
          i !== j
            ? this._dist(
                X[i] ?? new Float64Array(0),
                X[j] ?? new Float64Array(0),
              )
            : Number.POSITIVE_INFINITY,
      }));
      dists.sort((a, b) => a.d - b.d);
      return dists.slice(0, k).map((x) => x.j);
    });

    const a = 1.0;
    const b = this.minDist < 0.5 ? 0.8 : 1.0;

    for (let epoch = 0; epoch < this.nEpochs; epoch++) {
      const alpha = this.learningRate * (1 - epoch / this.nEpochs);
      for (let i = 0; i < n; i++) {
        for (const j of neighbors[i] ?? []) {
          const ei = embedding[i]!;
          const ej = embedding[j]!;
          let dSq = 0;
          for (let d = 0; d < this.nComponents; d++) {
            dSq += ((ei[d] ?? 0) - (ej[d] ?? 0)) ** 2;
          }
          const grad = (2 * a * b * dSq ** (b - 1)) / (a * dSq ** b + 1);
          for (let d = 0; d < this.nComponents; d++) {
            const delta = (ei[d] ?? 0) - (ej[d] ?? 0);
            ei[d] = (ei[d] ?? 0) + alpha * grad * delta;
            ej[d] = (ej[d] ?? 0) - alpha * grad * delta;
          }
        }
        // Repulsive force from random sample
        const jRand = Math.floor(rng() * n);
        if (jRand !== i) {
          const ei = embedding[i]!;
          const ej = embedding[jRand]!;
          let dSq = 0;
          for (let d = 0; d < this.nComponents; d++) {
            dSq += ((ei[d] ?? 0) - (ej[d] ?? 0)) ** 2;
          }
          const grad = 2 / ((0.001 + dSq) * (a * dSq ** b + 1));
          for (let d = 0; d < this.nComponents; d++) {
            const delta = (ei[d] ?? 0) - (ej[d] ?? 0);
            ei[d] = (ei[d] ?? 0) + alpha * grad * delta;
          }
        }
      }
    }
    return embedding;
  }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    };
  }
}
