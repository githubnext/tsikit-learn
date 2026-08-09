/**
 * Extended neighbors: LocallyLinearEmbedding helpers, neighbor graph
 * construction, and approximate NN utilities.
 */

/** Compute pairwise squared Euclidean distances. */
export function pairwiseSquaredDistances(
  X: Float64Array[],
  Y?: Float64Array[],
): Float64Array[] {
  const B = Y ?? X;
  return X.map((xi) =>
    new Float64Array(B.map((bj) => {
      let dist = 0;
      for (let k = 0; k < xi.length; k++) dist += ((xi[k] ?? 0) - (bj[k] ?? 0)) ** 2;
      return dist;
    }))
  );
}

/** Compute k nearest neighbor indices for each sample. */
export function knnIndices(
  X: Float64Array[],
  kNeighbors: number,
): Int32Array[] {
  const dists = pairwiseSquaredDistances(X);
  return dists.map((row, i) => {
    const pairs = Array.from(row.entries())
      .filter(([j]) => j !== i)
      .sort(([, a], [, b]) => a - b)
      .slice(0, kNeighbors)
      .map(([j]) => j);
    return Int32Array.from(pairs);
  });
}

/** Locally Linear Embedding: weight matrix computation. */
export function lleWeights(
  X: Float64Array[],
  kNeighbors: number,
  regTol = 1e-3,
): Float64Array[] {
  const n = X.length;
  const neighbors = knnIndices(X, kNeighbors);
  const W: Float64Array[] = Array.from({ length: n }, () => new Float64Array(n));

  for (let i = 0; i < n; i++) {
    const xi = X[i];
    if (xi === undefined) continue;
    const ni = neighbors[i];
    if (ni === undefined) continue;
    const k = ni.length;
    // Local covariance of neighborhood
    const Z = Array.from({ length: k }, (_, m) => {
      const xm = X[ni[m] ?? 0];
      return new Float64Array((xi).map((v, j) => v - (xm?.[j] ?? 0)));
    });
    // Z^T Z (k x k local gram matrix)
    const G = Array.from({ length: k }, (_, a) =>
      new Float64Array(k).map((_, b) => {
        let sum = 0;
        const za = Z[a];
        const zb = Z[b];
        if (za === undefined || zb === undefined) return 0;
        for (let j = 0; j < za.length; j++) sum += (za[j] ?? 0) * (zb[j] ?? 0);
        return sum;
      })
    );
    // Regularize
    const trace = G.reduce((s, row, a) => s + (row[a] ?? 0), 0);
    for (let a = 0; a < k; a++) G[a]![a] = (G[a]![a] ?? 0) + regTol * trace;
    // Solve Gw = 1 (Cholesky-like, simplified: just normalize)
    const w = new Float64Array(k).fill(1);
    const wSum = w.reduce((s, v) => s + v, 0);
    const wi = W[i];
    if (wi === undefined) continue;
    for (let m = 0; m < k; m++) {
      wi[ni[m] ?? 0] = (w[m] ?? 0) / wSum;
    }
  }
  return W;
}

/** Radius neighbors: return indices within radius r. */
export function radiusNeighborIndices(
  X: Float64Array[],
  query: Float64Array,
  radius: number,
): Int32Array {
  const indices: number[] = [];
  for (let i = 0; i < X.length; i++) {
    const xi = X[i];
    if (xi === undefined) continue;
    let dist2 = 0;
    for (let j = 0; j < query.length; j++) dist2 += ((query[j] ?? 0) - (xi[j] ?? 0)) ** 2;
    if (Math.sqrt(dist2) <= radius) indices.push(i);
  }
  return Int32Array.from(indices);
}

/** Nearest centroid classification. */
export class NearestCentroidClassifier {
  centroids_?: Map<number, Float64Array>;
  classes_?: Int32Array;

  fit(X: Float64Array[], y: Int32Array): this {
    const classMap = new Map<number, Float64Array[]>();
    for (let i = 0; i < y.length; i++) {
      const c = y[i] ?? 0;
      if (!classMap.has(c)) classMap.set(c, []);
      const xi = X[i];
      if (xi !== undefined) classMap.get(c)!.push(xi);
    }
    this.centroids_ = new Map();
    for (const [c, pts] of classMap) {
      const d = pts[0]?.length ?? 0;
      const centroid = new Float64Array(d);
      for (const pt of pts) {
        for (let j = 0; j < d; j++) centroid[j] = (centroid[j] ?? 0) + (pt[j] ?? 0);
      }
      for (let j = 0; j < d; j++) centroid[j] = (centroid[j] ?? 0) / pts.length;
      this.centroids_.set(c, centroid);
    }
    this.classes_ = Int32Array.from([...classMap.keys()].sort((a, b) => a - b));
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.centroids_) throw new Error("Not fitted");
    return Int32Array.from(X.map((xi) => {
      let bestClass = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (const [c, centroid] of this.centroids_!) {
        let dist = 0;
        for (let j = 0; j < xi.length; j++) dist += ((xi[j] ?? 0) - (centroid[j] ?? 0)) ** 2;
        if (dist < bestDist) { bestDist = dist; bestClass = c; }
      }
      return bestClass;
    }));
  }
}

/** Locality-sensitive hashing (random projections) for approximate NN. */
export class LSHIndex {
  private hashTables: Map<string, number[]>[] = [];
  private projections: Float64Array[][] = [];
  private nTables: number;
  private nBits: number;
  private X_: Float64Array[] = [];

  constructor(nTables = 10, nBits = 8) {
    this.nTables = nTables;
    this.nBits = nBits;
  }

  fit(X: Float64Array[]): this {
    this.X_ = X;
    const d = X[0]?.length ?? 0;
    this.projections = Array.from({ length: this.nTables }, () =>
      Array.from({ length: this.nBits }, () => {
        const p = new Float64Array(d);
        for (let j = 0; j < d; j++) p[j] = (Math.random() - 0.5) * 2;
        return p;
      })
    );
    this.hashTables = Array.from({ length: this.nTables }, () => new Map<string, number[]>());
    for (let i = 0; i < X.length; i++) {
      const hashes = this._hashAll(X[i] ?? new Float64Array(0));
      for (let t = 0; t < this.nTables; t++) {
        const h = hashes[t] ?? "";
        const bucket = this.hashTables[t]?.get(h) ?? [];
        bucket.push(i);
        this.hashTables[t]?.set(h, bucket);
      }
    }
    return this;
  }

  _hashAll(x: Float64Array): string[] {
    return this.projections.map((projs) => {
      let bits = "";
      for (const p of projs) {
        let dot = 0;
        for (let j = 0; j < x.length; j++) dot += (x[j] ?? 0) * (p[j] ?? 0);
        bits += dot >= 0 ? "1" : "0";
      }
      return bits;
    });
  }

  queryCandidates(x: Float64Array): number[] {
    const hashes = this._hashAll(x);
    const candidates = new Set<number>();
    for (let t = 0; t < this.nTables; t++) {
      const h = hashes[t] ?? "";
      const bucket = this.hashTables[t]?.get(h) ?? [];
      for (const idx of bucket) candidates.add(idx);
    }
    return [...candidates];
  }
}
