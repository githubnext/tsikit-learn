/**
 * SpectralClustering, MeanShift, Birch, and OPTICS clustering.
 * Mirrors sklearn.cluster SpectralClustering, MeanShift, Birch, OPTICS.
 */

import { NotFittedError } from "../exceptions.js";

// ─── SpectralClustering ───────────────────────────────────────────────────────

export interface SpectralClusteringOptions {
  nClusters?: number;
  nInit?: number;
  gamma?: number;
  affinityType?: "rbf" | "nearest_neighbors";
  nNeighbors?: number;
  randomState?: number;
}

function rbfKernel(a: Float64Array, b: Float64Array, gamma: number): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    d += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  }
  return Math.exp(-gamma * d);
}

function computeAffinityMatrix(
  X: Float64Array[],
  gamma: number,
): Float64Array[] {
  const n = X.length;
  return X.map((xi, i) =>
    Float64Array.from(X, (xj, j) => {
      if (i === j) return 0;
      return rbfKernel(xi as Float64Array, xj as Float64Array, gamma);
    }),
  );
}

function symmetricNormalizedLaplacian(W: Float64Array[]): Float64Array[] {
  const n = W.length;
  const D = W.map((row) => row.reduce((s, v) => s + v, 0));
  const Dinvhalf = D.map((d) => (d > 0 ? 1 / Math.sqrt(d) : 0));
  return W.map((row, i) =>
    Float64Array.from(row, (w, j) => (Dinvhalf[i] ?? 0) * w * (Dinvhalf[j] ?? 0)),
  );
}

function powerIterationEigenvectors(
  L: Float64Array[],
  k: number,
  maxIter = 300,
): Float64Array[] {
  const n = L.length;
  const rng = { seed: 42 };
  const rand = () => {
    rng.seed = (rng.seed * 1664525 + 1013904223) & 0xffffffff;
    return (rng.seed >>> 0) / 0xffffffff;
  };
  // Initialize random vectors
  const vecs: Float64Array[] = Array.from({ length: k }, () =>
    Float64Array.from({ length: n }, () => rand() - 0.5),
  );

  for (let iter = 0; iter < maxIter; iter++) {
    // Orthogonalize and normalize via QR (Gram-Schmidt)
    for (let col = 0; col < k; col++) {
      const v = vecs[col] as Float64Array;
      // Multiply: v = L @ v
      const Lv = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const row = L[i] as Float64Array;
        let s = 0;
        for (let j = 0; j < n; j++) s += (row[j] ?? 0) * (v[j] ?? 0);
        Lv[i] = s;
      }
      // Subtract projections of previous vectors
      for (let prev = 0; prev < col; prev++) {
        const u = vecs[prev] as Float64Array;
        let dot = 0;
        for (let i = 0; i < n; i++) dot += (Lv[i] ?? 0) * (u[i] ?? 0);
        for (let i = 0; i < n; i++) Lv[i]! -= dot * (u[i] ?? 0);
      }
      // Normalize
      let norm = 0;
      for (let i = 0; i < n; i++) norm += (Lv[i] ?? 0) ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < n; i++) Lv[i]! /= norm;
      vecs[col] = Lv;
    }
  }
  return vecs;
}

function kmeansOnRows(
  rows: Float64Array[],
  k: number,
  maxIter = 100,
  nInit = 10,
): Int32Array {
  const n = rows.length;
  const d = rows[0]?.length ?? 0;
  let bestLabels = new Int32Array(n);
  let bestInertia = Number.POSITIVE_INFINITY;

  const rng = { seed: 0 };
  const rand = () => {
    rng.seed = (rng.seed * 1664525 + 1013904223) & 0xffffffff;
    return (rng.seed >>> 0) / 0xffffffff;
  };

  for (let init = 0; init < nInit; init++) {
    rng.seed = init * 1234 + 5678;
    const centers: Float64Array[] = Array.from({ length: k }, () => {
      const idx = Math.floor(rand() * n);
      return Float64Array.from(rows[idx] ?? new Float64Array(d));
    });
    const labels = new Int32Array(n);

    for (let iter = 0; iter < maxIter; iter++) {
      // Assign
      let changed = false;
      for (let i = 0; i < n; i++) {
        const xi = rows[i] as Float64Array;
        let best = 0;
        let bestDist = Number.POSITIVE_INFINITY;
        for (let c = 0; c < k; c++) {
          const cc = centers[c] as Float64Array;
          let dd = 0;
          for (let j = 0; j < d; j++) dd += ((xi[j] ?? 0) - (cc[j] ?? 0)) ** 2;
          if (dd < bestDist) { bestDist = dd; best = c; }
        }
        if (labels[i] !== best) { labels[i]! = best; changed = true; }
      }
      if (!changed) break;
      // Update centers
      for (const c of centers) c.fill(0);
      const counts = new Int32Array(k);
      for (let i = 0; i < n; i++) {
        const c = labels[i] ?? 0;
        counts[c]! += 1;
        const cc = centers[c] as Float64Array;
        const xi = rows[i] as Float64Array;
        for (let j = 0; j < d; j++) cc[j]! += xi[j] ?? 0;
      }
      for (let c = 0; c < k; c++) {
        const cnt = counts[c] ?? 1;
        if (cnt > 0) {
          const cc = centers[c] as Float64Array;
          for (let j = 0; j < d; j++) cc[j]! /= cnt;
        }
      }
    }

    let inertia = 0;
    for (let i = 0; i < n; i++) {
      const xi = rows[i] as Float64Array;
      const cc = centers[labels[i] ?? 0] as Float64Array;
      for (let j = 0; j < d; j++) inertia += ((xi[j] ?? 0) - (cc[j] ?? 0)) ** 2;
    }
    if (inertia < bestInertia) {
      bestInertia = inertia;
      bestLabels = Int32Array.from(labels);
    }
  }
  return bestLabels;
}

export class SpectralClustering {
  nClusters: number;
  nInit: number;
  gamma: number;

  labels_: Int32Array | null = null;
  affinityMatrix_: Float64Array[] | null = null;

  constructor(opts: SpectralClusteringOptions = {}) {
    this.nClusters = opts.nClusters ?? 8;
    this.nInit = opts.nInit ?? 10;
    this.gamma = opts.gamma ?? 1.0;
  }

  fit(X: Float64Array[]): this {
    const W = computeAffinityMatrix(X, this.gamma);
    this.affinityMatrix_ = W;
    const L = symmetricNormalizedLaplacian(W);
    const vecs = powerIterationEigenvectors(L, this.nClusters);
    const n = X.length;
    const k = this.nClusters;
    // Assemble rows from eigenvectors
    const rows: Float64Array[] = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        row[c]! = (vecs[c] as Float64Array)[i] ?? 0;
      }
      return row;
    });
    // Normalize rows to unit norm
    for (const row of rows) {
      let norm = 0;
      for (let j = 0; j < k; j++) norm += (row[j] ?? 0) ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let j = 0; j < k; j++) row[j]! /= norm;
    }
    this.labels_ = kmeansOnRows(rows, this.nClusters, 100, this.nInit);
    return this;
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    return this.labels_ as Int32Array;
  }
}

// ─── MeanShift ────────────────────────────────────────────────────────────────

export interface MeanShiftOptions {
  bandwidth?: number;
  maxIter?: number;
  tol?: number;
}

function gaussianKernelWeight(dist2: number, bandwidth: number): number {
  return Math.exp(-dist2 / (2 * bandwidth * bandwidth));
}

export class MeanShift {
  bandwidth: number;
  maxIter: number;
  tol: number;

  clusterCenters_: Float64Array[] | null = null;
  labels_: Int32Array | null = null;

  constructor(opts: MeanShiftOptions = {}) {
    this.bandwidth = opts.bandwidth ?? 1.0;
    this.maxIter = opts.maxIter ?? 300;
    this.tol = opts.tol ?? 1e-3;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    // Initialize one seed per point
    const seeds: Float64Array[] = X.map((x) => Float64Array.from(x));

    for (const seed of seeds) {
      for (let iter = 0; iter < this.maxIter; iter++) {
        const newSeed = new Float64Array(d);
        let totalWeight = 0;
        for (const xi of X) {
          let dist2 = 0;
          for (let j = 0; j < d; j++) dist2 += ((seed[j] ?? 0) - (xi[j] ?? 0)) ** 2;
          const w = gaussianKernelWeight(dist2, this.bandwidth);
          totalWeight += w;
          for (let j = 0; j < d; j++) newSeed[j]! += w * (xi[j] ?? 0);
        }
        if (totalWeight > 0) {
          for (let j = 0; j < d; j++) newSeed[j]! /= totalWeight;
        }
        let shift = 0;
        for (let j = 0; j < d; j++) shift += ((newSeed[j] ?? 0) - (seed[j] ?? 0)) ** 2;
        for (let j = 0; j < d; j++) seed[j]! = newSeed[j] ?? 0;
        if (Math.sqrt(shift) < this.tol) break;
      }
    }

    // Merge nearby seeds
    const mergedCenters: Float64Array[] = [];
    for (const seed of seeds) {
      let merged = false;
      for (const center of mergedCenters) {
        let dist2 = 0;
        for (let j = 0; j < d; j++) dist2 += ((seed[j] ?? 0) - (center[j] ?? 0)) ** 2;
        if (Math.sqrt(dist2) < this.bandwidth) { merged = true; break; }
      }
      if (!merged) mergedCenters.push(Float64Array.from(seed));
    }

    this.clusterCenters_ = mergedCenters;

    // Assign labels
    const labels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const xi = X[i] as Float64Array;
      let best = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let c = 0; c < mergedCenters.length; c++) {
        const cc = mergedCenters[c] as Float64Array;
        let dist2 = 0;
        for (let j = 0; j < d; j++) dist2 += ((xi[j] ?? 0) - (cc[j] ?? 0)) ** 2;
        if (dist2 < bestDist) { bestDist = dist2; best = c; }
      }
      labels[i]! = best;
    }
    this.labels_ = labels;
    return this;
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    return this.labels_ as Int32Array;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.clusterCenters_) throw new NotFittedError("MeanShift");
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const labels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const xi = X[i] as Float64Array;
      let best = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let c = 0; c < this.clusterCenters_.length; c++) {
        const cc = this.clusterCenters_[c] as Float64Array;
        let dist2 = 0;
        for (let j = 0; j < d; j++) dist2 += ((xi[j] ?? 0) - (cc[j] ?? 0)) ** 2;
        if (dist2 < bestDist) { bestDist = dist2; best = c; }
      }
      labels[i]! = best;
    }
    return labels;
  }
}

// ─── Birch ────────────────────────────────────────────────────────────────────

export interface BirchOptions {
  threshold?: number;
  branchingFactor?: number;
  nClusters?: number;
}

interface CFEntry {
  n: number;
  ls: Float64Array;
  ss: number;
}

export class Birch {
  threshold: number;
  branchingFactor: number;
  nClusters: number;

  labels_: Int32Array | null = null;
  subclusterCenters_: Float64Array[] | null = null;

  constructor(opts: BirchOptions = {}) {
    this.threshold = opts.threshold ?? 0.5;
    this.branchingFactor = opts.branchingFactor ?? 50;
    this.nClusters = opts.nClusters ?? 3;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const entries: CFEntry[] = [];

    for (const xi of X) {
      let inserted = false;
      for (const entry of entries) {
        const centroid = Float64Array.from({ length: d }, (_, j) => (entry.ls[j] ?? 0) / entry.n);
        let dist2 = 0;
        for (let j = 0; j < d; j++) dist2 += ((xi[j] ?? 0) - (centroid[j] ?? 0)) ** 2;
        if (Math.sqrt(dist2) <= this.threshold) {
          entry.n += 1;
          for (let j = 0; j < d; j++) entry.ls[j]! += xi[j] ?? 0;
          entry.ss += xi.reduce((s, v) => s + v * v, 0);
          inserted = true;
          break;
        }
      }
      if (!inserted) {
        entries.push({ n: 1, ls: Float64Array.from(xi), ss: xi.reduce((s, v) => s + v * v, 0) });
      }
    }

    const centers: Float64Array[] = entries.map((e) =>
      Float64Array.from({ length: d }, (_, j) => (e.ls[j] ?? 0) / e.n),
    );
    this.subclusterCenters_ = centers;

    // Use k-means on subcluster centers
    const k = Math.min(this.nClusters, centers.length);
    const subcluLabels = kmeansOnRows(centers, k, 100, 3);

    // Assign original points to the nearest subcluster then to its k-means label
    const labels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const xi = X[i] as Float64Array;
      let bestIdx = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let c = 0; c < centers.length; c++) {
        const cc = centers[c] as Float64Array;
        let dist2 = 0;
        for (let j = 0; j < d; j++) dist2 += ((xi[j] ?? 0) - (cc[j] ?? 0)) ** 2;
        if (dist2 < bestDist) { bestDist = dist2; bestIdx = c; }
      }
      labels[i]! = subcluLabels[bestIdx] ?? 0;
    }
    this.labels_ = labels;
    return this;
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    return this.labels_ as Int32Array;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.subclusterCenters_) throw new NotFittedError("Birch");
    const n = X.length;
    const d = X[0]?.length ?? 0;
    const labels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const xi = X[i] as Float64Array;
      let bestIdx = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let c = 0; c < this.subclusterCenters_.length; c++) {
        const cc = this.subclusterCenters_[c] as Float64Array;
        let dist2 = 0;
        for (let j = 0; j < d; j++) dist2 += ((xi[j] ?? 0) - (cc[j] ?? 0)) ** 2;
        if (dist2 < bestDist) { bestDist = dist2; bestIdx = c; }
      }
      labels[i]! = bestIdx;
    }
    return labels;
  }
}

// ─── OPTICS ───────────────────────────────────────────────────────────────────

export interface OPTICSOptions {
  minSamples?: number;
  maxEps?: number;
  xi?: number;
}

export class OPTICS {
  minSamples: number;
  maxEps: number;
  xi: number;

  labels_: Int32Array | null = null;
  reachabilityDistances_: Float64Array | null = null;
  coreDistances_: Float64Array | null = null;
  ordering_: Int32Array | null = null;

  constructor(opts: OPTICSOptions = {}) {
    this.minSamples = opts.minSamples ?? 5;
    this.maxEps = opts.maxEps ?? Number.POSITIVE_INFINITY;
    this.xi = opts.xi ?? 0.05;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;

    const dist = (a: Float64Array, b: Float64Array): number => {
      let s = 0;
      for (let i = 0; i < d; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
      return Math.sqrt(s);
    };

    // Compute all pairwise distances (for small datasets)
    const dists: Float64Array[] = Array.from({ length: n }, (_, i) =>
      Float64Array.from({ length: n }, (__, j) =>
        dist(X[i] as Float64Array, X[j] as Float64Array),
      ),
    );

    // Compute core distances
    const coreDist = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const row = Array.from(dists[i] as Float64Array).sort((a, b) => a - b);
      coreDist[i]! = row[this.minSamples] ?? Number.POSITIVE_INFINITY;
    }

    const processed = new Uint8Array(n);
    const reachDist = new Float64Array(n).fill(Number.POSITIVE_INFINITY);
    const ordering: number[] = [];

    const seeds: number[] = [];
    const updateSeeds = (idx: number) => {
      const cd = coreDist[idx] ?? Number.POSITIVE_INFINITY;
      for (let j = 0; j < n; j++) {
        if (processed[j]) continue;
        const newRD = Math.max(cd, (dists[idx] as Float64Array)[j] ?? Number.POSITIVE_INFINITY);
        if (newRD < (reachDist[j] ?? Number.POSITIVE_INFINITY)) {
          reachDist[j]! = newRD;
          if (!seeds.includes(j)) seeds.push(j);
        }
      }
    };

    for (let start = 0; start < n; start++) {
      if (processed[start]) continue;
      processed[start]! = 1;
      ordering.push(start);
      if ((coreDist[start] ?? Number.POSITIVE_INFINITY) <= this.maxEps) {
        updateSeeds(start);
        while (seeds.length > 0) {
          // Pick seed with minimum reachability distance
          let minIdx = 0;
          let minRD = Number.POSITIVE_INFINITY;
          for (let s = 0; s < seeds.length; s++) {
            const sd = seeds[s] ?? 0;
            const rd = reachDist[sd] ?? Number.POSITIVE_INFINITY;
            if (rd < minRD) { minRD = rd; minIdx = s; }
          }
          const q = seeds[minIdx] ?? 0;
          seeds.splice(minIdx, 1);
          if (processed[q]) continue;
          processed[q]! = 1;
          ordering.push(q);
          if ((coreDist[q] ?? Number.POSITIVE_INFINITY) <= this.maxEps) {
            updateSeeds(q);
          }
        }
      }
    }

    // Assign labels via xi-cluster extraction (simplified: threshold-based)
    const labels = new Int32Array(n).fill(-1);
    let clusterId = 0;
    const eps = this.xi * (reachDist.reduce((mx, v) => Math.max(mx, isFinite(v) ? v : 0), 0));
    let currentCluster = -1;
    for (const idx of ordering) {
      const rd = reachDist[idx] ?? Number.POSITIVE_INFINITY;
      if (rd <= eps && (coreDist[idx] ?? Number.POSITIVE_INFINITY) <= this.maxEps) {
        if (currentCluster === -1) { currentCluster = clusterId++; }
        labels[idx]! = currentCluster;
      } else {
        currentCluster = -1;
      }
    }

    this.labels_ = labels;
    this.reachabilityDistances_ = reachDist;
    this.coreDistances_ = coreDist;
    this.ordering_ = Int32Array.from(ordering);
    return this;
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    return this.labels_ as Int32Array;
  }
}
