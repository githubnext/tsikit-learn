/**
 * BisectingKMeans: divisive hierarchical clustering using k-means bisection.
 * Mirrors sklearn.cluster.BisectingKMeans.
 */

import { NotFittedError } from "../exceptions.js";

function euclidean(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.sqrt(s);
}

function clusterMean(points: Float64Array[]): Float64Array {
  if (points.length === 0) return new Float64Array(0);
  const p = (points[0] ?? new Float64Array(0)).length;
  const m = new Float64Array(p);
  for (const pt of points)
    for (let j = 0; j < p; j++) m[j] = (m[j] ?? 0) + (pt[j] ?? 0);
  for (let j = 0; j < p; j++) m[j] = (m[j] ?? 0) / points.length;
  return m;
}

function clusterSSE(points: Float64Array[], center: Float64Array): number {
  let s = 0;
  for (const pt of points) {
    for (let j = 0; j < pt.length; j++)
      s += ((pt[j] ?? 0) - (center[j] ?? 0)) ** 2;
  }
  return s;
}

/** Run k-means with k=2 on the given points. Returns cluster assignments. */
function bisect(
  points: Float64Array[],
  maxIter: number,
  rng: number,
): { labels: Int32Array; centers: Float64Array[] } {
  const n = points.length;
  const p = (points[0] ?? new Float64Array(0)).length;

  if (n <= 1) {
    return {
      labels: new Int32Array(n),
      centers: [clusterMean(points), new Float64Array(p)],
    };
  }

  // Init: pick 2 random centers
  const i0 = Math.abs(rng) % n;
  const i1 = (Math.abs(rng) + 1) % n;
  let centers = [
    new Float64Array(points[i0] ?? new Float64Array(p)),
    new Float64Array(points[i1] ?? new Float64Array(p)),
  ];
  let labels = new Int32Array(n);

  for (let iter = 0; iter < maxIter; iter++) {
    // Assign
    const newLabels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const d0 = euclidean(
        points[i] ?? new Float64Array(p),
        centers[0] ?? new Float64Array(p),
      );
      const d1 = euclidean(
        points[i] ?? new Float64Array(p),
        centers[1] ?? new Float64Array(p),
      );
      newLabels[i] = d1 < d0 ? 1 : 0;
    }

    // Update centers
    const c0 = points.filter((_, i) => newLabels[i] === 0);
    const c1 = points.filter((_, i) => newLabels[i] === 1);
    const newCenters = [
      c0.length > 0 ? clusterMean(c0) : (centers[0] ?? new Float64Array(p)),
      c1.length > 0 ? clusterMean(c1) : (centers[1] ?? new Float64Array(p)),
    ];

    // Check convergence
    let changed = false;
    for (let i = 0; i < n; i++)
      if (newLabels[i] !== labels[i]) {
        changed = true;
        break;
      }
    labels = newLabels;
    centers = newCenters;
    if (!changed) break;
  }

  return {
    labels,
    centers: [
      centers[0] ?? new Float64Array(p),
      centers[1] ?? new Float64Array(p),
    ],
  };
}

/**
 * BisectingKMeans: hierarchical divisive clustering.
 * Repeatedly bisects the cluster with highest SSE.
 * Mirrors sklearn.cluster.BisectingKMeans.
 */
export class BisectingKMeans {
  nClusters: number;
  maxIter: number;
  randomState: number;
  bisectingStrategy: "biggest_inertia" | "largest_cluster";

  clusterCenters_: Float64Array[] | null = null;
  labels_: Int32Array | null = null;
  inertia_: number = 0;
  nIter_: number = 0;

  constructor(
    options: {
      nClusters?: number;
      maxIter?: number;
      randomState?: number;
      bisectingStrategy?: "biggest_inertia" | "largest_cluster";
    } = {},
  ) {
    this.nClusters = options.nClusters ?? 8;
    this.maxIter = options.maxIter ?? 300;
    this.randomState = options.randomState ?? 42;
    this.bisectingStrategy = options.bisectingStrategy ?? "biggest_inertia";
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const k = Math.min(this.nClusters, n);

    // Start: all points in one cluster
    const clusterLabels = new Int32Array(n);
    const clusterCenters: Float64Array[] = [clusterMean(X)];
    let nClusters = 1;

    let rng = this.randomState;

    while (nClusters < k) {
      // Find cluster to bisect
      let targetCluster = 0;
      let bestCrit = -Number.POSITIVE_INFINITY;

      for (let c = 0; c < nClusters; c++) {
        const pts = X.filter((_, i) => clusterLabels[i] === c);
        if (pts.length <= 1) continue;
        const crit =
          this.bisectingStrategy === "biggest_inertia"
            ? clusterSSE(pts, clusterCenters[c] ?? new Float64Array(p))
            : pts.length;
        if (crit > bestCrit) {
          bestCrit = crit;
          targetCluster = c;
        }
      }

      const targetPoints = X.filter(
        (_, i) => clusterLabels[i] === targetCluster,
      );
      const targetIndices = Array.from({ length: n }, (_, i) => i).filter(
        (i) => clusterLabels[i] === targetCluster,
      );

      if (targetPoints.length <= 1) break;

      rng = Math.abs(rng * 1664525 + 1013904223) % 2147483647;
      const { labels: subLabels } = bisect(targetPoints, this.maxIter, rng);

      // Update global labels: targetCluster stays for subLabel=0, nClusters for subLabel=1
      for (let i = 0; i < targetIndices.length; i++) {
        const idx = targetIndices[i] ?? 0;
        if ((subLabels[i] ?? 0) === 1) clusterLabels[idx] = nClusters;
      }

      // Recompute centers for the two new clusters
      const c0pts = X.filter((_, i) => clusterLabels[i] === targetCluster);
      const c1pts = X.filter((_, i) => clusterLabels[i] === nClusters);
      clusterCenters[targetCluster] =
        c0pts.length > 0 ? clusterMean(c0pts) : new Float64Array(p);
      clusterCenters.push(
        c1pts.length > 0 ? clusterMean(c1pts) : new Float64Array(p),
      );
      nClusters++;
      this.nIter_++;
    }

    this.labels_ = clusterLabels;
    this.clusterCenters_ = clusterCenters;

    // Compute inertia
    let inertia = 0;
    for (let i = 0; i < n; i++) {
      const c = clusterLabels[i] ?? 0;
      const center = clusterCenters[c] ?? new Float64Array(p);
      const xi = X[i] ?? new Float64Array(p);
      for (let j = 0; j < p; j++)
        inertia += ((xi[j] ?? 0) - (center[j] ?? 0)) ** 2;
    }
    this.inertia_ = inertia;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (this.clusterCenters_ === null)
      throw new NotFittedError("BisectingKMeans");
    const centers = this.clusterCenters_;
    return new Int32Array(
      X.map((xi) => {
        let bestC = 0;
        let bestD = Number.POSITIVE_INFINITY;
        for (let c = 0; c < centers.length; c++) {
          const d = euclidean(xi, centers[c] ?? new Float64Array(0));
          if (d < bestD) {
            bestD = d;
            bestC = c;
          }
        }
        return bestC;
      }),
    );
  }

  fitPredict(X: Float64Array[]): Int32Array {
    this.fit(X);
    return this.labels_!;
  }

  score(X: Float64Array[]): number {
    if (this.clusterCenters_ === null)
      throw new NotFittedError("BisectingKMeans");
    const labels = this.predict(X);
    const centers = this.clusterCenters_;
    let inertia = 0;
    for (let i = 0; i < X.length; i++) {
      const c = labels[i] ?? 0;
      const center = centers[c] ?? new Float64Array(0);
      const xi = X[i] ?? new Float64Array(0);
      for (let j = 0; j < xi.length; j++)
        inertia += ((xi[j] ?? 0) - (center[j] ?? 0)) ** 2;
    }
    return -inertia;
  }
}
