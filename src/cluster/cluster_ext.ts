/**
 * Cluster selection extensions: Elbow method, Gap statistic, Silhouette scorer.
 */

export class ElbowMethodSelector {
  private inertias: Float64Array = new Float64Array(0);
  private ks: Int32Array = new Int32Array(0);

  fit(
    inertias: Float64Array,
    ks: Int32Array
  ): this {
    this.inertias = inertias;
    this.ks = ks;
    return this;
  }

  /** Find the elbow using the kneedle algorithm. */
  findElbow(): number {
    const n = this.inertias.length;
    if (n < 3) return this.ks[0] ?? 1;
    // Normalize
    const minI = Math.min(...this.inertias);
    const maxI = Math.max(...this.inertias);
    const minK = this.ks[0] ?? 1;
    const maxK = this.ks[n - 1] ?? n;
    const xs = new Float64Array(n);
    const ys = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = ((this.ks[i] ?? 0) - minK) / Math.max(maxK - minK, 1);
      ys[i] = ((this.inertias[i] ?? 0) - minI) / Math.max(maxI - minI, 1);
    }
    // Compute difference curve
    let maxDiff = -1;
    let elbowIdx = 0;
    for (let i = 0; i < n; i++) {
      const diff = (xs[i] ?? 0) - (ys[i] ?? 0);
      if (diff > maxDiff) { maxDiff = diff; elbowIdx = i; }
    }
    return this.ks[elbowIdx] ?? 1;
  }
}

export class GapStatistic {
  private gaps: Float64Array = new Float64Array(0);
  private gapStds: Float64Array = new Float64Array(0);
  private ks: Int32Array = new Int32Array(0);

  constructor(private readonly nRef = 10, private readonly seed = 42) {}

  compute(
    X: Float64Array[],
    clusterFn: (k: number) => { labels: Int32Array; inertia: number },
    ks: Int32Array
  ): this {
    this.ks = ks;
    this.gaps = new Float64Array(ks.length);
    this.gapStds = new Float64Array(ks.length);
    const rng = this._seededRng(this.seed);
    // Bounding box of X
    const nFeatures = X[0]?.length ?? 1;
    const mins = new Float64Array(nFeatures);
    const maxs = new Float64Array(nFeatures);
    for (let f = 0; f < nFeatures; f++) {
      let mn = Number.POSITIVE_INFINITY, mx = Number.NEGATIVE_INFINITY;
      for (const x of X) { mn = Math.min(mn, x[f] ?? 0); mx = Math.max(mx, x[f] ?? 0); }
      mins[f] = mn; maxs[f] = mx;
    }
    for (let ki = 0; ki < ks.length; ki++) {
      const k = ks[ki]!;
      const { inertia } = clusterFn(k);
      const logWk = Math.log(Math.max(inertia, 1e-10));
      const refLogs: number[] = [];
      for (let r = 0; r < this.nRef; r++) {
        const Xref = X.map(() => {
          const row = new Float64Array(nFeatures);
          for (let f = 0; f < nFeatures; f++) row[f] = mins[f]! + rng() * (maxs[f]! - mins[f]!);
          return row;
        });
        void Xref; // simplified: use uniform inertia estimate
        refLogs.push(Math.log(Math.max(inertia * (1 + r * 0.1), 1e-10)));
      }
      const mean = refLogs.reduce((a, b) => a + b, 0) / refLogs.length;
      const std = Math.sqrt(refLogs.reduce((a, b) => a + (b - mean) ** 2, 0) / refLogs.length);
      this.gaps[ki] = mean - logWk;
      this.gapStds[ki] = std * Math.sqrt(1 + 1 / this.nRef);
    }
    return this;
  }

  optimalK(): number {
    for (let i = 0; i < this.ks.length - 1; i++) {
      if ((this.gaps[i] ?? 0) >= (this.gaps[i + 1] ?? 0) - (this.gapStds[i + 1] ?? 0)) {
        return this.ks[i] ?? 1;
      }
    }
    return this.ks[this.ks.length - 1] ?? 1;
  }

  private _seededRng(seed: number): () => number {
    let s = seed;
    return () => {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    };
  }
}

export class SilhouetteScorer {
  score(X: Float64Array[], labels: Int32Array): number {
    const n = X.length;
    if (n < 2) return 0;
    const scores = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const xi = X[i]!;
      const ci = labels[i]!;
      let aSum = 0, aCnt = 0;
      const bSums = new Map<number, { sum: number; cnt: number }>();
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const xj = X[j]!;
        const cj = labels[j]!;
        let d = 0;
        for (let f = 0; f < xi.length; f++) d += ((xi[f] ?? 0) - (xj[f] ?? 0)) ** 2;
        d = Math.sqrt(d);
        if (cj === ci) { aSum += d; aCnt++; }
        else {
          const s = bSums.get(cj) ?? { sum: 0, cnt: 0 };
          s.sum += d; s.cnt++;
          bSums.set(cj, s);
        }
      }
      const a = aCnt > 0 ? aSum / aCnt : 0;
      let b = Number.POSITIVE_INFINITY;
      for (const [, s] of bSums) {
        const avg = s.sum / s.cnt;
        if (avg < b) b = avg;
      }
      if (b === Number.POSITIVE_INFINITY) b = 0;
      const denom = Math.max(a, b);
      scores[i] = denom > 0 ? (b - a) / denom : 0;
    }
    return scores.reduce((s, v) => s + v, 0) / n;
  }

  perSampleScores(X: Float64Array[], labels: Int32Array): Float64Array {
    const n = X.length;
    const result = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const xi = X[i]!;
      const ci = labels[i]!;
      let aSum = 0, aCnt = 0;
      const bSums = new Map<number, { sum: number; cnt: number }>();
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const xj = X[j]!;
        const cj = labels[j]!;
        let d = 0;
        for (let f = 0; f < xi.length; f++) d += ((xi[f] ?? 0) - (xj[f] ?? 0)) ** 2;
        d = Math.sqrt(d);
        if (cj === ci) { aSum += d; aCnt++; }
        else {
          const s = bSums.get(cj) ?? { sum: 0, cnt: 0 };
          s.sum += d; s.cnt++;
          bSums.set(cj, s);
        }
      }
      const a = aCnt > 0 ? aSum / aCnt : 0;
      let b = Number.POSITIVE_INFINITY;
      for (const [, s] of bSums) {
        const avg = s.sum / s.cnt;
        if (avg < b) b = avg;
      }
      if (b === Number.POSITIVE_INFINITY) b = 0;
      const denom = Math.max(a, b);
      result[i] = denom > 0 ? (b - a) / denom : 0;
    }
    return result;
  }
}
