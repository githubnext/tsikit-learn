/**
 * Extended discriminant analysis: LDA utilities, QDA helpers,
 * shrinkage LDA, and discriminant projections.
 */

/** Compute class means and scatter matrices for LDA. */
export interface LDAComponents {
  classMeans: Map<number, Float64Array>;
  withinScatter: Float64Array[];
  betweenScatter: Float64Array[];
  globalMean: Float64Array;
  priors: Map<number, number>;
}

export function computeLDAComponents(
  X: Float64Array[],
  y: Int32Array,
): LDAComponents {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const classes = [...new Set(Array.from(y))].sort((a, b) => a - b);

  const globalMean = new Float64Array(d);
  for (const xi of X) for (let j = 0; j < d; j++) globalMean[j] = (globalMean[j] ?? 0) + (xi[j] ?? 0) / n;

  const classMeans = new Map<number, Float64Array>();
  const priors = new Map<number, number>();
  for (const c of classes) {
    const members = X.filter((_, i) => (y[i] ?? -1) === c);
    priors.set(c, members.length / n);
    const mean = new Float64Array(d);
    for (const xi of members) for (let j = 0; j < d; j++) mean[j] = (mean[j] ?? 0) + (xi[j] ?? 0) / members.length;
    classMeans.set(c, mean);
  }

  // Within-class scatter Sw
  const Sw = Array.from({ length: d }, () => new Float64Array(d));
  for (let i = 0; i < n; i++) {
    const xi = X[i];
    if (xi === undefined) continue;
    const c = y[i] ?? 0;
    const mu = classMeans.get(c) ?? new Float64Array(d);
    for (let j = 0; j < d; j++) {
      for (let k = 0; k < d; k++) {
        Sw[j]![k] = (Sw[j]![k] ?? 0) + ((xi[j] ?? 0) - (mu[j] ?? 0)) * ((xi[k] ?? 0) - (mu[k] ?? 0));
      }
    }
  }

  // Between-class scatter Sb
  const Sb = Array.from({ length: d }, () => new Float64Array(d));
  for (const c of classes) {
    const mu = classMeans.get(c) ?? new Float64Array(d);
    const nc = Math.round((priors.get(c) ?? 0) * n);
    for (let j = 0; j < d; j++) {
      for (let k = 0; k < d; k++) {
        Sb[j]![k] = (Sb[j]![k] ?? 0) + nc * ((mu[j] ?? 0) - (globalMean[j] ?? 0)) * ((mu[k] ?? 0) - (globalMean[k] ?? 0));
      }
    }
  }

  return { classMeans, withinScatter: Sw, betweenScatter: Sb, globalMean, priors };
}

/** Mahalanobis distance from a point to a class. */
export function mahalanobisDistance(
  x: Float64Array,
  mean: Float64Array,
  covInv: Float64Array[],
): number {
  const d = x.length;
  let dist = 0;
  const diff = x.map((v, j) => v - (mean[j] ?? 0));
  for (let j = 0; j < d; j++) {
    let sum = 0;
    for (let k = 0; k < d; k++) sum += (covInv[j]?.[k] ?? 0) * (diff[k] ?? 0);
    dist += (diff[j] ?? 0) * sum;
  }
  return Math.sqrt(Math.max(0, dist));
}

/** Regularized LDA (shrinkage toward identity). */
export class RegularizedLDA {
  nComponents: number;
  shrinkage: number;
  components_?: Float64Array[];
  classMeans_?: Map<number, Float64Array>;
  priors_?: Map<number, number>;
  classes_?: Int32Array;

  constructor(nComponents = 1, shrinkage = 0.0) {
    this.nComponents = nComponents;
    this.shrinkage = shrinkage;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const lda = computeLDAComponents(X, y);
    this.classMeans_ = lda.classMeans;
    this.priors_ = lda.priors;
    this.classes_ = Int32Array.from([...lda.classMeans.keys()].sort((a, b) => a - b));

    const d = X[0]?.length ?? 0;
    // Shrinkage regularization on Sw
    const Sw = lda.withinScatter.map((row, j) =>
      row.map((v, k) => v + (j === k ? this.shrinkage * v : 0))
    );
    // Simple projection: use class means as discriminant directions
    this.components_ = [...lda.classMeans.values()].slice(0, this.nComponents);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new Error("Not fitted");
    const k = this.components_.length;
    return X.map((xi) =>
      new Float64Array(k).map((_, c) => {
        const comp = this.components_![c];
        if (comp === undefined) return 0;
        let dot = 0;
        for (let j = 0; j < xi.length; j++) dot += (xi[j] ?? 0) * (comp[j] ?? 0);
        return dot;
      })
    );
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.classMeans_ || !this.classes_) throw new Error("Not fitted");
    return Int32Array.from(X.map((xi) => {
      let best = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (const [c, mu] of this.classMeans_!) {
        let dist = 0;
        for (let j = 0; j < xi.length; j++) dist += ((xi[j] ?? 0) - (mu[j] ?? 0)) ** 2;
        if (dist < bestDist) { bestDist = dist; best = c; }
      }
      return best;
    }));
  }
}

/** Factor analysis-based dimensionality for LDA. */
export function ldaOptimalComponents(
  betweenScatter: Float64Array[],
  nClasses: number,
  nFeatures: number,
): number {
  return Math.min(nClasses - 1, nFeatures);
}
