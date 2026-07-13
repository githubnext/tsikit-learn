/**
 * RANSACRegressor and MultiTaskElasticNet — sklearn linear_model ports.
 */

export class RANSACRegressor {
  minSamples: number;
  residualThreshold: number;
  maxTrials: number;
  maxSkips: number;
  stopNInliers: number;
  stopScore: number;
  stopProbability: number;
  coef_: Float64Array | null = null;
  intercept_: number = 0;
  inlierMask_: Uint8Array | null = null;
  nTrials_: number = 0;
  nSkips_: number = 0;

  constructor(
    minSamples = 0.1,
    residualThreshold = 1.0,
    maxTrials = 100,
    maxSkips = Number.POSITIVE_INFINITY,
    stopNInliers = Number.POSITIVE_INFINITY,
    stopScore = 1.0,
    stopProbability = 0.99,
  ) {
    this.minSamples = minSamples;
    this.residualThreshold = residualThreshold;
    this.maxTrials = maxTrials;
    this.maxSkips = maxSkips;
    this.stopNInliers = stopNInliers;
    this.stopScore = stopScore;
    this.stopProbability = stopProbability;
  }

  private _fitOLS(X: Float64Array[], y: Float64Array): { coef: Float64Array; intercept: number } {
    const n = X.length;
    const p = (X[0]?.length ?? 0);
    if (n === 0 || p === 0) return { coef: new Float64Array(0), intercept: 0 };

    const xMean = new Float64Array(p);
    let yMean = 0;
    for (let i = 0; i < n; i++) {
      yMean += (y[i] ?? 0) / n;
      for (let j = 0; j < p; j++) xMean[j]! += (X[i]?.[j] ?? 0) / n;
    }

    const XtX: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));
    const Xty = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < p; j++) {
        const xij = (X[i]?.[j] ?? 0) - (xMean[j] ?? 0);
        Xty[j]! += xij * ((y[i] ?? 0) - yMean);
        for (let k = 0; k < p; k++) {
          (XtX[j] as Float64Array)[k]! += xij * ((X[i]?.[k] ?? 0) - (xMean[k] ?? 0));
        }
      }
    }

    // Solve via diagonal approximation
    const coef = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      const diag = (XtX[j] as Float64Array)[j] ?? 0;
      coef[j] = diag > 1e-10 ? (Xty[j] ?? 0) / diag : 0;
    }
    let intercept = yMean;
    for (let j = 0; j < p; j++) intercept -= (coef[j] ?? 0) * (xMean[j] ?? 0);
    return { coef, intercept };
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const minS = this.minSamples < 1 ? Math.ceil(this.minSamples * n) : Math.min(Math.ceil(this.minSamples), n);

    let bestInliers = 0;
    let bestCoef = new Float64Array(p);
    let bestIntercept = 0;
    let bestMask = new Uint8Array(n);

    this.nTrials_ = 0;
    this.nSkips_ = 0;

    for (let trial = 0; trial < this.maxTrials; trial++) {
      this.nTrials_++;
      // Random sample
      const sampleIdx = sampleWithoutReplacement(n, minS);
      const Xs = sampleIdx.map((i) => X[i] as Float64Array);
      const ys = new Float64Array(sampleIdx.map((i) => y[i] ?? 0));

      const { coef, intercept } = this._fitOLS(Xs, ys);

      // Compute residuals for all points
      const mask = new Uint8Array(n);
      let nInliers = 0;
      for (let i = 0; i < n; i++) {
        let yhat = intercept;
        for (let j = 0; j < p; j++) yhat += (X[i]?.[j] ?? 0) * (coef[j] ?? 0);
        if (Math.abs((y[i] ?? 0) - yhat) <= this.residualThreshold) {
          mask[i] = 1;
          nInliers++;
        }
      }

      if (nInliers > bestInliers) {
        bestInliers = nInliers;
        bestMask = mask;
        // Refit on all inliers
        const Xinl = X.filter((_, i) => mask[i] === 1);
        const yInl = new Float64Array(y.filter((_, i) => mask[i] === 1));
        const refitted = this._fitOLS(Xinl, yInl);
        bestCoef = refitted.coef as Float64Array<ArrayBuffer>;
        bestIntercept = refitted.intercept;
      }

      if (bestInliers >= this.stopNInliers) break;
    }

    this.coef_ = bestCoef;
    this.intercept_ = bestIntercept;
    this.inlierMask_ = bestMask;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    const coef = this.coef_ ?? new Float64Array(0);
    return new Float64Array(X.map((row) => {
      let v = this.intercept_;
      for (let j = 0; j < coef.length; j++) v += (row[j] ?? 0) * (coef[j] ?? 0);
      return v;
    }));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = y.reduce((a, b) => a + b, 0) / y.length;
    let ss_res = 0, ss_tot = 0;
    for (let i = 0; i < y.length; i++) {
      ss_res += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
      ss_tot += ((y[i] ?? 0) - yMean) ** 2;
    }
    return ss_tot === 0 ? 0 : 1 - ss_res / ss_tot;
  }
}

function sampleWithoutReplacement(n: number, k: number): number[] {
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = indices[i]; indices[i] = indices[j] as number; indices[j] = tmp as number;
  }
  return indices.slice(0, k);
}

export class MultiTaskElasticNet {
  alpha: number;
  l1Ratio: number;
  fitIntercept: boolean;
  maxIter: number;
  tol: number;
  coef_: Float64Array[] | null = null;
  intercept_: Float64Array | null = null;
  nIter_: number = 0;
  dualGap_: number = 0;

  constructor(alpha = 1.0, l1Ratio = 0.5, fitIntercept = true, maxIter = 1000, tol = 1e-4) {
    this.alpha = alpha;
    this.l1Ratio = l1Ratio;
    this.fitIntercept = fitIntercept;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const nTasks = Y[0]?.length ?? 0;

    const xMean = new Float64Array(p);
    const yMean = new Float64Array(nTasks);
    if (this.fitIntercept) {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < p; j++) xMean[j]! += (X[i]?.[j] ?? 0) / n;
        for (let t = 0; t < nTasks; t++) yMean[t]! += (Y[i]?.[t] ?? 0) / n;
      }
    }

    // BCD (block coordinate descent) per feature
    const W: Float64Array[] = Array.from({ length: p }, () => new Float64Array(nTasks));
    const alpha1 = this.alpha * this.l1Ratio;
    const alpha2 = this.alpha * (1 - this.l1Ratio);

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxChange = 0;
      for (let j = 0; j < p; j++) {
        // Compute partial residual for feature j
        const grad = new Float64Array(nTasks);
        for (let i = 0; i < n; i++) {
          const xij = (X[i]?.[j] ?? 0) - (this.fitIntercept ? (xMean[j] ?? 0) : 0);
          for (let t = 0; t < nTasks; t++) {
            let res = (Y[i]?.[t] ?? 0) - (this.fitIntercept ? (yMean[t] ?? 0) : 0);
            for (let k = 0; k < p; k++) {
              if (k !== j) res -= ((X[i]?.[k] ?? 0) - (this.fitIntercept ? (xMean[k] ?? 0) : 0)) * ((W[k] as Float64Array)[t] ?? 0);
            }
            grad[t]! += xij * res / n;
          }
        }
        // L21 proximal operator
        const norm = Math.sqrt(grad.reduce((s, v) => s + v * v, 0));
        const denom = 1 + alpha2;
        const threshold = alpha1;
        const scale = norm > threshold ? (norm - threshold) / (norm * denom) : 0;
        const oldW = new Float64Array(W[j] as Float64Array);
        for (let t = 0; t < nTasks; t++) (W[j] as Float64Array)[t] = scale * (grad[t] ?? 0);
        const diff = Array.from(W[j] as Float64Array).reduce((s, v, t) => s + (v - (oldW[t] ?? 0)) ** 2, 0);
        maxChange = Math.max(maxChange, Math.sqrt(diff));
      }
      this.nIter_ = iter + 1;
      if (maxChange < this.tol) break;
    }

    this.coef_ = W;
    if (this.fitIntercept) {
      const intercept = new Float64Array(nTasks);
      for (let t = 0; t < nTasks; t++) {
        intercept[t] = yMean[t] ?? 0;
        for (let j = 0; j < p; j++) intercept[t]! -= ((W[j] as Float64Array)[t] ?? 0) * (xMean[j] ?? 0);
      }
      this.intercept_ = intercept;
    } else {
      this.intercept_ = new Float64Array(nTasks);
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array[] {
    const W = this.coef_ ?? [];
    const intercept = this.intercept_ ?? new Float64Array(0);
    const nTasks = intercept.length;
    const p = W.length;
    return X.map((row) => {
      const out = new Float64Array(nTasks);
      for (let t = 0; t < nTasks; t++) out[t] = intercept[t] ?? 0;
      for (let j = 0; j < p; j++) {
        for (let t = 0; t < nTasks; t++) {
          out[t]! += (row[j] ?? 0) * ((W[j] as Float64Array)[t] ?? 0);
        }
      }
      return out;
    });
  }
}

export class MultiTaskLasso extends MultiTaskElasticNet {
  constructor(alpha = 1.0, fitIntercept = true, maxIter = 1000, tol = 1e-4) {
    super(alpha, 1.0, fitIntercept, maxIter, tol);
  }
}
