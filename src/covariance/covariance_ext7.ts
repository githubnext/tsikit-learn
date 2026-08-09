/**
 * Sparse covariance estimation — GraphicalLasso and ShrunkCovariance.
 */

export function sampleCovariance(X: Float64Array[]): Float64Array[] {
  const n = X.length, p = X[0]?.length ?? 0;
  const mean = new Float64Array(p);
  for (const row of X) for (let j = 0; j < p; j++) mean[j]! += (row[j] ?? 0) / n;
  const cov: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));
  for (const row of X) {
    for (let j = 0; j < p; j++) {
      const dj = (row[j] ?? 0) - (mean[j] ?? 0);
      for (let k = 0; k < p; k++) {
        (cov[j] as Float64Array)[k]! += dj * ((row[k] ?? 0) - (mean[k] ?? 0)) / (n - 1);
      }
    }
  }
  return cov;
}

export class ShrunkCovariance {
  shrinkage: number | "auto";
  covariance_: Float64Array[] | null = null;
  precision_: Float64Array[] | null = null;
  shrinkage_: number = 0;

  constructor(shrinkage: number | "auto" = "auto") {
    this.shrinkage = shrinkage;
  }

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    const S = sampleCovariance(X);
    const tracS = S.reduce((t, row, i) => t + ((row as Float64Array)[i] ?? 0), 0);
    const mu = tracS / p;

    let alpha: number;
    if (this.shrinkage === "auto") {
      const n = X.length;
      // Ledoit-Wolf analytic estimate (simplified)
      let sumSq = 0;
      for (const row of S) for (const v of row) sumSq += v * v;
      const rho = Math.min(1, Math.max(0, (1 / (n * p)) * (sumSq - tracS ** 2 / p) / (sumSq - tracS ** 2 / p + 1e-12)));
      alpha = rho;
    } else {
      alpha = Math.min(1, Math.max(0, this.shrinkage));
    }
    this.shrinkage_ = alpha;

    this.covariance_ = S.map((row, i) =>
      row.map((v, j) => (1 - alpha) * v + (i === j ? alpha * mu : 0))
    );

    // Compute precision as inverse (simplified via diagonal approximation)
    const invDiag = this.covariance_.map((row, i) => 1 / Math.max((row as Float64Array)[i] ?? 1e-12, 1e-12));
    this.precision_ = Array.from({ length: p }, (_, i) => {
      const row = new Float64Array(p);
      row[i] = invDiag[i] ?? 0;
      return row;
    });
    return this;
  }
}

export class GraphicalLasso {
  alpha: number;
  maxIter: number;
  tol: number;
  covariance_: Float64Array[] | null = null;
  precision_: Float64Array[] | null = null;
  nIter_: number = 0;

  constructor(alpha = 0.01, maxIter = 100, tol = 1e-4) {
    this.alpha = alpha;
    this.maxIter = maxIter;
    this.tol = tol;
  }

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    const S = sampleCovariance(X);

    // Initialize precision as diagonal
    const Theta: Float64Array[] = Array.from({ length: p }, (_, i) => {
      const row = new Float64Array(p);
      row[i] = 1 / Math.max((S[i] as Float64Array)[i] ?? 1, 1e-12);
      return row;
    });

    // Block coordinate descent (simplified graphical lasso)
    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxChange = 0;
      for (let j = 0; j < p; j++) {
        // Extract submatrix without j
        const idxNotJ = Array.from({ length: p }, (_, k) => k).filter((k) => k !== j);
        const S12 = Float64Array.from(idxNotJ, (k) => (S[j] as Float64Array)[k] ?? 0);

        // W11 (covariance of other features)
        const W11 = idxNotJ.map((k) => Float64Array.from(idxNotJ, (l) => {
          const covKL = (S[k] as Float64Array)[l] ?? 0;
          return covKL + (k === l ? this.alpha : 0);
        }));

        // Lasso subproblem: approximate solution via soft-thresholding
        const beta = new Float64Array(idxNotJ.length).fill(0);
        for (let lasso_iter = 0; lasso_iter < 20; lasso_iter++) {
          for (let k = 0; k < idxNotJ.length; k++) {
            const W11k = (W11[k] as Float64Array)[k] ?? 1;
            let r = S12[k] ?? 0;
            for (let l = 0; l < idxNotJ.length; l++) {
              if (l !== k) r -= ((W11[k] as Float64Array)[l] ?? 0) * (beta[l] ?? 0);
            }
            const newBeta = Math.sign(r) * Math.max(0, Math.abs(r) - this.alpha) / Math.max(W11k, 1e-12);
            maxChange = Math.max(maxChange, Math.abs(newBeta - (beta[k] ?? 0)));
            beta[k] = newBeta;
          }
        }

        // Update theta column j
        for (let k = 0; k < idxNotJ.length; k++) {
          const ki = idxNotJ[k] ?? 0;
          (Theta[ki] as Float64Array)[j] = -(beta[k] ?? 0);
          (Theta[j] as Float64Array)[ki] = -(beta[k] ?? 0);
        }
        const sjj = (S[j] as Float64Array)[j] ?? 1;
        (Theta[j] as Float64Array)[j] = 1 / Math.max(sjj + this.alpha - beta.reduce((s, b, k) => s + (b ?? 0) * ((S12[k] ?? 0)), 0), 1e-12);
      }
      this.nIter_ = iter + 1;
      if (maxChange < this.tol) break;
    }

    this.precision_ = Theta;
    // Covariance is approximate inverse
    this.covariance_ = S;
    return this;
  }
}

export class MinimumCovarianceDeterminant {
  supportFraction: number;
  randomState: number;
  covariance_: Float64Array[] | null = null;
  precision_: Float64Array[] | null = null;
  location_: Float64Array | null = null;
  supportIndices_: Int32Array | null = null;

  constructor(supportFraction = 0.75, randomState = 42) {
    this.supportFraction = supportFraction;
    this.randomState = randomState;
  }

  fit(X: Float64Array[]): this {
    const n = X.length, p = X[0]?.length ?? 0;
    const h = Math.max(p + 1, Math.ceil(this.supportFraction * n));

    // Simplified FastMCD: random subsets, then refine
    let bestSupport: number[] = Array.from({ length: h }, (_, i) => i);
    let bestDetLog = Number.POSITIVE_INFINITY;

    for (let trial = 0; trial < 5; trial++) {
      // Random subsample
      const perm = Array.from({ length: n }, (_, i) => i).sort(() => Math.random() - 0.5);
      let support = perm.slice(0, h);

      for (let step = 0; step < 5; step++) {
        const Xsub = support.map((i) => X[i] as Float64Array);
        const cov = sampleCovariance(Xsub);
        const mean = new Float64Array(p);
        for (const row of Xsub) for (let j = 0; j < p; j++) mean[j]! += (row[j] ?? 0) / Xsub.length;

        // Mahalanobis distances (approximate: use diagonal)
        const variances = Float64Array.from({ length: p }, (_, j) => (cov[j] as Float64Array)[j] ?? 1);
        const dists = X.map((row) => row.reduce((s, v, j) => s + ((v - (mean[j] ?? 0)) ** 2 / Math.max(variances[j] ?? 1, 1e-12)), 0));

        const sortedIdx = Array.from({ length: n }, (_, i) => i).sort((a, b) => (dists[a] ?? 0) - (dists[b] ?? 0));
        support = sortedIdx.slice(0, h);

        const detLog = cov.reduce((s, row, i) => s + Math.log(Math.max((row as Float64Array)[i] ?? 1e-12, 1e-12)), 0);
        if (detLog < bestDetLog) { bestDetLog = detLog; bestSupport = support; }
      }
    }

    const Xbest = bestSupport.map((i) => X[i] as Float64Array);
    this.covariance_ = sampleCovariance(Xbest);
    this.location_ = new Float64Array(p);
    for (const row of Xbest) for (let j = 0; j < p; j++) (this.location_ as Float64Array)[j]! += (row[j] ?? 0) / Xbest.length;
    this.supportIndices_ = Int32Array.from(bestSupport);
    this.precision_ = Array.from({ length: p }, (_, i) => {
      const row = new Float64Array(p);
      row[i] = 1 / Math.max((this.covariance_![i] as Float64Array)[i] ?? 1e-12, 1e-12);
      return row;
    });
    return this;
  }

  mahalanobisDistance(X: Float64Array[]): Float64Array {
    if (!this.covariance_ || !this.location_) throw new Error("Not fitted");
    const p = this.location_.length;
    return Float64Array.from(X, (row) => {
      const diff = Float64Array.from({ length: p }, (_, j) => (row[j] ?? 0) - ((this.location_ as Float64Array)[j] ?? 0));
      return Math.sqrt(diff.reduce((s, v, j) => s + v * v / Math.max((this.covariance_![j] as Float64Array)[j] ?? 1, 1e-12), 0));
    });
  }
}
