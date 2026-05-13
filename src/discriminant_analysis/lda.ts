/**
 * Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA).
 * Mirrors sklearn.discriminant_analysis.LinearDiscriminantAnalysis and
 * QuadraticDiscriminantAnalysis.
 */

import { NotFittedError } from "../exceptions.js";

function dotVec(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
  return s;
}

function matVec(M: Float64Array[], v: Float64Array): Float64Array {
  return new Float64Array(M.map((row) => dotVec(row, v)));
}

/** Solve Ax = b via Gaussian elimination. */
function solveLinear(A: Float64Array[], b: Float64Array): Float64Array {
  const n = A.length;
  const aug: number[][] = A.map((row, i) => [...Array.from(row), b[i] ?? 0]);

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs((aug[r] as number[])[col] ?? 0) > Math.abs((aug[pivotRow] as number[])[col] ?? 0)) {
        pivotRow = r;
      }
    }
    [aug[col], aug[pivotRow]] = [aug[pivotRow] as number[], aug[col] as number[]];

    const pivot = (aug[col] as number[])[col] ?? 0;
    if (Math.abs(pivot) < 1e-12) continue;

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = ((aug[r] as number[])[col] ?? 0) / pivot;
      for (let c = col; c <= n; c++) {
        (aug[r] as number[])[c] = ((aug[r] as number[])[c] ?? 0) - factor * ((aug[col] as number[])[c] ?? 0);
      }
    }
  }

  const result = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const pivot = (aug[i] as number[])[i] ?? 0;
    result[i] = pivot !== 0 ? ((aug[i] as number[])[n] ?? 0) / pivot : 0;
  }
  return result;
}

export class LinearDiscriminantAnalysis {
  nComponents: number | null;
  solverTol: number;

  coef_: Float64Array[] | null = null;
  intercept_: Float64Array | null = null;
  classes_: Float64Array | null = null;
  means_: Float64Array[] | null = null;
  scalings_: Float64Array[] | null = null;
  priors_: Float64Array | null = null;

  constructor(options: { nComponents?: number; solverTol?: number } = {}) {
    this.nComponents = options.nComponents ?? null;
    this.solverTol = options.solverTol ?? 1e-4;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const uniqueClasses = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.classes_ = new Float64Array(uniqueClasses);
    const nClasses = uniqueClasses.length;
    const classToIdx = new Map(uniqueClasses.map((c, i) => [c, i]));

    // Compute class means and priors
    const means: Float64Array[] = Array.from({ length: nClasses }, () => new Float64Array(p));
    const counts = new Int32Array(nClasses);

    for (let i = 0; i < n; i++) {
      const c = classToIdx.get(y[i] ?? 0) ?? 0;
      counts[c] = (counts[c] ?? 0) + 1;
      const xi = X[i] ?? new Float64Array(p);
      const mean = means[c] ?? new Float64Array(p);
      for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) + (xi[j] ?? 0);
    }
    for (let c = 0; c < nClasses; c++) {
      const cnt = counts[c] ?? 1;
      const mean = means[c] ?? new Float64Array(p);
      for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) / cnt;
    }

    this.means_ = means;
    this.priors_ = new Float64Array(nClasses);
    for (let c = 0; c < nClasses; c++) {
      this.priors_[c] = (counts[c] ?? 0) / n;
    }

    // Compute within-class scatter matrix (pooled covariance)
    const Sw: Float64Array[] = Array.from({ length: p }, () => new Float64Array(p));
    for (let i = 0; i < n; i++) {
      const c = classToIdx.get(y[i] ?? 0) ?? 0;
      const xi = X[i] ?? new Float64Array(p);
      const mean = means[c] ?? new Float64Array(p);
      const diff = new Float64Array(p);
      for (let j = 0; j < p; j++) diff[j] = (xi[j] ?? 0) - (mean[j] ?? 0);
      for (let j = 0; j < p; j++) {
        const sw = Sw[j] ?? new Float64Array(p);
        for (let k = 0; k < p; k++) {
          sw[k] = (sw[k] ?? 0) + (diff[j] ?? 0) * (diff[k] ?? 0);
        }
      }
    }

    // Add regularization
    for (let j = 0; j < p; j++) {
      const sw = Sw[j] ?? new Float64Array(p);
      sw[j] = (sw[j] ?? 0) + this.solverTol * n;
    }

    // Compute coefficients: coef = Sw^{-1} (mu_1 - mu_0) for binary case
    // For multi-class, compute coef for each class
    this.coef_ = [];
    this.intercept_ = new Float64Array(nClasses);

    for (let c = 0; c < nClasses; c++) {
      const meanC = means[c] ?? new Float64Array(p);
      const coefC = solveLinear(Sw, meanC);
      this.coef_.push(coefC);
      const prior = (this.priors_[c] ?? 0);
      let dotMeanCCoef = dotVec(meanC, coefC);
      this.intercept_[c] = -0.5 * dotMeanCCoef + Math.log(prior + 1e-10);
    }

    return this;
  }

  decisionFunction(X: Float64Array[]): Float64Array[] {
    if (this.coef_ === null) throw new NotFittedError("LinearDiscriminantAnalysis");
    return X.map((xi) => {
      return new Float64Array(
        (this.coef_ as Float64Array[]).map((coefC, c) =>
          dotVec(xi, coefC) + ((this.intercept_ as Float64Array)[c] ?? 0),
        ),
      );
    });
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.classes_ === null) throw new NotFittedError("LinearDiscriminantAnalysis");
    const classes = this.classes_;
    const decisions = this.decisionFunction(X);
    return new Float64Array(
      decisions.map((d) => {
        let maxIdx = 0;
        let maxVal = d[0] ?? -Infinity;
        for (let c = 1; c < d.length; c++) {
          if ((d[c] ?? -Infinity) > maxVal) {
            maxVal = d[c] ?? -Infinity;
            maxIdx = c;
          }
        }
        return classes[maxIdx] ?? 0;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (pred[i] === y[i]) correct++;
    }
    return correct / y.length;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return this.decisionFunction(X);
  }
}

export class QuadraticDiscriminantAnalysis {
  regParam: number;

  classes_: Float64Array | null = null;
  means_: Float64Array[] | null = null;
  covariances_: Float64Array[][] | null = null;
  priors_: Float64Array | null = null;

  constructor(options: { regParam?: number } = {}) {
    this.regParam = options.regParam ?? 0.0;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = (X[0] ?? new Float64Array(0)).length;
    const uniqueClasses = Array.from(new Set(Array.from(y))).sort((a, b) => a - b);
    this.classes_ = new Float64Array(uniqueClasses);
    const nClasses = uniqueClasses.length;
    const classToIdx = new Map(uniqueClasses.map((c, i) => [c, i]));

    const means: Float64Array[] = Array.from({ length: nClasses }, () => new Float64Array(p));
    const covs: Float64Array[][] = Array.from({ length: nClasses }, () =>
      Array.from({ length: p }, () => new Float64Array(p)),
    );
    const counts = new Int32Array(nClasses);

    for (let i = 0; i < n; i++) {
      const c = classToIdx.get(y[i] ?? 0) ?? 0;
      counts[c] = (counts[c] ?? 0) + 1;
      const xi = X[i] ?? new Float64Array(p);
      const mean = means[c] ?? new Float64Array(p);
      for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) + (xi[j] ?? 0);
    }
    for (let c = 0; c < nClasses; c++) {
      const cnt = counts[c] ?? 1;
      const mean = means[c] ?? new Float64Array(p);
      for (let j = 0; j < p; j++) mean[j] = (mean[j] ?? 0) / cnt;
    }

    for (let i = 0; i < n; i++) {
      const c = classToIdx.get(y[i] ?? 0) ?? 0;
      const xi = X[i] ?? new Float64Array(p);
      const mean = means[c] ?? new Float64Array(p);
      const cov = covs[c] ?? [];
      const diff = new Float64Array(p);
      for (let j = 0; j < p; j++) diff[j] = (xi[j] ?? 0) - (mean[j] ?? 0);
      for (let j = 0; j < p; j++) {
        const row = cov[j] ?? new Float64Array(p);
        for (let k = 0; k < p; k++) {
          row[k] = (row[k] ?? 0) + (diff[j] ?? 0) * (diff[k] ?? 0);
        }
      }
    }

    for (let c = 0; c < nClasses; c++) {
      const cnt = counts[c] ?? 1;
      const cov = covs[c] ?? [];
      for (let j = 0; j < p; j++) {
        const row = cov[j] ?? new Float64Array(p);
        for (let k = 0; k < p; k++) {
          row[k] = (row[k] ?? 0) / cnt;
          if (j === k) row[k] = (row[k] ?? 0) + this.regParam;
        }
      }
    }

    this.means_ = means;
    this.covariances_ = covs;
    this.priors_ = new Float64Array(nClasses);
    for (let c = 0; c < nClasses; c++) {
      this.priors_[c] = (counts[c] ?? 0) / n;
    }

    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this.classes_ === null) throw new NotFittedError("QuadraticDiscriminantAnalysis");
    const classes = this.classes_;
    const nClasses = classes.length;
    const p = (X[0] ?? new Float64Array(0)).length;

    return new Float64Array(
      X.map((xi) => {
        let maxScore = -Infinity;
        let maxIdx = 0;
        for (let c = 0; c < nClasses; c++) {
          const mean = (this.means_ as Float64Array[])[c] ?? new Float64Array(p);
          const cov = (this.covariances_ as Float64Array[][])[c] ?? [];
          const prior = (this.priors_ as Float64Array)[c] ?? 0;

          const diff = new Float64Array(p);
          for (let j = 0; j < p; j++) diff[j] = (xi[j] ?? 0) - (mean[j] ?? 0);

          const solved = solveLinear(cov.length > 0 ? cov as Float64Array[] : [new Float64Array(p)], diff);
          let mahal = dotVec(diff, solved);

          const score = -0.5 * mahal + Math.log(prior + 1e-10);
          if (score > maxScore) {
            maxScore = score;
            maxIdx = c;
          }
        }
        return classes[maxIdx] ?? 0;
      }),
    );
  }

  score(X: Float64Array[], y: Float64Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (pred[i] === y[i]) correct++;
    }
    return correct / y.length;
  }
}

void matVec; // suppress unused
