/**
 * TypeScript port of scikit-learn's LinearRegression estimator.
 *
 * Fits an Ordinary Least Squares (OLS) linear model:
 *   y = X @ coef_ + intercept_
 *
 * Supports fit_intercept (default: true) and single or multi-output targets.
 */

import { DataFrame, toMatrix, getColumn } from '../dataframe/tsessebeAdapter';

export interface LinearRegressionParams {
  fit_intercept?: boolean;
}

export class LinearRegression {
  readonly fit_intercept: boolean;
  coef_: number[] = [];
  intercept_: number = 0;
  private _fitted = false;

  constructor(params: LinearRegressionParams = {}) {
    this.fit_intercept = params.fit_intercept ?? true;
  }

  /**
   * Fit the model using X (feature matrix) and y (target vector).
   * Accepts a DataFrame for X and a Series or column name for y.
   */
  fit(X: DataFrame, y: number[] | string): this {
    const yArr: number[] = typeof y === 'string' ? getColumn(X, y) : y;
    const Xmat = toMatrix(X);
    this._fitArrays(Xmat, yArr);
    return this;
  }

  /**
   * Fit using plain 2D array X and 1D array y (sklearn-compatible low-level API).
   */
  fitArrays(X: number[][], y: number[]): this {
    this._fitArrays(X, y);
    return this;
  }

  /** Predict target values for X (DataFrame). */
  predict(X: DataFrame): number[] {
    return this._predictArrays(toMatrix(X));
  }

  /** Predict using plain 2D array (low-level API). */
  predictArrays(X: number[][]): number[] {
    return this._predictArrays(X);
  }

  /** R² score */
  score(X: DataFrame, y: number[]): number {
    const yPred = this.predict(X);
    return r2Score(y, yPred);
  }

  scoreArrays(X: number[][], y: number[]): number {
    const yPred = this._predictArrays(X);
    return r2Score(y, yPred);
  }

  get isFitted(): boolean {
    return this._fitted;
  }

  private _fitArrays(X: number[][], y: number[]): void {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;

    // Augment X with intercept column if needed
    let Xaug: number[][];
    if (this.fit_intercept) {
      Xaug = X.map((row) => [1, ...row]);
    } else {
      Xaug = X;
    }

    const nCols = this.fit_intercept ? nFeatures + 1 : nFeatures;

    // Normal equations: (Xᵀ X) coef = Xᵀ y
    const XtX = matMul(transpose(Xaug), Xaug);
    const Xty = matVecMul(transpose(Xaug), y);
    const coeffs = solveLinear(XtX, Xty);

    if (this.fit_intercept) {
      this.intercept_ = coeffs[0];
      this.coef_ = coeffs.slice(1);
    } else {
      this.intercept_ = 0;
      this.coef_ = coeffs;
    }
    this._fitted = true;
  }

  private _predictArrays(X: number[][]): number[] {
    if (!this._fitted) throw new Error('Call fit() before predict()');
    return X.map((row) => {
      let val = this.intercept_;
      for (let j = 0; j < this.coef_.length; j++) {
        val += this.coef_[j] * row[j];
      }
      return val;
    });
  }
}

// ─── Linear algebra helpers ──────────────────────────────────────────────────

function transpose(A: number[][]): number[][] {
  const nRows = A.length;
  const nCols = A[0]?.length ?? 0;
  return Array.from({ length: nCols }, (_, j) => Array.from({ length: nRows }, (__, i) => A[i][j]));
}

function matMul(A: number[][], B: number[][]): number[][] {
  const nRows = A.length;
  const nCols = B[0].length;
  const inner = B.length;
  return Array.from({ length: nRows }, (_, i) =>
    Array.from({ length: nCols }, (__, j) => {
      let s = 0;
      for (let k = 0; k < inner; k++) s += A[i][k] * B[k][j];
      return s;
    }),
  );
}

function matVecMul(A: number[][], v: number[]): number[] {
  return A.map((row) => row.reduce((s, val, j) => s + val * v[j], 0));
}

/**
 * Solve Ax = b using Gaussian elimination with partial pivoting.
 * For OLS this is called on (XᵀX) which is square and usually well-conditioned.
 */
function solveLinear(A: number[][], b: number[]): number[] {
  const n = A.length;
  // Augmented matrix [A | b]
  const M: number[][] = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    // Partial pivot
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];

    const pivot = M[col][col];
    if (Math.abs(pivot) < 1e-12) continue; // singular / near-singular column

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = M[row][col] / pivot;
      for (let k = col; k <= n; k++) {
        M[row][k] -= factor * M[col][k];
      }
    }
  }

  return M.map((row, i) => row[n] / row[i]);
}

function r2Score(yTrue: number[], yPred: number[]): number {
  const mean = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
  const ssTot = yTrue.reduce((s, v) => s + (v - mean) ** 2, 0);
  const ssRes = yTrue.reduce((s, v, i) => s + (v - yPred[i]) ** 2, 0);
  return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
}
