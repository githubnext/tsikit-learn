/**
 * Extended linear models: Bayesian Ridge extensions, Orthogonal Matching Pursuit CV ext
 */
export class BayesianRidgeExt {
  private alpha1_: number;
  private alpha2_: number;
  private lambda1_: number;
  private lambda2_: number;
  private alpha_: number;
  private lambda_: number;
  private coef_: Float64Array | null = null;
  private intercept_: number = 0;
  private sigmaInv_: Float64Array | null = null;
  private nIter_: number = 0;
  private fitted_ = false;

  constructor(options: {
    nIter?: number;
    tol?: number;
    alpha1?: number;
    alpha2?: number;
    lambda1?: number;
    lambda2?: number;
    fitIntercept?: boolean;
    computeScore?: boolean;
  } = {}) {
    this.alpha1_ = options.alpha1 ?? 1e-6;
    this.alpha2_ = options.alpha2 ?? 1e-6;
    this.lambda1_ = options.lambda1 ?? 1e-6;
    this.lambda2_ = options.lambda2 ?? 1e-6;
    this.alpha_ = 1.0;
    this.lambda_ = 1.0;
    this.nIter_ = options.nIter ?? 300;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const tol = 1e-3;
    let alpha = this.alpha_;
    let lambda = this.lambda_;

    // Build X matrix and XtX
    const Xt = new Float64Array(p * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < p; j++) {
        Xt[j * n + i] = X[i]?.[j] ?? 0;
      }
    }
    const XtX = new Float64Array(p * p);
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < p; k++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += (Xt[j * n + i] ?? 0) * (Xt[k * n + i] ?? 0);
        XtX[j * p + k] = s;
      }
    }
    const Xty = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      for (let i = 0; i < n; i++) Xty[j] = (Xty[j] ?? 0) + (Xt[j * n + i] ?? 0) * (y[i] ?? 0);
    }

    for (let iter = 0; iter < this.nIter_; iter++) {
      // Solve (alpha * I + lambda * XtX) w = lambda * Xty
      const A = new Float64Array(p * p);
      for (let j = 0; j < p; j++) {
        for (let k = 0; k < p; k++) A[j * p + k] = lambda * (XtX[j * p + k] ?? 0) + (j === k ? alpha : 0);
      }
      const w = this._solveLinear(A, p, Xty.map(v => lambda * v));

      // Update hyperparameters
      let gamma = 0;
      for (let j = 0; j < p; j++) gamma += (w[j] ?? 0) ** 2;
      const residNorm = this._residualNorm(X, y, w, n, p);
      const alphaNew = (p - gamma * alpha) / (this.alpha1_ + this.alpha2_ * gamma);
      const lambdaNew = (n - gamma) / (this.lambda1_ + 0.5 * residNorm + this.lambda2_);

      if (Math.abs(alphaNew - alpha) < tol && Math.abs(lambdaNew - lambda) < tol) break;
      alpha = alphaNew;
      lambda = lambdaNew;
    }

    this.coef_ = this._solveLinear(
      this._buildA(XtX, p, alpha, lambda),
      p,
      Xty.map(v => lambda * v)
    );
    this.alpha_ = alpha;
    this.lambda_ = lambda;
    this.fitted_ = true;
    return this;
  }

  private _buildA(XtX: Float64Array, p: number, alpha: number, lambda: number): Float64Array {
    const A = new Float64Array(p * p);
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < p; k++) A[j * p + k] = lambda * (XtX[j * p + k] ?? 0) + (j === k ? alpha : 0);
    }
    return A;
  }

  private _residualNorm(X: Float64Array[], y: Float64Array, w: Float64Array, n: number, p: number): number {
    let s = 0;
    for (let i = 0; i < n; i++) {
      let pred = 0;
      for (let j = 0; j < p; j++) pred += (X[i]?.[j] ?? 0) * (w[j] ?? 0);
      s += ((y[i] ?? 0) - pred) ** 2;
    }
    return s;
  }

  private _solveLinear(A: Float64Array, p: number, b: Float64Array | number[]): Float64Array {
    // Gaussian elimination
    const mat = new Float64Array(p * (p + 1));
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < p; k++) mat[j * (p + 1) + k] = A[j * p + k] ?? 0;
      mat[j * (p + 1) + p] = b[j] ?? 0;
    }
    for (let col = 0; col < p; col++) {
      let maxRow = col;
      let maxVal = Math.abs(mat[col * (p + 1) + col] ?? 0);
      for (let row = col + 1; row < p; row++) {
        const v = Math.abs(mat[row * (p + 1) + col] ?? 0);
        if (v > maxVal) { maxVal = v; maxRow = row; }
      }
      for (let k = 0; k <= p; k++) {
        const tmp = mat[col * (p + 1) + k] ?? 0;
        mat[col * (p + 1) + k] = mat[maxRow * (p + 1) + k] ?? 0;
        mat[maxRow * (p + 1) + k] = tmp;
      }
      const pivot = mat[col * (p + 1) + col] ?? 1e-10;
      for (let row = col + 1; row < p; row++) {
        const factor = (mat[row * (p + 1) + col] ?? 0) / pivot;
        for (let k = col; k <= p; k++) mat[row * (p + 1) + k] = (mat[row * (p + 1) + k] ?? 0) - factor * (mat[col * (p + 1) + k] ?? 0);
      }
    }
    const x = new Float64Array(p);
    for (let j = p - 1; j >= 0; j--) {
      let s = mat[j * (p + 1) + p] ?? 0;
      for (let k = j + 1; k < p; k++) s -= (mat[j * (p + 1) + k] ?? 0) * (x[k] ?? 0);
      x[j] = s / (mat[j * (p + 1) + j] ?? 1e-10);
    }
    return x;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_ || !this.coef_) throw new Error('Not fitted');
    const coef = this.coef_;
    return new Float64Array(X.map(row => row.reduce((s, v, j) => s + v * (coef[j] ?? 0), this.intercept_)));
  }

  get coef(): Float64Array { return this.coef_ ?? new Float64Array(0); }
  get alpha(): number { return this.alpha_; }
  get lambda(): number { return this.lambda_; }
}

export class OrthogonalMatchingPursuitCVExt {
  private bestNComp_: number = 0;
  private fitted_ = false;
  private coef_: Float64Array | null = null;

  constructor(private maxComp: number = 10, private cv: number = 5) {}

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const foldSize = Math.floor(n / this.cv);
    let bestScore = Number.NEGATIVE_INFINITY;

    for (let k = 1; k <= Math.min(this.maxComp, X[0]?.length ?? 1); k++) {
      let cvScore = 0;
      for (let fold = 0; fold < this.cv; fold++) {
        const valStart = fold * foldSize;
        const valEnd = Math.min(valStart + foldSize, n);
        const trainX: Float64Array[] = [], trainY: number[] = [], valX: Float64Array[] = [], valY: number[] = [];
        for (let i = 0; i < n; i++) {
          if (i >= valStart && i < valEnd) { valX.push(X[i]!); valY.push(y[i] ?? 0); }
          else { trainX.push(X[i]!); trainY.push(y[i] ?? 0); }
        }
        const coef = this._omp(trainX, new Float64Array(trainY), k);
        let ss = 0, st = 0, mean = valY.reduce((a, b) => a + b, 0) / valY.length;
        for (let i = 0; i < valX.length; i++) {
          const pred = (valX[i]!).reduce((s, v, j) => s + v * (coef[j] ?? 0), 0);
          ss += ((valY[i] ?? 0) - pred) ** 2;
          st += ((valY[i] ?? 0) - mean) ** 2;
        }
        cvScore += 1 - ss / (st + 1e-10);
      }
      if (cvScore > bestScore) { bestScore = cvScore; this.bestNComp_ = k; }
    }
    this.coef_ = this._omp(X, y, this.bestNComp_);
    this.fitted_ = true;
    return this;
  }

  private _omp(X: Float64Array[], y: Float64Array, k: number): Float64Array {
    const n = X.length, p = X[0]?.length ?? 0;
    const support: number[] = [];
    let residual = new Float64Array(y);
    for (let iter = 0; iter < k; iter++) {
      let bestJ = 0, bestCorr = -1;
      for (let j = 0; j < p; j++) {
        if (support.includes(j)) continue;
        let corr = 0;
        for (let i = 0; i < n; i++) corr += (X[i]?.[j] ?? 0) * (residual[i] ?? 0);
        if (Math.abs(corr) > bestCorr) { bestCorr = Math.abs(corr); bestJ = j; }
      }
      support.push(bestJ);
    }
    // Least squares on support
    const subX = X.map(row => new Float64Array(support.map(j => row[j] ?? 0)));
    const coefSub = this._leastSquares(subX, y, support.length);
    const coef = new Float64Array(p);
    for (let i = 0; i < support.length; i++) coef[support[i]!] = coefSub[i] ?? 0;
    return coef;
  }

  private _leastSquares(X: Float64Array[], y: Float64Array, p: number): Float64Array {
    const n = X.length;
    const XtX = new Float64Array(p * p);
    const Xty = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < p; k++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += (X[i]?.[j] ?? 0) * (X[i]?.[k] ?? 0);
        XtX[j * p + k] = s;
      }
      for (let i = 0; i < n; i++) Xty[j]! += (X[i]?.[j] ?? 0) * (y[i] ?? 0);
    }
    for (let j = 0; j < p; j++) XtX[j * p + j]! += 1e-10;
    return this._solve(XtX, p, Xty);
  }

  private _solve(A: Float64Array, p: number, b: Float64Array): Float64Array {
    const mat = Array.from({ length: p }, (_, j) => {
      const row = new Float64Array(p + 1);
      for (let k = 0; k < p; k++) row[k] = A[j * p + k] ?? 0;
      row[p] = b[j] ?? 0;
      return row;
    });
    for (let col = 0; col < p; col++) {
      let maxRow = col;
      for (let row = col + 1; row < p; row++) if (Math.abs(mat[row]?.[col] ?? 0) > Math.abs(mat[maxRow]?.[col] ?? 0)) maxRow = row;
      const tmp = mat[col]!; mat[col] = mat[maxRow]!; mat[maxRow] = tmp;
      const pivot = mat[col]?.[col] ?? 1e-10;
      for (let row = col + 1; row < p; row++) {
        const f = (mat[row]?.[col] ?? 0) / pivot;
        for (let k = col; k <= p; k++) mat[row]![k] = (mat[row]?.[k] ?? 0) - f * (mat[col]?.[k] ?? 0);
      }
    }
    const x = new Float64Array(p);
    for (let j = p - 1; j >= 0; j--) {
      let s = mat[j]?.[p] ?? 0;
      for (let k = j + 1; k < p; k++) s -= (mat[j]?.[k] ?? 0) * (x[k] ?? 0);
      x[j] = s / (mat[j]?.[j] ?? 1e-10);
    }
    return x;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.fitted_ || !this.coef_) throw new Error('Not fitted');
    const coef = this.coef_;
    return new Float64Array(X.map(row => row.reduce((s, v, j) => s + v * (coef[j] ?? 0), 0)));
  }

  get bestNComponents(): number { return this.bestNComp_; }
  get coef(): Float64Array { return this.coef_ ?? new Float64Array(0); }
}

export class MultiTaskElasticNetCVExt {
  private alphas_: Float64Array | null = null;
  private bestAlpha_: number = 1.0;
  private fitted_ = false;
  private coef_: Float64Array[] | null = null;

  constructor(private l1Ratio: number = 0.5, private cv: number = 5, private nAlphas: number = 10) {}

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const alphas = this._generateAlphas(X, Y);
    this.alphas_ = new Float64Array(alphas);
    const n = X.length;
    const foldSize = Math.floor(n / this.cv);
    let bestScore = Number.NEGATIVE_INFINITY;

    for (const alpha of alphas) {
      let cvScore = 0;
      for (let fold = 0; fold < this.cv; fold++) {
        const valStart = fold * foldSize;
        const valEnd = Math.min(valStart + foldSize, n);
        const trainX: Float64Array[] = [], trainY: Float64Array[] = [], valX: Float64Array[] = [], valY: Float64Array[] = [];
        for (let i = 0; i < n; i++) {
          if (i >= valStart && i < valEnd) { valX.push(X[i]!); valY.push(Y[i]!); }
          else { trainX.push(X[i]!); trainY.push(Y[i]!); }
        }
        const coef = this._fitElasticNet(trainX, trainY, alpha);
        let ss = 0, st = 0;
        for (let i = 0; i < valX.length; i++) {
          for (let t = 0; t < (valY[0]?.length ?? 0); t++) {
            const pred = (valX[i]!).reduce((s, v, j) => s + v * (coef[t]?.[j] ?? 0), 0);
            ss += ((valY[i]?.[t] ?? 0) - pred) ** 2;
            st += (valY[i]?.[t] ?? 0) ** 2;
          }
        }
        cvScore -= ss;
      }
      if (cvScore > bestScore) { bestScore = cvScore; this.bestAlpha_ = alpha; }
    }

    this.coef_ = this._fitElasticNet(X, Y, this.bestAlpha_);
    this.fitted_ = true;
    return this;
  }

  private _generateAlphas(X: Float64Array[], Y: Float64Array[]): number[] {
    const alphas: number[] = [];
    const alphaMax = 1.0;
    for (let i = 0; i < this.nAlphas; i++) {
      alphas.push(alphaMax * Math.pow(0.001, i / (this.nAlphas - 1)));
    }
    return alphas;
  }

  private _fitElasticNet(X: Float64Array[], Y: Float64Array[], alpha: number): Float64Array[] {
    const p = X[0]?.length ?? 0;
    const T = Y[0]?.length ?? 0;
    const coef: Float64Array[] = Array.from({ length: T }, () => new Float64Array(p));
    const l1 = alpha * this.l1Ratio;
    const l2 = alpha * (1 - this.l1Ratio);
    for (let t = 0; t < T; t++) {
      const yt = new Float64Array(Y.map(row => row[t] ?? 0));
      const ct = coef[t]!;
      for (let iter = 0; iter < 100; iter++) {
        for (let j = 0; j < p; j++) {
          const xj = new Float64Array(X.map(row => row[j] ?? 0));
          let rho = 0;
          for (let i = 0; i < X.length; i++) {
            let pred = 0;
            for (let k = 0; k < p; k++) if (k !== j) pred += (X[i]?.[k] ?? 0) * (ct[k] ?? 0);
            rho += (xj[i] ?? 0) * ((yt[i] ?? 0) - pred);
          }
          const denom = X.length + l2;
          ct[j] = Math.sign(rho) * Math.max(Math.abs(rho) - l1, 0) / denom;
        }
      }
    }
    return coef;
  }

  predict(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_ || !this.coef_) throw new Error('Not fitted');
    return this.coef_.map(coef => new Float64Array(X.map(row => row.reduce((s, v, j) => s + v * (coef[j] ?? 0), 0))));
  }

  get bestAlpha(): number { return this.bestAlpha_; }
  get alphas(): Float64Array { return this.alphas_ ?? new Float64Array(0); }
}
