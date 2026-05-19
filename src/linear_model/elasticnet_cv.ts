/**
 * ElasticNetCV — ElasticNet with built-in cross-validation for alpha/l1_ratio.
 * Mirrors sklearn.linear_model.ElasticNetCV.
 */

export interface ElasticNetCVOptions {
  l1Ratio?: number | number[];
  eps?: number;
  nAlphas?: number;
  alphas?: Float64Array | null;
  fitIntercept?: boolean;
  maxIter?: number;
  tol?: number;
  cv?: number;
  randomState?: number | null;
}

/**
 * ElasticNet linear regression with iterative fitting along a regularization path.
 */
export class ElasticNetCV {
  l1Ratio: number | number[];
  eps: number;
  nAlphas: number;
  alphas: Float64Array | null;
  fitIntercept: boolean;
  maxIter: number;
  tol: number;
  cv: number;
  randomState: number | null;

  alpha_: number | null = null;
  l1Ratio_: number | null = null;
  coef_: Float64Array | null = null;
  intercept_: number = 0;
  msePathValues_: Float64Array[][] | null = null;
  alphasPath_: Float64Array | null = null;

  constructor(options: ElasticNetCVOptions = {}) {
    this.l1Ratio = options.l1Ratio ?? 0.5;
    this.eps = options.eps ?? 1e-3;
    this.nAlphas = options.nAlphas ?? 100;
    this.alphas = options.alphas ?? null;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-4;
    this.cv = options.cv ?? 5;
    this.randomState = options.randomState ?? null;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;

    // Center X and y if fitIntercept
    let xMean = new Float64Array(nFeatures);
    let yMean = 0;

    if (this.fitIntercept) {
      for (const row of X) {
        for (let j = 0; j < nFeatures; j++) xMean[j] = (xMean[j] ?? 0) + (row[j] ?? 0);
      }
      for (let j = 0; j < nFeatures; j++) xMean[j] = (xMean[j] ?? 0) / nSamples;
      for (let i = 0; i < nSamples; i++) yMean += y[i] ?? 0;
      yMean /= nSamples;
    }

    const Xc = X.map(row => {
      const r = new Float64Array(row);
      for (let j = 0; j < nFeatures; j++) r[j] = (r[j] ?? 0) - (xMean[j] ?? 0);
      return r;
    });
    const yc = y.map(v => v - yMean);

    // Compute alpha path
    const l1Ratios = Array.isArray(this.l1Ratio) ? this.l1Ratio : [this.l1Ratio];
    let bestAlpha = 1.0;
    let bestL1Ratio = l1Ratios[0] ?? 0.5;
    let bestMse = Infinity;

    for (const l1r of l1Ratios) {
      // Compute alpha max
      let alphaMax = 0;
      for (let j = 0; j < nFeatures; j++) {
        let corr = 0;
        for (let i = 0; i < nSamples; i++) corr += (Xc[i]?.[j] ?? 0) * (yc[i] ?? 0);
        alphaMax = Math.max(alphaMax, Math.abs(corr) / nSamples);
      }
      alphaMax = alphaMax / Math.max(l1r, 1e-10);

      const alphas = this.alphas ?? (() => {
        const arr = new Float64Array(this.nAlphas);
        const logMin = Math.log(alphaMax * this.eps);
        const logMax = Math.log(alphaMax);
        for (let i = 0; i < this.nAlphas; i++) {
          arr[i] = Math.exp(logMax - (logMax - logMin) * i / (this.nAlphas - 1));
        }
        return arr;
      })();

      // Simple CV: split into cv folds
      const foldSize = Math.floor(nSamples / this.cv);
      for (let a = 0; a < alphas.length; a++) {
        const alpha = alphas[a] ?? 1.0;
        let cvMse = 0;
        for (let f = 0; f < this.cv; f++) {
          const testStart = f * foldSize;
          const testEnd = Math.min(testStart + foldSize, nSamples);
          const trainX: Float64Array[] = [];
          const trainY: number[] = [];
          const testX: Float64Array[] = [];
          const testY: number[] = [];
          for (let i = 0; i < nSamples; i++) {
            if (i >= testStart && i < testEnd) {
              testX.push(Xc[i]!);
              testY.push(yc[i] ?? 0);
            } else {
              trainX.push(Xc[i]!);
              trainY.push(yc[i] ?? 0);
            }
          }

          // Fit ElasticNet on train fold using coordinate descent
          const coef = new Float64Array(nFeatures);
          this._fitCoordDescent(trainX, new Float64Array(trainY), coef, alpha, l1r);

          // Predict on test fold
          let mse = 0;
          for (let i = 0; i < testX.length; i++) {
            let pred = 0;
            for (let j = 0; j < nFeatures; j++) pred += (testX[i]?.[j] ?? 0) * (coef[j] ?? 0);
            mse += (pred - (testY[i] ?? 0)) ** 2;
          }
          cvMse += testX.length > 0 ? mse / testX.length : 0;
        }
        cvMse /= this.cv;
        if (cvMse < bestMse) {
          bestMse = cvMse;
          bestAlpha = alphas[a] ?? 1.0;
          bestL1Ratio = l1r;
        }
      }
    }

    this.alpha_ = bestAlpha;
    this.l1Ratio_ = bestL1Ratio;

    // Refit on all data
    this.coef_ = new Float64Array(nFeatures);
    this._fitCoordDescent(Xc, new Float64Array(yc), this.coef_, bestAlpha, bestL1Ratio);

    if (this.fitIntercept) {
      let intercept = yMean;
      for (let j = 0; j < nFeatures; j++) intercept -= (this.coef_[j] ?? 0) * (xMean[j] ?? 0);
      this.intercept_ = intercept;
    }

    return this;
  }

  private _fitCoordDescent(
    X: Float64Array[], y: Float64Array, coef: Float64Array, alpha: number, l1Ratio: number
  ): void {
    const nSamples = X.length;
    const nFeatures = coef.length;
    const alphaL1 = alpha * l1Ratio;
    const alphaL2 = alpha * (1 - l1Ratio);

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxChange = 0;
      for (let j = 0; j < nFeatures; j++) {
        // Compute residual contribution
        let rho = 0;
        for (let i = 0; i < nSamples; i++) {
          let pred = 0;
          for (let k = 0; k < nFeatures; k++) {
            if (k !== j) pred += (X[i]?.[k] ?? 0) * (coef[k] ?? 0);
          }
          rho += (X[i]?.[j] ?? 0) * ((y[i] ?? 0) - pred);
        }
        rho /= nSamples;

        // Feature norm
        let norm = alphaL2;
        for (let i = 0; i < nSamples; i++) norm += (X[i]?.[j] ?? 0) ** 2 / nSamples;

        // Soft threshold
        const oldCoef = coef[j] ?? 0;
        const sign = rho > 0 ? 1 : -1;
        coef[j] = sign * Math.max(Math.abs(rho) - alphaL1, 0) / (norm || 1e-10);

        maxChange = Math.max(maxChange, Math.abs((coef[j] ?? 0) - oldCoef));
      }
      if (maxChange < this.tol) break;
    }
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new Error("ElasticNetCV not fitted");
    const nFeatures = this.coef_.length;
    return new Float64Array(X.map(row => {
      let pred = this.intercept_;
      for (let j = 0; j < nFeatures; j++) pred += (row[j] ?? 0) * (this.coef_![j] ?? 0);
      return pred;
    }));
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    const yMean = y.reduce((s, v) => s + v, 0) / y.length;
    let ss_res = 0;
    let ss_tot = 0;
    for (let i = 0; i < y.length; i++) {
      ss_res += ((y[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
      ss_tot += ((y[i] ?? 0) - yMean) ** 2;
    }
    return ss_tot < 1e-10 ? 1 : 1 - ss_res / ss_tot;
  }
}
