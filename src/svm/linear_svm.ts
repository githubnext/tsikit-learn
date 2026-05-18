/**
 * LinearSVC, LinearSVR, and OneClassSVM — linear SVM variants
 * Ported from sklearn.svm
 */

export interface LinearSVCOptions {
  penalty?: "l1" | "l2";
  loss?: "hinge" | "squared_hinge";
  dual?: boolean;
  tol?: number;
  C?: number;
  multiClass?: "ovr" | "crammer_singer";
  fitIntercept?: boolean;
  interceptScaling?: number;
  classWeight?: Record<number, number> | "balanced" | null;
  verbose?: number;
  randomState?: number | null;
  maxIter?: number;
}

export class LinearSVC {
  penalty: "l1" | "l2";
  loss: "hinge" | "squared_hinge";
  C: number;
  tol: number;
  fitIntercept: boolean;
  maxIter: number;

  private coef_: Float64Array | null = null;
  private intercept_: number = 0;
  private classes_: Int32Array | null = null;
  private nFeatures_: number = 0;

  constructor(options: LinearSVCOptions = {}) {
    this.penalty = options.penalty ?? "l2";
    this.loss = options.loss ?? "squared_hinge";
    this.C = options.C ?? 1.0;
    this.tol = options.tol ?? 1e-4;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    this.nFeatures_ = nFeatures;

    const classSet = new Set<number>();
    for (let i = 0; i < nSamples; i++) {
      classSet.add(y[i] ?? 0);
    }
    this.classes_ = new Int32Array(Array.from(classSet).sort((a, b) => a - b));

    // Coordinate-descent style pegasos for binary SVC (hinge loss, L2 reg)
    const coef = new Float64Array(nFeatures);
    let intercept = 0.0;
    const lr0 = 0.1;
    const lambda = 1.0 / (this.C * nSamples);

    for (let iter = 0; iter < this.maxIter; iter++) {
      const lr = lr0 / (1 + lambda * lr0 * (iter + 1));
      for (let i = 0; i < nSamples; i++) {
        const xi = X[i]!;
        const yi = (y[i] ?? 0) === (this.classes_[0] ?? 0) ? -1 : 1;
        let dot = intercept;
        for (let j = 0; j < nFeatures; j++) {
          dot += coef[j]! * (xi[j] ?? 0);
        }
        const margin = yi * dot;
        if (margin < 1) {
          for (let j = 0; j < nFeatures; j++) {
            coef[j]! += lr * (yi * (xi[j] ?? 0) - 2 * lambda * (coef[j] ?? 0));
          }
          if (this.fitIntercept) {
            intercept += lr * yi;
          }
        } else {
          for (let j = 0; j < nFeatures; j++) {
            coef[j]! -= lr * 2 * lambda * (coef[j] ?? 0);
          }
        }
      }
    }

    this.coef_ = coef;
    this.intercept_ = intercept;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.coef_ || !this.classes_) {
      throw new Error("Not fitted");
    }
    const nSamples = X.length;
    const result = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      const xi = X[i]!;
      let dot = this.intercept_;
      for (let j = 0; j < this.nFeatures_; j++) {
        dot += (this.coef_[j] ?? 0) * (xi[j] ?? 0);
      }
      result[i] = dot >= 0 ? (this.classes_[1] ?? 1) : (this.classes_[0] ?? 0);
    }
    return result;
  }

  score(X: Float64Array[], y: Int32Array): number {
    const yPred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if ((yPred[i] ?? 0) === (y[i] ?? 0)) correct++;
    }
    return correct / y.length;
  }

  get coef(): Float64Array {
    if (!this.coef_) throw new Error("Not fitted");
    return this.coef_;
  }

  get intercept(): number {
    return this.intercept_;
  }
}

export interface LinearSVROptions {
  epsilon?: number;
  tol?: number;
  C?: number;
  loss?: "epsilon_insensitive" | "squared_epsilon_insensitive";
  fitIntercept?: boolean;
  maxIter?: number;
}

export class LinearSVR {
  epsilon: number;
  C: number;
  tol: number;
  fitIntercept: boolean;
  maxIter: number;

  private coef_: Float64Array | null = null;
  private intercept_: number = 0;
  private nFeatures_: number = 0;

  constructor(options: LinearSVROptions = {}) {
    this.epsilon = options.epsilon ?? 0.0;
    this.C = options.C ?? 1.0;
    this.tol = options.tol ?? 1e-4;
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 1000;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const nSamples = X.length;
    const nFeatures = X[0]?.length ?? 0;
    this.nFeatures_ = nFeatures;

    const coef = new Float64Array(nFeatures);
    let intercept = 0.0;
    const lambda = 1.0 / (this.C * nSamples);
    const lr0 = 0.01;

    for (let iter = 0; iter < this.maxIter; iter++) {
      const lr = lr0 / (1 + lambda * lr0 * iter);
      for (let i = 0; i < nSamples; i++) {
        const xi = X[i]!;
        let dot = intercept;
        for (let j = 0; j < nFeatures; j++) {
          dot += (coef[j] ?? 0) * (xi[j] ?? 0);
        }
        const residual = (y[i] ?? 0) - dot;
        if (Math.abs(residual) > this.epsilon) {
          const sign = residual > 0 ? 1 : -1;
          for (let j = 0; j < nFeatures; j++) {
            coef[j]! += lr * (sign * (xi[j] ?? 0) - 2 * lambda * (coef[j] ?? 0));
          }
          if (this.fitIntercept) {
            intercept += lr * sign;
          }
        }
      }
    }

    this.coef_ = coef;
    this.intercept_ = intercept;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (!this.coef_) throw new Error("Not fitted");
    const nSamples = X.length;
    const result = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      const xi = X[i]!;
      let dot = this.intercept_;
      for (let j = 0; j < this.nFeatures_; j++) {
        dot += (this.coef_[j] ?? 0) * (xi[j] ?? 0);
      }
      result[i] = dot;
    }
    return result;
  }

  score(X: Float64Array[], y: Float64Array): number {
    const yPred = this.predict(X);
    let ssTot = 0;
    let ssRes = 0;
    let mean = 0;
    for (let i = 0; i < y.length; i++) mean += y[i] ?? 0;
    mean /= y.length;
    for (let i = 0; i < y.length; i++) {
      const diff = (y[i] ?? 0) - mean;
      ssTot += diff * diff;
      const r = (y[i] ?? 0) - (yPred[i] ?? 0);
      ssRes += r * r;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }

  get coef(): Float64Array {
    if (!this.coef_) throw new Error("Not fitted");
    return this.coef_;
  }
}

export interface OneClassSVMOptions {
  kernel?: "rbf" | "linear" | "poly" | "sigmoid";
  degree?: number;
  gamma?: number | "scale" | "auto";
  nu?: number;
  tol?: number;
  maxIter?: number;
}

/**
 * One-class SVM for novelty/outlier detection.
 * Implements a simplified SGD-based approximation.
 */
export class OneClassSVM {
  kernel: "rbf" | "linear" | "poly" | "sigmoid";
  nu: number;
  tol: number;
  maxIter: number;
  private gamma_: number = 1.0;
  private gammaParam: number | "scale" | "auto";

  private supportVectors_: Float64Array[] | null = null;
  private dualCoef_: Float64Array | null = null;
  private rho_: number = 0;
  private nFeatures_: number = 0;

  constructor(options: OneClassSVMOptions = {}) {
    this.kernel = options.kernel ?? "rbf";
    this.nu = options.nu ?? 0.5;
    this.tol = options.tol ?? 1e-3;
    this.maxIter = options.maxIter ?? 100;
    this.gammaParam = options.gamma ?? "scale";
  }

  private rbfKernel(a: Float64Array, b: Float64Array): number {
    let dist = 0;
    for (let i = 0; i < a.length; i++) {
      const d = (a[i] ?? 0) - (b[i] ?? 0);
      dist += d * d;
    }
    return Math.exp(-this.gamma_ * dist);
  }

  fit(X: Float64Array[]): this {
    const nSamples = X.length;
    this.nFeatures_ = X[0]?.length ?? 0;

    if (this.gammaParam === "scale") {
      // Estimate variance
      let sumSq = 0;
      let sum = 0;
      let n = 0;
      for (const xi of X) {
        for (let j = 0; j < xi.length; j++) {
          const v = xi[j] ?? 0;
          sum += v;
          sumSq += v * v;
          n++;
        }
      }
      const mean = sum / n;
      const variance = sumSq / n - mean * mean;
      this.gamma_ = variance > 0 ? 1.0 / (this.nFeatures_ * variance) : 1.0;
    } else if (this.gammaParam === "auto") {
      this.gamma_ = 1.0 / this.nFeatures_;
    } else {
      this.gamma_ = this.gammaParam;
    }

    // Store a random subset as support vectors (simplified)
    const nSV = Math.max(1, Math.floor(this.nu * nSamples));
    this.supportVectors_ = X.slice(0, nSV);
    this.dualCoef_ = new Float64Array(nSV).fill(1.0 / nSV);

    // Compute rho (decision threshold) as mean kernel value
    let rhoSum = 0;
    for (let i = 0; i < nSamples; i++) {
      let kernelSum = 0;
      for (let s = 0; s < nSV; s++) {
        kernelSum += (this.dualCoef_[s] ?? 0) * this.rbfKernel(X[i]!, this.supportVectors_![s]!);
      }
      rhoSum += kernelSum;
    }
    this.rho_ = rhoSum / nSamples;
    return this;
  }

  decisionFunction(X: Float64Array[]): Float64Array {
    if (!this.supportVectors_ || !this.dualCoef_) throw new Error("Not fitted");
    const nSamples = X.length;
    const nSV = this.supportVectors_.length;
    const scores = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let score = 0;
      for (let s = 0; s < nSV; s++) {
        score += (this.dualCoef_[s] ?? 0) * this.rbfKernel(X[i]!, this.supportVectors_[s]!);
      }
      scores[i] = score - this.rho_;
    }
    return scores;
  }

  predict(X: Float64Array[]): Int32Array {
    const scores = this.decisionFunction(X);
    const result = new Int32Array(X.length);
    for (let i = 0; i < X.length; i++) {
      result[i] = (scores[i] ?? 0) >= 0 ? 1 : -1;
    }
    return result;
  }

  get supportVectors(): Float64Array[] {
    if (!this.supportVectors_) throw new Error("Not fitted");
    return this.supportVectors_;
  }
}
