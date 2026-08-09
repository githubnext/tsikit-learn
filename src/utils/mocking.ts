/**
 * Mock estimators for testing — ported from sklearn.utils._mocking
 */

export interface MockClassifierOptions {
  strategy?: "stratified" | "most_frequent" | "constant";
  constant?: number;
  randomState?: number | null;
}

/**
 * A mock classifier for use in testing pipelines and meta-estimators.
 * Always predicts based on the configured strategy.
 */
export class MockClassifier {
  strategy: "stratified" | "most_frequent" | "constant";
  constant: number;
  private classes_: Int32Array | null = null;
  private classCounts_: Int32Array | null = null;
  private mostFrequent_: number = 0;

  constructor(options: MockClassifierOptions = {}) {
    this.strategy = options.strategy ?? "most_frequent";
    this.constant = options.constant ?? 0;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    void X;
    const counts = new Map<number, number>();
    for (let i = 0; i < y.length; i++) {
      const v = y[i] ?? 0;
      counts.set(v, (counts.get(v) ?? 0) + 1);
    }
    const sorted = Array.from(counts.entries()).sort((a, b) => a[0] - b[0]);
    this.classes_ = new Int32Array(sorted.map(([k]) => k));
    this.classCounts_ = new Int32Array(sorted.map(([, v]) => v));

    let maxCount = 0;
    for (const [k, v] of counts) {
      if (v > maxCount) {
        maxCount = v;
        this.mostFrequent_ = k;
      }
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.classes_) throw new Error("Not fitted");
    const n = X.length;
    const result = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      if (this.strategy === "constant") {
        result[i] = this.constant;
      } else {
        result[i] = this.mostFrequent_;
      }
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

  get classes(): Int32Array {
    if (!this.classes_) throw new Error("Not fitted");
    return this.classes_;
  }
}

export interface MockRegressorOptions {
  strategy?: "mean" | "median" | "constant";
  constant?: number;
}

/**
 * A mock regressor for use in testing.
 */
export class MockRegressor {
  strategy: "mean" | "median" | "constant";
  constant: number;
  private prediction_: number = 0;

  constructor(options: MockRegressorOptions = {}) {
    this.strategy = options.strategy ?? "mean";
    this.constant = options.constant ?? 0;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    void X;
    if (this.strategy === "mean") {
      let sum = 0;
      for (let i = 0; i < y.length; i++) sum += y[i] ?? 0;
      this.prediction_ = sum / y.length;
    } else if (this.strategy === "median") {
      const sorted = Array.from(y).sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      this.prediction_ =
        sorted.length % 2 === 1
          ? (sorted[mid] ?? 0)
          : ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2;
    } else {
      this.prediction_ = this.constant;
    }
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    const result = new Float64Array(X.length);
    result.fill(this.prediction_);
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
      const d = (y[i] ?? 0) - mean;
      ssTot += d * d;
      const r = (y[i] ?? 0) - (yPred[i] ?? 0);
      ssRes += r * r;
    }
    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  }
}

export interface CheckingClassifierOptions {
  checkX?: ((X: Float64Array[]) => void) | null;
  checkY?: ((y: Int32Array) => void) | null;
  expectedFitParams?: string[];
}

/**
 * Classifier for testing that checks inputs match expected conditions.
 */
export class CheckingClassifier {
  checkX: ((X: Float64Array[]) => void) | null;
  checkY: ((y: Int32Array) => void) | null;
  expectedFitParams: string[];
  private fitted_: boolean = false;
  private classes_: Int32Array | null = null;

  constructor(options: CheckingClassifierOptions = {}) {
    this.checkX = options.checkX ?? null;
    this.checkY = options.checkY ?? null;
    this.expectedFitParams = options.expectedFitParams ?? [];
  }

  fit(
    X: Float64Array[],
    y: Int32Array,
    params?: Record<string, unknown>,
  ): this {
    if (this.checkX) this.checkX(X);
    if (this.checkY) this.checkY(y);

    if (params) {
      for (const p of this.expectedFitParams) {
        if (!(p in params)) {
          throw new Error(`Expected fit parameter '${p}' not found`);
        }
      }
    }

    const classSet = new Set<number>();
    for (let i = 0; i < y.length; i++) classSet.add(y[i] ?? 0);
    this.classes_ = new Int32Array(Array.from(classSet).sort((a, b) => a - b));
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_ || !this.classes_) throw new Error("Not fitted");
    if (this.checkX) this.checkX(X);
    return new Int32Array(X.length).fill(this.classes_[0] ?? 0);
  }

  get isFitted(): boolean {
    return this.fitted_;
  }

  get classes(): Int32Array {
    if (!this.classes_) throw new Error("Not fitted");
    return this.classes_;
  }
}

/**
 * A no-op transformer that passes data through unchanged.
 */
export class NoOpTransformer {
  fit(X: Float64Array[]): this {
    void X;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X;
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
