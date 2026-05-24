/**
 * Target Encoder for categorical features.
 * Mirrors scikit-learn's preprocessing.TargetEncoder.
 */

export interface TargetEncoderOptions {
  smoothing?: number;
  cv?: number;
  shuffle?: boolean;
  randomState?: number;
}

/**
 * Encode categorical features using the mean target value for each category.
 * Smoothing is applied to avoid overfitting on rare categories.
 */
export class TargetEncoder {
  readonly smoothing: number;
  readonly cv: number;
  readonly shuffle: boolean;
  readonly randomState: number;

  private _encodings: Map<string | number, number>[] | null = null;
  private _globalMean: number = 0;
  private _nFeatures: number = 0;

  constructor(options: TargetEncoderOptions = {}) {
    this.smoothing = options.smoothing ?? 10.0;
    this.cv = options.cv ?? 5;
    this.shuffle = options.shuffle ?? true;
    this.randomState = options.randomState ?? 42;
  }

  fit(
    X: Array<Array<string | number>>,
    y: Float64Array,
  ): this {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;
    this._nFeatures = nFeatures;
    this._globalMean = Array.from(y).reduce((s, v) => s + v, 0) / n;
    this._encodings = [];

    for (let j = 0; j < nFeatures; j++) {
      const enc = new Map<string | number, number>();
      const cats = Array.from(new Set(X.map((row) => row[j] ?? "")));

      for (const cat of cats) {
        const catY: number[] = [];
        for (let i = 0; i < n; i++) {
          if (X[i]?.[j] === cat) catY.push(y[i] ?? 0);
        }
        const catMean = catY.reduce((s, v) => s + v, 0) / catY.length;
        const catN = catY.length;
        // Smoothing: blend cat mean with global mean
        const lambda = catN / (catN + this.smoothing);
        enc.set(cat, lambda * catMean + (1 - lambda) * this._globalMean);
      }
      this._encodings.push(enc);
    }
    return this;
  }

  transform(X: Array<Array<string | number>>): Float64Array[] {
    if (this._encodings === null) throw new Error("TargetEncoder must be fitted first");
    return X.map((row) =>
      Float64Array.from({ length: this._nFeatures }, (_, j) => {
        const cat = row[j] ?? "";
        return this._encodings![j]?.get(cat) ?? this._globalMean;
      }),
    );
  }

  fitTransform(
    X: Array<Array<string | number>>,
    y: Float64Array,
  ): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

/**
 * Leave-one-out encoding for target encoding to avoid data leakage.
 */
export class LeaveOneOutEncoder {
  private _encodings: Map<string | number, number[]>[] | null = null;
  private _globalMean: number = 0;
  private _nFeatures: number = 0;

  fit(X: Array<Array<string | number>>, y: Float64Array): this {
    const n = X.length;
    const nFeatures = X[0]?.length ?? 0;
    this._nFeatures = nFeatures;
    this._globalMean = Array.from(y).reduce((s, v) => s + v, 0) / n;
    this._encodings = [];

    for (let j = 0; j < nFeatures; j++) {
      const enc = new Map<string | number, number[]>();
      for (let i = 0; i < n; i++) {
        const cat = X[i]?.[j] ?? "";
        if (!enc.has(cat)) enc.set(cat, []);
        enc.get(cat)!.push(y[i] ?? 0);
      }
      this._encodings.push(enc);
    }
    return this;
  }

  transformTrain(X: Array<Array<string | number>>, y: Float64Array): Float64Array[] {
    if (this._encodings === null) throw new Error("LeaveOneOutEncoder must be fitted first");
    return X.map((row, i) =>
      Float64Array.from({ length: this._nFeatures }, (_, j) => {
        const cat = row[j] ?? "";
        const catY = this._encodings![j]?.get(cat) ?? [];
        const yi = y[i] ?? 0;
        const sum = catY.reduce((s, v) => s + v, 0) - yi;
        const cnt = catY.length - 1;
        return cnt > 0 ? sum / cnt : this._globalMean;
      }),
    );
  }

  transform(X: Array<Array<string | number>>): Float64Array[] {
    if (this._encodings === null) throw new Error("LeaveOneOutEncoder must be fitted first");
    return X.map((row) =>
      Float64Array.from({ length: this._nFeatures }, (_, j) => {
        const cat = row[j] ?? "";
        const catY = this._encodings![j]?.get(cat) ?? [];
        return catY.length > 0 ? catY.reduce((s, v) => s + v, 0) / catY.length : this._globalMean;
      }),
    );
  }
}
