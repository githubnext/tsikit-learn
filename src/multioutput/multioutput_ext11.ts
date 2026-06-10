/**
 * Classifier chains and regressor chains for multi-output learning.
 */

export interface ChainableClassifier {
  fit(X: Float64Array[], y: Int32Array): this;
  predict(X: Float64Array[]): Int32Array;
  predictProba?(X: Float64Array[]): Float64Array[];
}

export interface ChainableRegressor {
  fit(X: Float64Array[], y: Float64Array): this;
  predict(X: Float64Array[]): Float64Array;
}

export class ClassifierChain {
  private estimators_: ChainableClassifier[] = [];
  private order_: number[];
  private fitted_ = false;

  constructor(
    private estimatorFactory: () => ChainableClassifier,
    private nTargets: number,
    order?: number[]
  ) {
    this.order_ = order ?? Array.from({ length: nTargets }, (_, i) => i);
  }

  fit(X: Float64Array[], Y: Int32Array[]): this {
    const n = X.length;
    let augX = X.map(row => new Float64Array(row));
    this.estimators_ = [];
    for (const targetIdx of this.order_) {
      const y = new Int32Array(n).map((_, i) => Y[i]![targetIdx] ?? 0);
      const est = this.estimatorFactory();
      est.fit(augX, y);
      this.estimators_.push(est);
      // Augment X with predictions for this target
      const preds = est.predict(augX);
      augX = augX.map((row, i) => {
        const newRow = new Float64Array(row.length + 1);
        newRow.set(row);
        newRow[row.length] = preds[i] ?? 0;
        return newRow;
      });
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const n = X.length;
    const predictions: Int32Array[] = Array.from({ length: this.nTargets }, () => new Int32Array(n));
    let augX = X.map(row => new Float64Array(row));
    for (let step = 0; step < this.order_.length; step++) {
      const targetIdx = this.order_[step]!;
      const est = this.estimators_[step]!;
      const preds = est.predict(augX);
      for (let i = 0; i < n; i++) predictions[targetIdx]![i] = preds[i] ?? 0;
      augX = augX.map((row, i) => {
        const newRow = new Float64Array(row.length + 1);
        newRow.set(row);
        newRow[row.length] = preds[i] ?? 0;
        return newRow;
      });
    }
    return predictions;
  }
}

export class RegressorChain {
  private estimators_: ChainableRegressor[] = [];
  private order_: number[];
  private fitted_ = false;

  constructor(
    private estimatorFactory: () => ChainableRegressor,
    private nTargets: number,
    order?: number[]
  ) {
    this.order_ = order ?? Array.from({ length: nTargets }, (_, i) => i);
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const n = X.length;
    let augX = X.map(row => new Float64Array(row));
    this.estimators_ = [];
    for (const targetIdx of this.order_) {
      const y = new Float64Array(n).map((_, i) => Y[i]![targetIdx] ?? 0);
      const est = this.estimatorFactory();
      est.fit(augX, y);
      this.estimators_.push(est);
      const preds = est.predict(augX);
      augX = augX.map((row, i) => {
        const newRow = new Float64Array(row.length + 1);
        newRow.set(row);
        newRow[row.length] = preds[i] ?? 0;
        return newRow;
      });
    }
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const n = X.length;
    const predictions: Float64Array[] = Array.from({ length: this.nTargets }, () => new Float64Array(n));
    let augX = X.map(row => new Float64Array(row));
    for (let step = 0; step < this.order_.length; step++) {
      const targetIdx = this.order_[step]!;
      const est = this.estimators_[step]!;
      const preds = est.predict(augX);
      for (let i = 0; i < n; i++) predictions[targetIdx]![i] = preds[i] ?? 0;
      augX = augX.map((row, i) => {
        const newRow = new Float64Array(row.length + 1);
        newRow.set(row);
        newRow[row.length] = preds[i] ?? 0;
        return newRow;
      });
    }
    return predictions;
  }
}

export class MultiOutputClassifier {
  private estimators_: ChainableClassifier[] = [];
  private fitted_ = false;

  constructor(private estimatorFactory: () => ChainableClassifier) {}

  fit(X: Float64Array[], Y: Int32Array[]): this {
    const nTargets = Y[0]?.length ?? 0;
    this.estimators_ = Array.from({ length: nTargets }, (_, k) => {
      const est = this.estimatorFactory();
      const y = new Int32Array(Y.length).map((_, i) => Y[i]![k] ?? 0);
      est.fit(X, y);
      return est;
    });
    this.fitted_ = true;
    return this;
  }

  predict(X: Float64Array[]): Int32Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return this.estimators_.map(est => est.predict(X));
  }
}
