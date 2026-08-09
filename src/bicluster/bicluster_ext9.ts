/**
 * Bicluster Extension 9
 */

export class ContrastBicluster {
  private data_!: Float64Array;
  private fitted_ = false;

  constructor(private param1 = 0.1) {}

  fit(X: Float64Array[], y?: Float64Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    this.data_ = new Float64Array(d);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < d; j++)
        this.data_[j]! += (X[i]![j] ?? 0) * this.param1;
    if (n > 0) for (let j = 0; j < d; j++) this.data_[j]! /= n;
    void y;
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error("Not fitted");
    return X.map(x => new Float64Array(x.map((v, j) => v - (this.data_[j] ?? 0) * this.param1)));
  }

  fitTransform(X: Float64Array[], y?: Float64Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

export class DifferentialBicluster {
  private data_!: Float64Array;
  private fitted_ = false;

  constructor(private param2 = 0.2) {}

  fit(X: Float64Array[], y?: Float64Array): this {
    const n = X.length;
    const d = X[0]?.length ?? 0;
    this.data_ = new Float64Array(d);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < d; j++)
        this.data_[j]! += (X[i]![j] ?? 0) * this.param2;
    if (n > 0) for (let j = 0; j < d; j++) this.data_[j]! /= n;
    void y;
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error("Not fitted");
    return X.map(x => new Float64Array(x.map((v, j) => v - (this.data_[j] ?? 0) * this.param2)));
  }

  fitTransform(X: Float64Array[], y?: Float64Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}
