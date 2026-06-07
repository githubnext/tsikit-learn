/**
 * BatchNormalizer, LayerNormTransformer, and InstanceNormalizer — normalization transformers.
 */

export class BatchNormalizer {
  momentum: number;
  epsilon: number;
  private mean_: Float64Array | null = null;
  private var_: Float64Array | null = null;
  private gamma_: Float64Array | null = null;
  private beta_: Float64Array | null = null;
  nFeaturesIn_: number = 0;

  constructor(momentum = 0.1, epsilon = 1e-5) {
    this.momentum = momentum;
    this.epsilon = epsilon;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    this.mean_ = new Float64Array(p);
    this.var_ = new Float64Array(p);
    this.gamma_ = new Float64Array(p).fill(1);
    this.beta_ = new Float64Array(p).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) this.mean_[j] += (X[i]?.[j] ?? 0) / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) this.var_[j] += ((X[i]?.[j] ?? 0) - (this.mean_[j] ?? 0)) ** 2 / n;
    return this;
  }

  partialFit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    if (!this.mean_) {
      this.mean_ = new Float64Array(p);
      this.var_ = new Float64Array(p);
      this.gamma_ = new Float64Array(p).fill(1);
      this.beta_ = new Float64Array(p).fill(0);
    }
    const batchMean = new Float64Array(p);
    const batchVar = new Float64Array(p);
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) batchMean[j] += (X[i]?.[j] ?? 0) / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) batchVar[j] += ((X[i]?.[j] ?? 0) - (batchMean[j] ?? 0)) ** 2 / n;
    for (let j = 0; j < p; j++) {
      this.mean_[j] = (1 - this.momentum) * (this.mean_[j] ?? 0) + this.momentum * (batchMean[j] ?? 0);
      (this.var_ as Float64Array)[j] = (1 - this.momentum) * ((this.var_ as Float64Array)[j] ?? 0) + this.momentum * (batchVar[j] ?? 0);
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const mean = this.mean_ as Float64Array;
    const vari = this.var_ as Float64Array;
    const gamma = this.gamma_ as Float64Array;
    const beta = this.beta_ as Float64Array;
    return X.map((row) => {
      const out = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) {
        const normalized = ((row[j] ?? 0) - (mean[j] ?? 0)) / Math.sqrt((vari[j] ?? 0) + this.epsilon);
        out[j] = (gamma[j] ?? 1) * normalized + (beta[j] ?? 0);
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }

  setGammaBeta(gamma: Float64Array, beta: Float64Array): void {
    this.gamma_ = gamma;
    this.beta_ = beta;
  }
}

export class LayerNormTransformer {
  epsilon: number;
  gamma_: Float64Array | null = null;
  beta_: Float64Array | null = null;
  nFeaturesIn_: number = 0;

  constructor(epsilon = 1e-5) {
    this.epsilon = epsilon;
  }

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    this.gamma_ = new Float64Array(p).fill(1);
    this.beta_ = new Float64Array(p).fill(0);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const gamma = this.gamma_ ?? new Float64Array(0).fill(1);
    const beta = this.beta_ ?? new Float64Array(0);
    return X.map((row) => {
      const mean = row.reduce((s, v) => s + v, 0) / row.length;
      const vari = row.reduce((s, v) => s + (v - mean) ** 2, 0) / row.length;
      const std = Math.sqrt(vari + this.epsilon);
      return row.map((v, j) => (gamma[j] ?? 1) * ((v - mean) / std) + (beta[j] ?? 0));
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class InstanceNormalizer {
  norm: "l1" | "l2" | "max";
  private p_: number;

  constructor(norm: "l1" | "l2" | "max" = "l2") {
    this.norm = norm;
    this.p_ = norm === "l1" ? 1 : norm === "l2" ? 2 : Number.POSITIVE_INFINITY;
    void this.p_;
  }

  fit(_X: Float64Array[]): this {
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((row) => {
      let normVal: number;
      if (this.norm === "l1") normVal = row.reduce((s, v) => s + Math.abs(v), 0);
      else if (this.norm === "l2") normVal = Math.sqrt(row.reduce((s, v) => s + v * v, 0));
      else normVal = Math.max(...Array.from(row).map(Math.abs));
      if (normVal === 0) return new Float64Array(row);
      return row.map((v) => v / normVal);
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.transform(X);
  }

  inverseTransform(X: Float64Array[]): Float64Array[] {
    return X.map((row) => new Float64Array(row));
  }
}

export class GroupNormalizer {
  groups: number[];
  norm: "l2" | "l1";
  nFeaturesIn_: number = 0;

  constructor(groups: number[], norm: "l2" | "l1" = "l2") {
    this.groups = groups;
    this.norm = norm;
  }

  fit(X: Float64Array[]): this {
    this.nFeaturesIn_ = X[0]?.length ?? 0;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const maxGroup = Math.max(...this.groups) + 1;
    return X.map((row) => {
      const out = new Float64Array(row.length);
      // Normalize within each group
      const groupNorms = new Float64Array(maxGroup);
      for (let j = 0; j < row.length; j++) {
        const g = this.groups[j] ?? 0;
        if (this.norm === "l2") groupNorms[g] += (row[j] ?? 0) ** 2;
        else groupNorms[g] += Math.abs(row[j] ?? 0);
      }
      if (this.norm === "l2") for (let g = 0; g < maxGroup; g++) groupNorms[g] = Math.sqrt(groupNorms[g]);
      for (let j = 0; j < row.length; j++) {
        const g = this.groups[j] ?? 0;
        out[j] = (groupNorms[g] ?? 1) > 0 ? (row[j] ?? 0) / (groupNorms[g] ?? 1) : 0;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
