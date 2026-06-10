/**
 * Convolutional Neural Network layers and 1D convolution utilities.
 */

export class Conv1D {
  private filters_!: Float64Array[];
  private bias_!: Float64Array;
  private fitted_ = false;

  constructor(
    private nFilters = 32,
    private kernelSize = 3,
    private stride = 1,
    private padding: 'valid' | 'same' = 'valid',
    private activation: 'relu' | 'linear' = 'relu'
  ) {}

  private _activate(x: number): number {
    return this.activation === 'relu' ? Math.max(0, x) : x;
  }

  fit(X: Float64Array[], y?: Float64Array | Int32Array): this {
    // Weights are initialized randomly — actual training is done via backprop in MLP
    const inChannels = 1;
    this.filters_ = Array.from({ length: this.nFilters }, () =>
      new Float64Array(this.kernelSize * inChannels).map(() => (Math.random() - 0.5) * 0.1)
    );
    this.bias_ = new Float64Array(this.nFilters);
    void y;
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(seq => {
      const n = seq.length;
      const outLen = this.padding === 'same' ? Math.ceil(n / this.stride) : Math.floor((n - this.kernelSize) / this.stride) + 1;
      const out = new Float64Array(outLen * this.nFilters);
      for (let t = 0; t < outLen; t++) {
        const tStart = t * this.stride - (this.padding === 'same' ? Math.floor(this.kernelSize / 2) : 0);
        for (let f = 0; f < this.nFilters; f++) {
          let sum = this.bias_[f] ?? 0;
          for (let k = 0; k < this.kernelSize; k++) {
            const pos = tStart + k;
            if (pos >= 0 && pos < n) sum += (seq[pos] ?? 0) * (this.filters_[f]![k] ?? 0);
          }
          out[t * this.nFilters + f] = this._activate(sum);
        }
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[], y?: Float64Array | Int32Array): Float64Array[] { return this.fit(X, y).transform(X); }
}

export class MaxPool1D {
  constructor(private poolSize = 2, private stride: number | null = null) {}

  transform(X: Float64Array[], nChannels = 1): Float64Array[] {
    const s = this.stride ?? this.poolSize;
    return X.map(seq => {
      const seqLen = seq.length / nChannels;
      const outLen = Math.floor((seqLen - this.poolSize) / s) + 1;
      const out = new Float64Array(outLen * nChannels);
      for (let c = 0; c < nChannels; c++) {
        for (let t = 0; t < outLen; t++) {
          let maxVal = -Number.POSITIVE_INFINITY;
          for (let k = 0; k < this.poolSize; k++) {
            const pos = (t * s + k) * nChannels + c;
            maxVal = Math.max(maxVal, seq[pos] ?? -Number.POSITIVE_INFINITY);
          }
          out[t * nChannels + c] = maxVal;
        }
      }
      return out;
    });
  }
}

export class BatchNormalization {
  private gamma_!: Float64Array;
  private beta_!: Float64Array;
  private runningMean_!: Float64Array;
  private runningVar_!: Float64Array;
  private fitted_ = false;

  constructor(private momentum = 0.9, private epsilon = 1e-5) {}

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    this.gamma_ = new Float64Array(p).fill(1);
    this.beta_ = new Float64Array(p).fill(0);
    const n = X.length;
    this.runningMean_ = new Float64Array(p).map((_, j) => X.reduce((s, row) => s + (row[j] ?? 0), 0) / n);
    this.runningVar_ = new Float64Array(p).map((_, j) => {
      const mean = this.runningMean_[j] ?? 0;
      return X.reduce((s, row) => s + ((row[j] ?? 0) - mean) ** 2, 0) / n;
    });
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[], training = false): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const p = X[0]?.length ?? 0, n = X.length;
    let mean = this.runningMean_, variance = this.runningVar_;
    if (training) {
      mean = new Float64Array(p).map((_, j) => X.reduce((s, row) => s + (row[j] ?? 0), 0) / n);
      variance = new Float64Array(p).map((_, j) => {
        const m = mean[j] ?? 0;
        return X.reduce((s, row) => s + ((row[j] ?? 0) - m) ** 2, 0) / n;
      });
      for (let j = 0; j < p; j++) {
        this.runningMean_[j] = this.momentum * (this.runningMean_[j] ?? 0) + (1 - this.momentum) * (mean[j] ?? 0);
        this.runningVar_[j] = this.momentum * (this.runningVar_[j] ?? 0) + (1 - this.momentum) * (variance[j] ?? 0);
      }
    }
    return X.map(row => new Float64Array(row.map((v, j) => {
      const xNorm = (v - (mean[j] ?? 0)) / Math.sqrt((variance[j] ?? 0) + this.epsilon);
      return (this.gamma_[j] ?? 1) * xNorm + (this.beta_[j] ?? 0);
    })));
  }

  fitTransform(X: Float64Array[]): Float64Array[] { return this.fit(X).transform(X); }
}
