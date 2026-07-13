/**
 * Neural network utilities: batch normalization, dropout, weight initializers, loss functions
 */

export type ActivationFn = (x: Float64Array) => Float64Array;

export const activations = {
  relu: (x: Float64Array): Float64Array =>
    x.map((v) => Math.max(0, v)) as unknown as Float64Array,
  sigmoid: (x: Float64Array): Float64Array =>
    x.map((v) => 1 / (1 + Math.exp(-v))) as unknown as Float64Array,
  tanh: (x: Float64Array): Float64Array =>
    x.map(Math.tanh) as unknown as Float64Array,
  softmax: (x: Float64Array): Float64Array => {
    const max = Math.max(...Array.from(x));
    const exp = x.map((v) => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map((v) => v / sum) as unknown as Float64Array;
  },
  leaky_relu:
    (alpha = 0.01) =>
    (x: Float64Array): Float64Array =>
      x.map((v) => (v > 0 ? v : alpha * v)) as unknown as Float64Array,
  elu:
    (alpha = 1.0) =>
    (x: Float64Array): Float64Array =>
      x.map((v) =>
        v >= 0 ? v : alpha * (Math.exp(v) - 1),
      ) as unknown as Float64Array,
  gelu: (x: Float64Array): Float64Array =>
    x.map(
      (v) =>
        0.5 *
        v *
        (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3))),
    ) as unknown as Float64Array,
  silu: (x: Float64Array): Float64Array =>
    x.map((v) => v / (1 + Math.exp(-v))) as unknown as Float64Array,
};

export const losses = {
  mse: (yTrue: Float64Array, yPred: Float64Array): number => {
    let s = 0;
    for (let i = 0; i < yTrue.length; i++)
      s += ((yTrue[i] ?? 0) - (yPred[i] ?? 0)) ** 2;
    return s / yTrue.length;
  },
  mae: (yTrue: Float64Array, yPred: Float64Array): number => {
    let s = 0;
    for (let i = 0; i < yTrue.length; i++)
      s += Math.abs((yTrue[i] ?? 0) - (yPred[i] ?? 0));
    return s / yTrue.length;
  },
  crossEntropy: (yTrue: Float64Array, yPred: Float64Array): number => {
    let s = 0;
    for (let i = 0; i < yTrue.length; i++) {
      s -= (yTrue[i] ?? 0) * Math.log(Math.max(1e-15, yPred[i] ?? 0));
    }
    return s / yTrue.length;
  },
  huber:
    (delta = 1.0) =>
    (yTrue: Float64Array, yPred: Float64Array): number => {
      let s = 0;
      for (let i = 0; i < yTrue.length; i++) {
        const e = Math.abs((yTrue[i] ?? 0) - (yPred[i] ?? 0));
        s += e <= delta ? 0.5 * e * e : delta * (e - 0.5 * delta);
      }
      return s / yTrue.length;
    },
};

export type InitializerFn = (shape: [number, number]) => Float64Array[];

export const initializers = {
  glorot_uniform: ([fanIn, fanOut]: [number, number]): Float64Array[] => {
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    return Array.from({ length: fanIn }, () => {
      const row = new Float64Array(fanOut);
      for (let i = 0; i < fanOut; i++) row[i] = (Math.random() * 2 - 1) * limit;
      return row;
    });
  },
  he_normal: ([fanIn, fanOut]: [number, number]): Float64Array[] => {
    const std = Math.sqrt(2 / fanIn);
    return Array.from({ length: fanIn }, () => {
      const row = new Float64Array(fanOut);
      for (let i = 0; i < fanOut; i++) {
        // Box-Muller
        const u1 = Math.random();
        const u2 = Math.random();
        row[i] =
          std *
          Math.sqrt(-2 * Math.log(u1 + 1e-10)) *
          Math.cos(2 * Math.PI * u2);
      }
      return row;
    });
  },
  zeros: ([fanIn, fanOut]: [number, number]): Float64Array[] =>
    Array.from({ length: fanIn }, () => new Float64Array(fanOut)),
  ones: ([fanIn, fanOut]: [number, number]): Float64Array[] =>
    Array.from({ length: fanIn }, () => new Float64Array(fanOut).fill(1)),
};

export class BatchNormLayer {
  private gamma: Float64Array;
  private beta: Float64Array;
  private runningMean: Float64Array;
  private runningVar: Float64Array;
  private momentum: number;
  private eps: number;
  nFeatures: number;

  constructor(nFeatures: number, momentum = 0.1, eps = 1e-5) {
    this.nFeatures = nFeatures;
    this.momentum = momentum;
    this.eps = eps;
    this.gamma = new Float64Array(nFeatures).fill(1);
    this.beta = new Float64Array(nFeatures);
    this.runningMean = new Float64Array(nFeatures);
    this.runningVar = new Float64Array(nFeatures).fill(1);
  }

  forward(X: Float64Array[], training = true): Float64Array[] {
    const n = X.length;
    const result: Float64Array[] = [];

    if (training) {
      const mean = new Float64Array(this.nFeatures);
      for (const row of X)
        for (let j = 0; j < this.nFeatures; j++)
          mean[j] = (mean[j] ?? 0) + (row[j] ?? 0) / n;
      const variance = new Float64Array(this.nFeatures);
      for (const row of X)
        for (let j = 0; j < this.nFeatures; j++)
          variance[j] =
            (variance[j] ?? 0) + ((row[j] ?? 0) - (mean[j] ?? 0)) ** 2 / n;
      for (let j = 0; j < this.nFeatures; j++) {
        this.runningMean[j] =
          (1 - this.momentum) * (this.runningMean[j] ?? 0) +
          this.momentum * (mean[j] ?? 0);
        this.runningVar[j] =
          (1 - this.momentum) * (this.runningVar[j] ?? 1) +
          this.momentum * (variance[j] ?? 1);
      }
      for (const row of X) {
        const out = new Float64Array(this.nFeatures);
        for (let j = 0; j < this.nFeatures; j++) {
          out[j] =
            (((row[j] ?? 0) - (mean[j] ?? 0)) /
              Math.sqrt((variance[j] ?? 1) + this.eps)) *
              (this.gamma[j] ?? 1) +
            (this.beta[j] ?? 0);
        }
        result.push(out);
      }
    } else {
      for (const row of X) {
        const out = new Float64Array(this.nFeatures);
        for (let j = 0; j < this.nFeatures; j++) {
          out[j] =
            (((row[j] ?? 0) - (this.runningMean[j] ?? 0)) /
              Math.sqrt((this.runningVar[j] ?? 1) + this.eps)) *
              (this.gamma[j] ?? 1) +
            (this.beta[j] ?? 0);
        }
        result.push(out);
      }
    }
    return result;
  }
}

export class DropoutLayer {
  private rate: number;

  constructor(rate = 0.5) {
    this.rate = rate;
  }

  forward(X: Float64Array[], training = true): Float64Array[] {
    if (!training) return X;
    return X.map((row) => {
      const out = new Float64Array(row.length);
      for (let i = 0; i < row.length; i++) {
        if (Math.random() > this.rate) out[i] = (row[i] ?? 0) / (1 - this.rate);
      }
      return out;
    });
  }
}
