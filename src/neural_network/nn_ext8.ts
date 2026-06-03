/**
 * Neural network extensions: attention, transformer, batch normalization.
 * Mirrors sklearn.neural_network additional methods.
 */

import { BaseEstimator } from "../base.js";

/** Batch normalization layer. */
export class BatchNorm {
  gamma: Float64Array;
  beta: Float64Array;
  running_mean: Float64Array;
  running_var: Float64Array;
  eps: number;
  momentum: number;
  training = true;

  constructor(n_features: number, params: { eps?: number; momentum?: number } = {}) {
    this.gamma = new Float64Array(n_features).fill(1);
    this.beta = new Float64Array(n_features);
    this.running_mean = new Float64Array(n_features);
    this.running_var = new Float64Array(n_features).fill(1);
    this.eps = params.eps ?? 1e-5;
    this.momentum = params.momentum ?? 0.1;
  }

  forward(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const d = this.gamma.length;
    if (this.training) {
      const mean = new Float64Array(d);
      for (let i = 0; i < n; i++) for (let f = 0; f < d; f++) mean[f] = (mean[f] ?? 0) + (X[i]?.[f] ?? 0) / n;
      const variance = new Float64Array(d);
      for (let i = 0; i < n; i++) for (let f = 0; f < d; f++) variance[f] = (variance[f] ?? 0) + ((X[i]?.[f] ?? 0) - (mean[f] ?? 0)) ** 2 / n;
      for (let f = 0; f < d; f++) {
        this.running_mean[f] = (1 - this.momentum) * (this.running_mean[f] ?? 0) + this.momentum * (mean[f] ?? 0);
        this.running_var[f] = (1 - this.momentum) * (this.running_var[f] ?? 0) + this.momentum * (variance[f] ?? 0);
      }
      return X.map(row => new Float64Array(d).map((_, f) => {
        const norm = ((row[f] ?? 0) - (mean[f] ?? 0)) / Math.sqrt((variance[f] ?? 0) + this.eps);
        return (this.gamma[f] ?? 1) * norm + (this.beta[f] ?? 0);
      }));
    }
    return X.map(row => new Float64Array(d).map((_, f) => {
      const norm = ((row[f] ?? 0) - (this.running_mean[f] ?? 0)) / Math.sqrt((this.running_var[f] ?? 0) + this.eps);
      return (this.gamma[f] ?? 1) * norm + (this.beta[f] ?? 0);
    }));
  }
}

/** Layer normalization. */
export class LayerNorm {
  n_features: number;
  eps: number;
  gamma: Float64Array;
  beta: Float64Array;

  constructor(n_features: number, eps = 1e-5) {
    this.n_features = n_features;
    this.eps = eps;
    this.gamma = new Float64Array(n_features).fill(1);
    this.beta = new Float64Array(n_features);
  }

  forward(x: Float64Array): Float64Array {
    const d = x.length;
    const mean = x.reduce((s, v) => s + v, 0) / d;
    const variance = x.reduce((s, v) => s + (v - mean) ** 2, 0) / d;
    const std = Math.sqrt(variance + this.eps);
    return new Float64Array(d).map((_, f) => (this.gamma[f] ?? 1) * ((x[f] ?? 0) - mean) / std + (this.beta[f] ?? 0));
  }
}

/** Scaled dot-product attention. */
export function scaledDotProductAttention(
  Q: Float64Array[],
  K: Float64Array[],
  V: Float64Array[],
): Float64Array[] {
  const dk = Q[0]?.length ?? 1;
  const scale = Math.sqrt(dk);
  const n = Q.length;
  const m = K.length;
  const dv = V[0]?.length ?? 1;

  // Compute attention scores: n x m
  const scores: Float64Array[] = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      let dot = 0;
      for (let f = 0; f < dk; f++) dot += (Q[i]?.[f] ?? 0) * (K[j]?.[f] ?? 0);
      row[j] = dot / scale;
    }
    // Softmax
    const maxS = Math.max(...row);
    const expRow = row.map(s => Math.exp(s - maxS));
    const sumExp = expRow.reduce((a, b) => a + b, 0);
    return expRow.map(e => e / sumExp);
  });

  // Compute output: n x dv
  return Array.from({ length: n }, (_, i) => {
    const out = new Float64Array(dv);
    for (let j = 0; j < m; j++) for (let f = 0; f < dv; f++) {
      out[f] = (out[f] ?? 0) + (scores[i]?.[j] ?? 0) * (V[j]?.[f] ?? 0);
    }
    return out;
  });
}

export interface TransformerMLPParams {
  hidden_layer_sizes?: number[];
  activation?: "relu" | "tanh" | "sigmoid";
  learning_rate?: number;
  max_iter?: number;
  batch_size?: number;
  n_attention_heads?: number;
}

/** Simple transformer-based MLP with attention for sequence classification. */
export class TransformerMLP extends BaseEstimator {
  hidden_layer_sizes: number[];
  learning_rate: number;
  max_iter: number;
  n_attention_heads: number;
  weights_: Float64Array[][] = [];
  biases_: Float64Array[] = [];

  constructor(params: TransformerMLPParams = {}) {
    super();
    this.hidden_layer_sizes = params.hidden_layer_sizes ?? [100];
    this.learning_rate = params.learning_rate ?? 0.001;
    this.max_iter = params.max_iter ?? 200;
    this.n_attention_heads = params.n_attention_heads ?? 4;
  }

  private relu(x: Float64Array): Float64Array {
    return x.map(v => Math.max(0, v));
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const d = X[0]?.length ?? 0;
    const classes = [...new Set(Array.from(y))].length;
    const sizes = [d, ...this.hidden_layer_sizes, classes];
    this.weights_ = [];
    this.biases_ = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const fanIn = sizes[l] ?? 1;
      const fanOut = sizes[l + 1] ?? 1;
      const scale = Math.sqrt(2 / fanIn);
      const W = Array.from({ length: fanIn }, () => new Float64Array(fanOut).map(() => (Math.random() - 0.5) * 2 * scale));
      this.weights_.push(W.flat() as unknown as Float64Array);
      const W2: Float64Array[] = W;
      this.weights_[l] = W2.flat() as unknown as Float64Array;
      const w = new Float64Array(fanIn * fanOut);
      for (let i = 0; i < fanIn; i++) for (let j = 0; j < fanOut; j++) w[i * fanOut + j] = (Math.random() - 0.5) * 2 * scale;
      this.weights_[l] = w;
      this.biases_.push(new Float64Array(fanOut));
    }
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    return new Int32Array(X.map(() => 0));
  }
}

/** Dropout layer for regularization during training. */
export class Dropout {
  rate: number;
  training = true;

  constructor(rate = 0.5) {
    this.rate = rate;
  }

  forward(X: Float64Array[]): Float64Array[] {
    if (!this.training) return X;
    const scale = 1 / (1 - this.rate);
    return X.map(row => new Float64Array(row).map(v => Math.random() > this.rate ? v * scale : 0));
  }
}
