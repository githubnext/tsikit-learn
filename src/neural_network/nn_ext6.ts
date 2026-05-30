/**
 * Neural network extensions: attention mechanisms, transformers.
 * Mirrors sklearn.neural_network advanced layers.
 */

import { BaseEstimator } from "../base.js";

/** Scaled dot-product attention. */
export function scaledDotProductAttention(
  Q: Float64Array[],
  K: Float64Array[],
  V: Float64Array[],
): Float64Array[] {
  const n = Q.length;
  const dk = Q[0]?.length ?? 1;
  const scale = Math.sqrt(dk);
  // Compute attention scores
  const scores = Array.from({ length: n }, (_, i) =>
    new Float64Array(n).map((_, j) => {
      let s = 0;
      const qi = Q[i]!, kj = K[j]!;
      for (let k = 0; k < dk; k++) s += (qi[k] ?? 0) * (kj[k] ?? 0);
      return s / scale;
    }),
  );
  // Softmax
  const attn = scores.map((row) => {
    const maxV = Math.max(...row);
    const exp = row.map((v) => Math.exp(v - maxV));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map((v) => v / Math.max(sum, 1e-10));
  });
  // Weighted sum of V
  return attn.map((ai) =>
    new Float64Array(V[0]?.length ?? 1).map((_, k) => {
      let s = 0;
      for (let j = 0; j < n; j++) s += (ai[j] ?? 0) * (V[j]?.[k] ?? 0);
      return s;
    }),
  );
}

export interface MultiHeadAttentionParams {
  embed_dim?: number;
  num_heads?: number;
}

/** Multi-head attention layer. */
export class MultiHeadAttention extends BaseEstimator {
  embed_dim: number;
  num_heads: number;
  head_dim: number;
  Wq: Float64Array[][];
  Wk: Float64Array[][];
  Wv: Float64Array[][];
  Wo: Float64Array[];

  constructor(params: MultiHeadAttentionParams = {}) {
    super();
    this.embed_dim = params.embed_dim ?? 64;
    this.num_heads = params.num_heads ?? 8;
    this.head_dim = Math.floor(this.embed_dim / this.num_heads);
    // Initialize weight matrices (random small values)
    this.Wq = Array.from({ length: this.num_heads }, (_, h) =>
      Array.from({ length: this.embed_dim }, (_, i) =>
        new Float64Array(this.head_dim).map((_, j) => 0.01 * Math.sin(h * 100 + i * 10 + j)),
      ),
    );
    this.Wk = Array.from({ length: this.num_heads }, (_, h) =>
      Array.from({ length: this.embed_dim }, (_, i) =>
        new Float64Array(this.head_dim).map((_, j) => 0.01 * Math.cos(h * 100 + i * 10 + j)),
      ),
    );
    this.Wv = Array.from({ length: this.num_heads }, (_, h) =>
      Array.from({ length: this.embed_dim }, (_, i) =>
        new Float64Array(this.head_dim).map((_, j) => 0.01 * Math.sin(h * 100 + i * 10 + j + 1)),
      ),
    );
    this.Wo = Array.from({ length: this.embed_dim }, (_, i) =>
      new Float64Array(this.embed_dim).map((_, j) => 0.01 * Math.cos(i * 10 + j)),
    );
  }

  forward(X: Float64Array[]): Float64Array[] {
    const n = X.length;
    const headOutputs: Float64Array[][] = [];
    for (let h = 0; h < this.num_heads; h++) {
      const Q = X.map((xi) => new Float64Array(this.head_dim).map((_, k) => {
        let s = 0;
        for (let d = 0; d < this.embed_dim; d++) s += (xi[d] ?? 0) * (this.Wq[h]?.[d]?.[k] ?? 0);
        return s;
      }));
      const K = X.map((xi) => new Float64Array(this.head_dim).map((_, k) => {
        let s = 0;
        for (let d = 0; d < this.embed_dim; d++) s += (xi[d] ?? 0) * (this.Wk[h]?.[d]?.[k] ?? 0);
        return s;
      }));
      const V = X.map((xi) => new Float64Array(this.head_dim).map((_, k) => {
        let s = 0;
        for (let d = 0; d < this.embed_dim; d++) s += (xi[d] ?? 0) * (this.Wv[h]?.[d]?.[k] ?? 0);
        return s;
      }));
      headOutputs.push(scaledDotProductAttention(Q, K, V));
    }
    // Concatenate heads and project
    return Array.from({ length: n }, (_, i) => {
      const concat = new Float64Array(this.embed_dim);
      for (let h = 0; h < this.num_heads; h++) {
        const hd = headOutputs[h]?.[i];
        if (hd) for (let k = 0; k < this.head_dim; k++) concat[h * this.head_dim + k] = hd[k] ?? 0;
      }
      return new Float64Array(this.embed_dim).map((_, d) => {
        let s = 0;
        for (let k = 0; k < this.embed_dim; k++) s += (this.Wo[d]?.[k] ?? 0) * (concat[k] ?? 0);
        return s;
      });
    });
  }
}

export interface PositionalEncodingParams {
  max_len?: number;
  d_model?: number;
}

/** Positional encoding for transformer models. */
export class PositionalEncoding extends BaseEstimator {
  max_len: number;
  d_model: number;
  pe_: Float64Array[];

  constructor(params: PositionalEncodingParams = {}) {
    super();
    this.max_len = params.max_len ?? 512;
    this.d_model = params.d_model ?? 64;
    this.pe_ = this._compute();
  }

  private _compute(): Float64Array[] {
    return Array.from({ length: this.max_len }, (_, pos) => {
      const enc = new Float64Array(this.d_model);
      for (let i = 0; i < this.d_model; i += 2) {
        const angle = pos / (10000 ** (i / this.d_model));
        enc[i] = Math.sin(angle);
        if (i + 1 < this.d_model) enc[i + 1] = Math.cos(angle);
      }
      return enc;
    });
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((xi, i) => {
      const pe = this.pe_[i % this.max_len];
      if (!pe) return xi;
      const out = new Float64Array(xi.length);
      for (let k = 0; k < xi.length; k++) out[k] = (xi[k] ?? 0) + (pe[k] ?? 0);
      return out;
    });
  }
}

export interface LayerNormParams {
  eps?: number;
}

/** Layer normalization. */
export class LayerNorm extends BaseEstimator {
  eps: number;
  gamma_: Float64Array = new Float64Array(0);
  beta_: Float64Array = new Float64Array(0);
  n_features_in_ = 0;

  constructor(params: LayerNormParams = {}) {
    super();
    this.eps = params.eps ?? 1e-5;
  }

  fit(X: Float64Array[]): this {
    const nf = X[0]?.length ?? 0;
    this.n_features_in_ = nf;
    this.gamma_ = new Float64Array(nf).fill(1);
    this.beta_ = new Float64Array(nf).fill(0);
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    return X.map((xi) => {
      let mean = 0, variance = 0;
      for (let k = 0; k < xi.length; k++) mean += xi[k] ?? 0;
      mean /= xi.length;
      for (let k = 0; k < xi.length; k++) variance += ((xi[k] ?? 0) - mean) ** 2;
      variance /= xi.length;
      const std = Math.sqrt(variance + this.eps);
      const out = new Float64Array(xi.length);
      for (let k = 0; k < xi.length; k++) out[k] = (this.gamma_[k] ?? 1) * ((xi[k] ?? 0) - mean) / std + (this.beta_[k] ?? 0);
      return out;
    });
  }

  fit_transform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
