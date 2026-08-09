/**
 * Neural network extensions: Transformer, Attention mechanisms, Capsule network
 */

export class MultiHeadAttentionExt {
  private Wq_: Float64Array[][] = [];
  private Wk_: Float64Array[][] = [];
  private Wv_: Float64Array[][] = [];
  private Wo_: Float64Array[] = [];
  private fitted_ = false;

  constructor(
    private dModel: number = 64,
    private nHeads: number = 8,
    private dropout: number = 0.0,
    private randomState: number = 42
  ) {
    if (dModel % nHeads !== 0) throw new Error('dModel must be divisible by nHeads');
  }

  private get dK(): number { return this.dModel / this.nHeads; }

  initialize(): this {
    let rng = this.randomState;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return ((rng / 0xffffffff) * 2 - 1) * Math.sqrt(2 / this.dModel); };
    const init = (rows: number, cols: number) => Array.from({ length: rows }, () => new Float64Array(cols).map(() => rand()));

    this.Wq_ = Array.from({ length: this.nHeads }, () => init(this.dModel, this.dK));
    this.Wk_ = Array.from({ length: this.nHeads }, () => init(this.dModel, this.dK));
    this.Wv_ = Array.from({ length: this.nHeads }, () => init(this.dModel, this.dK));
    this.Wo_ = init(this.dModel, this.dModel);
    this.fitted_ = true;
    return this;
  }

  forward(query: Float64Array[], key: Float64Array[], value: Float64Array[]): Float64Array[] {
    if (!this.fitted_) this.initialize();
    const seqLen = query.length;
    const headOutputs: Float64Array[][] = [];

    for (let h = 0; h < this.nHeads; h++) {
      const Q = this._project(query, this.Wq_[h]!);
      const K = this._project(key, this.Wk_[h]!);
      const V = this._project(value, this.Wv_[h]!);
      headOutputs.push(this._scaledDotProductAttention(Q, K, V));
    }

    // Concatenate heads
    const concat = Array.from({ length: seqLen }, (_, i) =>
      new Float64Array(headOutputs.flatMap(h => Array.from(h[i]!)))
    );

    // Apply output projection
    return this._project(concat, this.Wo_);
  }

  private _project(X: Float64Array[], W: Float64Array[]): Float64Array[] {
    const outDim = W[0]?.length ?? 0;
    return X.map(x => {
      const out = new Float64Array(outDim);
      for (let j = 0; j < outDim; j++) for (let k = 0; k < x.length; k++) out[j] = (out[j] ?? 0) + (x[k] ?? 0) * (W[k]?.[j] ?? 0);
      return out;
    });
  }

  private _scaledDotProductAttention(Q: Float64Array[], K: Float64Array[], V: Float64Array[]): Float64Array[] {
    const scale = Math.sqrt(this.dK);
    const seqLen = Q.length;
    const scores = Array.from({ length: seqLen }, (_, i) =>
      new Float64Array(seqLen).map((_, j) => Q[i]!.reduce((s, v, k) => s + v * (K[j]?.[k] ?? 0), 0) / scale)
    );
    // Softmax
    const attn = scores.map(row => {
      const maxV = Math.max(...Array.from(row));
      const exp = row.map(v => Math.exp(v - maxV));
      const sum = exp.reduce((s, v) => s + v, 0);
      return exp.map(v => v / sum);
    });
    // Weighted sum of V
    return Array.from({ length: seqLen }, (_, i) => {
      const out = new Float64Array(this.dK);
      for (let j = 0; j < seqLen; j++) for (let k = 0; k < this.dK; k++) out[k] = (out[k] ?? 0) + (attn[i]?.[j] ?? 0) * (V[j]?.[k] ?? 0);
      return out;
    });
  }
}

export class TransformerEncoderLayerExt {
  private attention_: MultiHeadAttentionExt;
  private norm1W_: Float64Array;
  private norm1B_: Float64Array;
  private norm2W_: Float64Array;
  private norm2B_: Float64Array;
  private ff1W_: Float64Array[];
  private ff1B_: Float64Array;
  private ff2W_: Float64Array[];
  private ff2B_: Float64Array;

  constructor(
    private dModel: number = 64,
    private nHeads: number = 8,
    private dimFeedforward: number = 256,
    private dropout: number = 0.1,
    randomState: number = 42
  ) {
    this.attention_ = new MultiHeadAttentionExt(dModel, nHeads, dropout, randomState);
    let rng = randomState;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return ((rng / 0xffffffff) * 2 - 1) * Math.sqrt(2 / dModel); };
    this.norm1W_ = new Float64Array(dModel).fill(1);
    this.norm1B_ = new Float64Array(dModel);
    this.norm2W_ = new Float64Array(dModel).fill(1);
    this.norm2B_ = new Float64Array(dModel);
    this.ff1W_ = Array.from({ length: dModel }, () => new Float64Array(dimFeedforward).map(() => rand()));
    this.ff1B_ = new Float64Array(dimFeedforward);
    this.ff2W_ = Array.from({ length: dimFeedforward }, () => new Float64Array(dModel).map(() => rand()));
    this.ff2B_ = new Float64Array(dModel);
  }

  forward(src: Float64Array[]): Float64Array[] {
    // Self-attention + residual
    const attn = this.attention_.forward(src, src, src);
    const src2 = src.map((x, i) => this._layerNorm(x.map((v, j) => v + (attn[i]?.[j] ?? 0)), this.norm1W_, this.norm1B_));
    // FFN + residual
    const ff = src2.map(x => this._ffn(x));
    return src2.map((x, i) => this._layerNorm(x.map((v, j) => v + (ff[i]?.[j] ?? 0)), this.norm2W_, this.norm2B_));
  }

  private _layerNorm(x: Float64Array, w: Float64Array, b: Float64Array): Float64Array {
    const mean = x.reduce((s, v) => s + v, 0) / x.length;
    const variance = x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length;
    const std = Math.sqrt(variance + 1e-5);
    return new Float64Array(x.map((v, j) => ((v - mean) / std) * (w[j] ?? 1) + (b[j] ?? 0)));
  }

  private _ffn(x: Float64Array): Float64Array {
    // x -> ff1 -> ReLU -> ff2
    const h = new Float64Array(this.dimFeedforward);
    for (let j = 0; j < this.dimFeedforward; j++) {
      h[j] = Math.max(0, x.reduce((s, v, k) => s + v * (this.ff1W_[k]?.[j] ?? 0), 0) + (this.ff1B_[j] ?? 0));
    }
    return new Float64Array(this.dModel).map((_, k) => h.reduce((s, v, j) => s + v * (this.ff2W_[j]?.[k] ?? 0), 0) + (this.ff2B_[k] ?? 0));
  }
}

export class CapsuleLayerExt {
  private W_: Float64Array[][][] = []; // [in_caps][out_caps][in_dim][out_dim]
  private nIterations_: number;
  private fitted_ = false;

  constructor(
    private inCapsules: number = 32,
    private outCapsules: number = 10,
    private inDim: number = 8,
    private outDim: number = 16,
    nIterations: number = 3,
    randomState: number = 42
  ) {
    this.nIterations_ = nIterations;
    let rng = randomState;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return ((rng / 0xffffffff) * 2 - 1) * 0.1; };
    this.W_ = Array.from({ length: inCapsules }, () =>
      Array.from({ length: outCapsules }, () =>
        Array.from({ length: inDim }, () => new Float64Array(outDim).map(() => rand()))
      )
    );
    this.fitted_ = true;
  }

  forward(inputs: Float64Array[]): Float64Array[] {
    // inputs: [batchSize x inCapsules x inDim] - treated as [inCapsules] capsules each with inDim
    // For simplicity, treat inputs as [inCapsules x inDim]
    const uHat = Array.from({ length: this.inCapsules }, (_, i) =>
      Array.from({ length: this.outCapsules }, (_, j) => {
        const out = new Float64Array(this.outDim);
        for (let d = 0; d < this.outDim; d++) for (let k = 0; k < this.inDim; k++) out[d] = (out[d] ?? 0) + (inputs[i]?.[k] ?? 0) * (this.W_[i]?.[j]?.[k]?.[d] ?? 0);
        return out;
      })
    );

    // Dynamic routing
    const b = Array.from({ length: this.inCapsules }, () => new Float64Array(this.outCapsules));
    let vj: Float64Array[] = [];

    for (let iter = 0; iter < this.nIterations_; iter++) {
      const c = b.map(row => this._softmax(row));
      vj = Array.from({ length: this.outCapsules }, (_, j) => {
        const s = new Float64Array(this.outDim);
        for (let i = 0; i < this.inCapsules; i++) for (let d = 0; d < this.outDim; d++) s[d] = (s[d] ?? 0) + (c[i]?.[j] ?? 0) * (uHat[i]?.[j]?.[d] ?? 0);
        return this._squash(s);
      });
      for (let i = 0; i < this.inCapsules; i++) for (let j = 0; j < this.outCapsules; j++) {
        b[i]![j] = (b[i]?.[j] ?? 0) + (uHat[i]?.[j] ?? new Float64Array(this.outDim)).reduce((s, v, d) => s + v * (vj[j]?.[d] ?? 0), 0);
      }
    }
    return vj;
  }

  private _softmax(x: Float64Array): Float64Array {
    const maxV = Math.max(...Array.from(x));
    const exp = x.map(v => Math.exp(v - maxV));
    const sum = exp.reduce((s, v) => s + v, 0);
    return exp.map(v => v / sum);
  }

  private _squash(s: Float64Array): Float64Array {
    const norm2 = s.reduce((ss, v) => ss + v * v, 0);
    const scale = norm2 / (1 + norm2) / (Math.sqrt(norm2) + 1e-10);
    return s.map(v => v * scale);
  }
}

export class SelfNormalizingNetworkExt {
  private layers_: Array<{ W: Float64Array[]; b: Float64Array }> = [];
  private fitted_ = false;

  constructor(
    private hiddenSizes: number[] = [128, 64],
    private inputSize: number = 10,
    private outputSize: number = 1,
    private alpha: number = 1.6732632423543772,
    private scale: number = 1.0507009873554805,
    randomState: number = 42
  ) {
    let rng = randomState;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return ((rng / 0xffffffff) * 2 - 1); };
    const sizes = [inputSize, ...hiddenSizes, outputSize];
    this.layers_ = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const fan_in = sizes[l]!;
      const fan_out = sizes[l + 1]!;
      const std = Math.sqrt(1 / fan_in);
      this.layers_.push({
        W: Array.from({ length: fan_in }, () => new Float64Array(fan_out).map(() => rand() * std)),
        b: new Float64Array(fan_out)
      });
    }
    this.fitted_ = true;
  }

  forward(X: Float64Array[]): Float64Array[] {
    return X.map(row => {
      let h = row;
      for (let l = 0; l < this.layers_.length; l++) {
        const { W, b } = this.layers_[l]!;
        const next = new Float64Array(b.length).map((_, j) => {
          const z = h.reduce((s, v, k) => s + v * (W[k]?.[j] ?? 0), 0) + (b[j] ?? 0);
          // SELU activation (except last layer)
          if (l < this.layers_.length - 1) return this.scale * (z > 0 ? z : this.alpha * (Math.exp(z) - 1));
          return z;
        });
        h = next;
      }
      return h;
    });
  }

  get layers(): Array<{ W: Float64Array[]; b: Float64Array }> { return this.layers_; }
}
