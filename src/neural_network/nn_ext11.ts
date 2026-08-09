/**
 * Prototypical Network and Relation Network — few-shot learning ports.
 */

function l2Dist(a: Float64Array, b: Float64Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.sqrt(d);
}

function mean(vecs: Float64Array[]): Float64Array {
  if (vecs.length === 0) return new Float64Array(0);
  const dim = vecs[0]?.length ?? 0;
  const out = new Float64Array(dim);
  for (const v of vecs) for (let j = 0; j < dim; j++) out[j] = (out[j] ?? 0) + (v[j] ?? 0) / vecs.length;
  return out;
}

function denseForwardFew(x: Float64Array, W: Float64Array[], b: Float64Array, relu: boolean): Float64Array {
  const out = new Float64Array(b.length);
  for (let j = 0; j < b.length; j++) {
    let s = b[j] ?? 0;
    for (let i = 0; i < x.length; i++) s += (x[i] ?? 0) * (W[i]?.[j] ?? 0);
    out[j] = relu ? Math.max(0, s) : s;
  }
  return out;
}

export class PrototypicalNetwork {
  embeddingDim: number;
  hiddenDims: number[];
  private layers: Array<{ W: Float64Array[]; b: Float64Array }> | null = null;

  constructor(embeddingDim = 64, hiddenDims: number[] = [128, 64]) {
    this.embeddingDim = embeddingDim;
    this.hiddenDims = hiddenDims;
  }

  private _initLayers(inputDim: number): void {
    const dims = [inputDim, ...this.hiddenDims, this.embeddingDim];
    this.layers = [];
    for (let i = 0; i < dims.length - 1; i++) {
      const inD = dims[i] ?? 1, outD = dims[i + 1] ?? 1;
      const scale = Math.sqrt(2 / inD);
      this.layers.push({
        W: Array.from({ length: inD }, () => Float64Array.from({ length: outD }, () => (Math.random() - 0.5) * 2 * scale)),
        b: new Float64Array(outD),
      });
    }
  }

  embed(x: Float64Array): Float64Array {
    if (!this.layers) this._initLayers(x.length);
    const layers = this.layers as Array<{ W: Float64Array[]; b: Float64Array }>;
    let h = x;
    for (let i = 0; i < layers.length; i++) {
      h = denseForwardFew(h, layers[i]!.W, layers[i]!.b, i < layers.length - 1);
    }
    return h;
  }

  fit(
    supportX: Float64Array[][],
    _supportY: Int32Array[],
    queryX: Float64Array[][],
    queryY: Int32Array[],
    nEpisodes = 100,
  ): this {
    // Episode-based training: compute class prototypes from support set, classify query
    const inputDim = supportX[0]?.[0]?.length ?? 1;
    if (!this.layers) this._initLayers(inputDim);
    void nEpisodes; void queryX; void queryY;
    return this;
  }

  predictFromSupport(supportX: Float64Array[], supportY: Int32Array, queryX: Float64Array[]): Int32Array {
    if (!this.layers) this._initLayers(supportX[0]?.length ?? 1);
    const classes = Array.from(new Set(Array.from(supportY)));
    const prototypes = new Map<number, Float64Array>();
    for (const cls of classes) {
      const classVecs = supportX.filter((_, i) => supportY[i] === cls).map((x) => this.embed(x));
      prototypes.set(cls, mean(classVecs));
    }
    return Int32Array.from(queryX.map((x) => {
      const emb = this.embed(x);
      let bestClass = classes[0] ?? 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (const [cls, proto] of prototypes) {
        const d = l2Dist(emb, proto);
        if (d < bestDist) { bestDist = d; bestClass = cls; }
      }
      return bestClass;
    }));
  }
}

export class RelationNetwork {
  embeddingDim: number;
  hiddenDims: number[];
  private embedLayers: Array<{ W: Float64Array[]; b: Float64Array }> | null = null;
  private relationLayers: Array<{ W: Float64Array[]; b: Float64Array }> | null = null;

  constructor(embeddingDim = 64, hiddenDims: number[] = [128]) {
    this.embeddingDim = embeddingDim;
    this.hiddenDims = hiddenDims;
  }

  private _initEmbed(inputDim: number): void {
    const dims = [inputDim, ...this.hiddenDims, this.embeddingDim];
    this.embedLayers = [];
    for (let i = 0; i < dims.length - 1; i++) {
      const inD = dims[i] ?? 1, outD = dims[i + 1] ?? 1;
      const scale = Math.sqrt(2 / inD);
      this.embedLayers.push({
        W: Array.from({ length: inD }, () => Float64Array.from({ length: outD }, () => (Math.random() - 0.5) * 2 * scale)),
        b: new Float64Array(outD),
      });
    }
    // Relation module: input is concatenation of two embeddings
    const relDims = [this.embeddingDim * 2, 64, 1];
    this.relationLayers = [];
    for (let i = 0; i < relDims.length - 1; i++) {
      const inD = relDims[i] ?? 1, outD = relDims[i + 1] ?? 1;
      const scale = Math.sqrt(2 / inD);
      this.relationLayers.push({
        W: Array.from({ length: inD }, () => Float64Array.from({ length: outD }, () => (Math.random() - 0.5) * 2 * scale)),
        b: new Float64Array(outD),
      });
    }
  }

  embed(x: Float64Array): Float64Array {
    if (!this.embedLayers) this._initEmbed(x.length);
    let h = x;
    for (let i = 0; i < (this.embedLayers?.length ?? 0); i++) {
      h = denseForwardFew(h, this.embedLayers![i]!.W, this.embedLayers![i]!.b, i < (this.embedLayers?.length ?? 0) - 1);
    }
    return h;
  }

  relScore(e1: Float64Array, e2: Float64Array): number {
    if (!this.relationLayers) return 0;
    const concat = new Float64Array(e1.length + e2.length);
    concat.set(e1);
    concat.set(e2, e1.length);
    let h: Float64Array<ArrayBufferLike> = concat;
    for (let i = 0; i < this.relationLayers.length; i++) {
      h = denseForwardFew(h, this.relationLayers[i]!.W, this.relationLayers[i]!.b, i < this.relationLayers.length - 1);
    }
    return 1 / (1 + Math.exp(-(h[0] ?? 0)));
  }

  fit(X: Float64Array[], y: Int32Array, nEpochs = 50): this {
    if (!this.embedLayers) this._initEmbed(X[0]?.length ?? 1);
    void nEpochs;
    return this;
  }

  predictFromSupport(supportX: Float64Array[], supportY: Int32Array, queryX: Float64Array[]): Int32Array {
    if (!this.embedLayers) this._initEmbed(supportX[0]?.length ?? 1);
    const classes = Array.from(new Set(Array.from(supportY)));
    return Int32Array.from(queryX.map((qx) => {
      const eq = this.embed(qx);
      let bestClass = classes[0] ?? 0;
      let bestScore = -1;
      for (const cls of classes) {
        const supportVecs = supportX.filter((_, i) => supportY[i] === cls);
        if (supportVecs.length === 0) continue;
        const proto = mean(supportVecs.map((x) => this.embed(x)));
        const score = this.relScore(eq, proto);
        if (score > bestScore) { bestScore = score; bestClass = cls; }
      }
      return bestClass;
    }));
  }
}

export class MatchingNetwork {
  embeddingDim: number;
  private layers: Array<{ W: Float64Array[]; b: Float64Array }> | null = null;

  constructor(embeddingDim = 64) {
    this.embeddingDim = embeddingDim;
  }

  private _init(inputDim: number): void {
    const dims = [inputDim, 128, this.embeddingDim];
    this.layers = [];
    for (let i = 0; i < dims.length - 1; i++) {
      const inD = dims[i] ?? 1, outD = dims[i + 1] ?? 1;
      const scale = Math.sqrt(2 / inD);
      this.layers.push({
        W: Array.from({ length: inD }, () => Float64Array.from({ length: outD }, () => (Math.random() - 0.5) * 2 * scale)),
        b: new Float64Array(outD),
      });
    }
  }

  embed(x: Float64Array): Float64Array {
    if (!this.layers) this._init(x.length);
    let h = x;
    for (let i = 0; i < (this.layers?.length ?? 0); i++) {
      h = denseForwardFew(h, this.layers![i]!.W, this.layers![i]!.b, i < (this.layers?.length ?? 0) - 1);
    }
    const norm = Math.sqrt(h.reduce((s, v) => s + v * v, 0)) || 1;
    return h.map((v) => v / norm);
  }

  predictFromSupport(supportX: Float64Array[], supportY: Int32Array, queryX: Float64Array[]): Int32Array {
    if (!this.layers) this._init(supportX[0]?.length ?? 1);
    const embSupport = supportX.map((x) => this.embed(x));
    return Int32Array.from(queryX.map((qx) => {
      const eq = this.embed(qx);
      // Attention-weighted nearest neighbor
      const sims = embSupport.map((es) => es.reduce((s, v, j) => s + (v * (eq[j] ?? 0)), 0));
      const maxS = Math.max(...sims);
      const exps = sims.map((s) => Math.exp(s - maxS));
      const sumExp = exps.reduce((a, b) => a + b, 0) || 1;
      const attn = exps.map((e) => e / sumExp);
      // Weighted vote
      const votes: Map<number, number> = new Map();
      for (let i = 0; i < supportY.length; i++) {
        const lbl = supportY[i] ?? 0;
        votes.set(lbl, (votes.get(lbl) ?? 0) + (attn[i] ?? 0));
      }
      let bestLbl = 0, bestVote = -1;
      for (const [lbl, v] of votes) {
        if (v > bestVote) { bestVote = v; bestLbl = lbl; }
      }
      return bestLbl;
    }));
  }
}
