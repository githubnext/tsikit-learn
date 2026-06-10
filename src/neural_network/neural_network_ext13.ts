/**
 * LSTM, GRU, and attention mechanisms for sequence modeling.
 */

function sigmoid(x: number): number { return 1 / (1 + Math.exp(-x)); }
function tanh(x: number): number { return Math.tanh(x); }

export class LSTMCell {
  private Wf: Float64Array; private Wi: Float64Array; private Wc: Float64Array; private Wo: Float64Array;
  private Uf: Float64Array; private Ui: Float64Array; private Uc: Float64Array; private Uo: Float64Array;
  private bf: Float64Array; private bi: Float64Array; private bc: Float64Array; private bo: Float64Array;

  constructor(private inputSize: number, private hiddenSize: number) {
    const init = (n: number) => new Float64Array(n).map(() => (Math.random() - 0.5) * 0.1);
    const h = hiddenSize, i = inputSize;
    this.Wf = init(h * i); this.Wi = init(h * i); this.Wc = init(h * i); this.Wo = init(h * i);
    this.Uf = init(h * h); this.Ui = init(h * h); this.Uc = init(h * h); this.Uo = init(h * h);
    this.bf = new Float64Array(h); this.bi = new Float64Array(h); this.bc = new Float64Array(h); this.bo = new Float64Array(h);
  }

  forward(x: Float64Array, h: Float64Array, c: Float64Array): { h: Float64Array; c: Float64Array } {
    const H = this.hiddenSize, I = this.inputSize;
    const matvec = (W: Float64Array, v: Float64Array, size: number, inSize: number) =>
      new Float64Array(size).map((_, i) => v.reduce((s, vv, j) => s + vv * (W[i * inSize + j] ?? 0), 0));
    const add = (a: Float64Array, b: Float64Array) => new Float64Array(a.map((v, i) => v + (b[i] ?? 0)));
    const f = add(add(matvec(this.Wf, x, H, I), matvec(this.Uf, h, H, H)), this.bf).map(sigmoid);
    const i_ = add(add(matvec(this.Wi, x, H, I), matvec(this.Ui, h, H, H)), this.bi).map(sigmoid);
    const g = add(add(matvec(this.Wc, x, H, I), matvec(this.Uc, h, H, H)), this.bc).map(tanh);
    const o = add(add(matvec(this.Wo, x, H, I), matvec(this.Uo, h, H, H)), this.bo).map(sigmoid);
    const cNew = new Float64Array(H).map((_, k) => (f[k] ?? 0) * (c[k] ?? 0) + (i_[k] ?? 0) * (g[k] ?? 0));
    const hNew = new Float64Array(H).map((_, k) => (o[k] ?? 0) * tanh(cNew[k] ?? 0));
    return { h: hNew, c: cNew };
  }
}

export class GRUCell {
  private Wz: Float64Array; private Wr: Float64Array; private Wh: Float64Array;
  private Uz: Float64Array; private Ur: Float64Array; private Uh: Float64Array;
  private bz: Float64Array; private br: Float64Array; private bh: Float64Array;

  constructor(private inputSize: number, private hiddenSize: number) {
    const init = (n: number) => new Float64Array(n).map(() => (Math.random() - 0.5) * 0.1);
    const h = hiddenSize, i = inputSize;
    this.Wz = init(h * i); this.Wr = init(h * i); this.Wh = init(h * i);
    this.Uz = init(h * h); this.Ur = init(h * h); this.Uh = init(h * h);
    this.bz = new Float64Array(h); this.br = new Float64Array(h); this.bh = new Float64Array(h);
  }

  forward(x: Float64Array, h: Float64Array): Float64Array {
    const H = this.hiddenSize, I = this.inputSize;
    const mv = (W: Float64Array, v: Float64Array, sz: number, isz: number) =>
      new Float64Array(sz).map((_, i) => v.reduce((s, vv, j) => s + vv * (W[i * isz + j] ?? 0), 0));
    const add = (a: Float64Array, b: Float64Array) => new Float64Array(a.map((v, i) => v + (b[i] ?? 0)));
    const z = add(add(mv(this.Wz, x, H, I), mv(this.Uz, h, H, H)), this.bz).map(sigmoid);
    const r = add(add(mv(this.Wr, x, H, I), mv(this.Ur, h, H, H)), this.br).map(sigmoid);
    const rh = new Float64Array(H).map((_, k) => (r[k] ?? 0) * (h[k] ?? 0));
    const hTilde = add(add(mv(this.Wh, x, H, I), mv(this.Uh, rh, H, H)), this.bh).map(tanh);
    return new Float64Array(H).map((_, k) => (1 - (z[k] ?? 0)) * (h[k] ?? 0) + (z[k] ?? 0) * (hTilde[k] ?? 0));
  }
}

export class DotProductAttention {
  constructor(private scale = true) {}

  forward(queries: Float64Array[], keys: Float64Array[], values: Float64Array[]): Float64Array[] {
    const q = queries.length, k = keys.length, d = queries[0]?.length ?? 1;
    const scaleFactor = this.scale ? Math.sqrt(d) : 1;
    return queries.map((query, i) => {
      const scores = new Float64Array(k).map((_, j) => {
        const dot = query.reduce((s, v, di) => s + v * (keys[j]![di] ?? 0), 0);
        return dot / scaleFactor;
      });
      // Softmax
      const maxScore = Math.max(...scores);
      const expScores = scores.map(s => Math.exp(s - maxScore));
      const sumExp = expScores.reduce((s, v) => s + v, 0);
      const weights = expScores.map(v => v / sumExp);
      const dv = values[0]?.length ?? 1;
      return new Float64Array(dv).map((_, di) =>
        weights.reduce((s, w, j) => s + w * (values[j]![di] ?? 0), 0)
      );
      void i;
    });
  }
}
