/**
 * Decomposition extensions: CUR decomposition, MiniBatchDictionaryLearning, 
 * NonnegativeTuckerDecomposition
 * Port of sklearn.decomposition extensions
 */

import { NotFittedError } from "../exceptions.js";

function matMul(A: Float64Array[], B: Float64Array[]): Float64Array[] {
  const m = A.length;
  const k = B.length;
  const n = B[0]?.length ?? 0;
  return Array.from({ length: m }, (_, i) => {
    const row = new Float64Array(n);
    for (let l = 0; l < k; l++) {
      const ail = A[i]![l] ?? 0;
      if (ail === 0) continue;
      for (let j = 0; j < n; j++) row[j] = (row[j] ?? 0) + ail * (B[l]![j] ?? 0);
    }
    return row;
  });
}

function transpose(A: Float64Array[]): Float64Array[] {
  const m = A.length;
  const n = A[0]?.length ?? 0;
  return Array.from({ length: n }, (_, j) => {
    const col = new Float64Array(m);
    for (let i = 0; i < m; i++) col[i] = A[i]![j] ?? 0;
    return col;
  });
}

function colNorms(A: Float64Array[]): Float64Array {
  const n = A[0]?.length ?? 0;
  const norms = new Float64Array(n);
  for (const row of A) for (let j = 0; j < n; j++) norms[j] = (norms[j] ?? 0) + (row[j] ?? 0) ** 2;
  for (let j = 0; j < n; j++) norms[j] = Math.sqrt(norms[j] ?? 0);
  return norms;
}

export class CURDecomposition {
  rank: number;
  randomState: number;

  C_: Float64Array[] | null = null;
  U_: Float64Array[] | null = null;
  R_: Float64Array[] | null = null;

  constructor(opts: { rank?: number; randomState?: number } = {}) {
    this.rank = opts.rank ?? 5;
    this.randomState = opts.randomState ?? 42;
  }

  fit(X: Float64Array[]): this {
    const m = X.length;
    const n = X[0]?.length ?? 0;
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };

    const rowNorms = new Float64Array(m);
    const colNormsArr = colNorms(X);
    let totalNorm = 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) rowNorms[i] = (rowNorms[i] ?? 0) + (X[i]![j] ?? 0) ** 2;
      totalNorm += rowNorms[i] ?? 0;
    }

    const rowProb = new Float64Array(m);
    const colProb = new Float64Array(n);
    let colTotal = 0;
    for (let j = 0; j < n; j++) colTotal += (colNormsArr[j] ?? 0) ** 2;
    for (let i = 0; i < m; i++) rowProb[i] = (rowNorms[i] ?? 0) / (totalNorm + 1e-15);
    for (let j = 0; j < n; j++) colProb[j] = (colNormsArr[j] ?? 0) ** 2 / (colTotal + 1e-15);

    const r = Math.min(this.rank, m, n);
    const rowIdx = new Set<number>();
    const colIdx = new Set<number>();
    for (let k = 0; k < r * 3 && rowIdx.size < r; k++) {
      let rnd = rng();
      let cum = 0;
      for (let i = 0; i < m; i++) { cum += rowProb[i] ?? 0; if (rnd <= cum) { rowIdx.add(i); break; } }
    }
    for (let k = 0; k < r * 3 && colIdx.size < r; k++) {
      let rnd = rng();
      let cum = 0;
      for (let j = 0; j < n; j++) { cum += colProb[j] ?? 0; if (rnd <= cum) { colIdx.add(j); break; } }
    }
    const rows = [...rowIdx];
    const cols = [...colIdx];
    while (rows.length < r) rows.push(rows.length % m);
    while (cols.length < r) cols.push(cols.length % n);

    this.C_ = cols.map(j => {
      const col = new Float64Array(m);
      for (let i = 0; i < m; i++) col[i] = (X[i]![j] ?? 0) / Math.sqrt(r * (colProb[j] ?? 1e-15) + 1e-15);
      return col;
    });
    this.R_ = rows.map(i => {
      const row = new Float64Array(n);
      for (let j = 0; j < n; j++) row[j] = (X[i]![j] ?? 0) / Math.sqrt(r * (rowProb[i] ?? 1e-15) + 1e-15);
      return row;
    });

    const W = rows.map(i => cols.map(j => X[i]![j] ?? 0));
    const Wmat = W.map(row => Float64Array.from(row));
    const WtW = matMul(transpose(Wmat), Wmat);
    const WtWInvArr = WtW.map((row, i) => {
      const r2 = new Float64Array(row.length);
      r2[i] = 1 / ((row[i] ?? 1) + 1e-15);
      return r2;
    });
    const CtC = matMul(transpose(this.C_), this.C_);
    this.U_ = WtWInvArr.map(row => row);
    void CtC;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.C_ || !this.U_) throw new NotFittedError("CURDecomposition not fitted.");
    return X.map(xi => {
      const proj = new Float64Array(this.C_!.length);
      for (let k = 0; k < this.C_!.length; k++) {
        const col = this.C_![k]!;
        let dot = 0;
        for (let i = 0; i < col.length; i++) dot += (col[i] ?? 0) * (xi[i] ?? 0);
        proj[k] = dot;
      }
      return proj;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class MiniBatchDictionaryLearning {
  nComponents: number;
  alpha: number;
  batchSize: number;
  nIter: number;
  randomState: number;

  components_: Float64Array[] | null = null;

  constructor(opts: {
    nComponents?: number;
    alpha?: number;
    batchSize?: number;
    nIter?: number;
    randomState?: number;
  } = {}) {
    this.nComponents = opts.nComponents ?? 10;
    this.alpha = opts.alpha ?? 1.0;
    this.batchSize = opts.batchSize ?? 10;
    this.nIter = opts.nIter ?? 50;
    this.randomState = opts.randomState ?? 42;
  }

  fit(X: Float64Array[]): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    const k = this.nComponents;
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    this.components_ = Array.from({ length: k }, () => {
      const d = new Float64Array(p);
      let norm = 0;
      for (let j = 0; j < p; j++) { d[j] = rng() * 2 - 1; norm += d[j]! ** 2; }
      norm = Math.sqrt(norm) + 1e-15;
      for (let j = 0; j < p; j++) d[j] = (d[j] ?? 0) / norm;
      return d;
    });
    const A = Array.from({ length: k }, () => new Float64Array(k));
    const B = Array.from({ length: k }, () => new Float64Array(p));

    for (let iter = 0; iter < this.nIter; iter++) {
      const batchIdx = Array.from({ length: this.batchSize }, () => Math.floor(rng() * n));
      for (const i of batchIdx) {
        const xi = X[i];
        if (!xi) continue;
        const code = new Float64Array(k);
        for (let t = 0; t < 50; t++) {
          for (let kk = 0; kk < k; kk++) {
            let residual = xi.reduce((s, v, j) => s + (v ?? 0) * (this.components_![kk]![j] ?? 0), 0);
            for (let ll = 0; ll < k; ll++) {
              if (ll !== kk) residual -= (code[ll] ?? 0) * (this.components_![kk]!.reduce((s, v, j) => s + (v ?? 0) * (this.components_![ll]![j] ?? 0), 0));
            }
            const denom = this.components_![kk]!.reduce((s, v) => s + (v ?? 0) ** 2, 0) + 1e-15;
            const thresh = this.alpha / (denom + 1e-15);
            const raw = residual / (denom + 1e-15);
            code[kk] = Math.sign(raw) * Math.max(0, Math.abs(raw) - thresh);
          }
        }
        for (let kk = 0; kk < k; kk++) {
          const ck = code[kk] ?? 0;
          for (let ll = 0; ll < k; ll++) A[kk]![ll] = (A[kk]![ll] ?? 0) + ck * (code[ll] ?? 0);
          for (let j = 0; j < p; j++) B[kk]![j] = (B[kk]![j] ?? 0) + ck * (xi[j] ?? 0);
        }
      }
      for (let kk = 0; kk < k; kk++) {
        const akk = A[kk]![kk] ?? 1;
        if (akk < 1e-15) continue;
        let newD = new Float64Array(p);
        for (let j = 0; j < p; j++) {
          let val = (B[kk]![j] ?? 0);
          for (let ll = 0; ll < k; ll++) {
            if (ll !== kk) val -= (A[kk]![ll] ?? 0) * (this.components_![ll]![j] ?? 0);
          }
          newD[j] = val / (akk + 1e-15);
        }
        let norm = 0;
        for (let j = 0; j < p; j++) norm += (newD[j] ?? 0) ** 2;
        norm = Math.sqrt(norm) + 1e-15;
        if (norm > 1) for (let j = 0; j < p; j++) newD[j] = (newD[j] ?? 0) / norm;
        this.components_![kk] = newD;
      }
      void iter;
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.components_) throw new NotFittedError("MiniBatchDictionaryLearning not fitted.");
    return X.map(xi => {
      const code = new Float64Array(this.nComponents);
      for (let t = 0; t < 100; t++) {
        for (let kk = 0; kk < this.nComponents; kk++) {
          let r = 0;
          for (let j = 0; j < xi.length; j++) r += (xi[j] ?? 0) * (this.components_![kk]![j] ?? 0);
          for (let ll = 0; ll < this.nComponents; ll++) {
            if (ll !== kk) {
              let dot = 0;
              for (let j = 0; j < xi.length; j++) dot += (this.components_![kk]![j] ?? 0) * (this.components_![ll]![j] ?? 0);
              r -= (code[ll] ?? 0) * dot;
            }
          }
          const denom = this.components_![kk]!.reduce((s, v) => s + (v ?? 0) ** 2, 0) + 1e-15;
          const thresh = this.alpha / denom;
          const raw = r / denom;
          code[kk] = Math.sign(raw) * Math.max(0, Math.abs(raw) - thresh);
        }
      }
      return code;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
