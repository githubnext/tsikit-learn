/**
 * Utility extensions: RandomizedSearchCV, ParameterSampler, SparseMatrixUtils
 * Port of sklearn.utils extensions
 */

export function shuffle<T>(arr: T[], randomState = 42): T[] {
  const result = [...arr];
  let seed = randomState;
  const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = result[i]!;
    result[i] = result[j]!;
    result[j] = tmp;
  }
  return result;
}

export function resample<T>(arr: T[], opts: { nSamples?: number; replace?: boolean; randomState?: number } = {}): T[] {
  const n = opts.nSamples ?? arr.length;
  const replace = opts.replace ?? true;
  let seed = opts.randomState ?? 0;
  const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
  if (replace) {
    return Array.from({ length: n }, () => arr[Math.floor(rng() * arr.length)]!);
  }
  const shuffled = shuffle(arr, opts.randomState ?? 0);
  return shuffled.slice(0, Math.min(n, arr.length));
}

export class ParameterSamplerExt {
  paramDistributions: Record<string, number[] | { low: number; high: number; log?: boolean }>;
  nIter: number;
  randomState: number;

  constructor(opts: {
    paramDistributions?: Record<string, number[] | { low: number; high: number; log?: boolean }>;
    nIter?: number;
    randomState?: number;
  } = {}) {
    this.paramDistributions = opts.paramDistributions ?? {};
    this.nIter = opts.nIter ?? 10;
    this.randomState = opts.randomState ?? 0;
  }

  *[Symbol.iterator](): Iterator<Record<string, number>> {
    let seed = this.randomState;
    const rng = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };
    for (let i = 0; i < this.nIter; i++) {
      const params: Record<string, number> = {};
      for (const [key, dist] of Object.entries(this.paramDistributions)) {
        if (Array.isArray(dist)) {
          params[key] = dist[Math.floor(rng() * dist.length)] ?? 0;
        } else {
          const u = rng();
          params[key] = dist.log
            ? Math.exp(Math.log(dist.low) + u * (Math.log(dist.high) - Math.log(dist.low)))
            : dist.low + u * (dist.high - dist.low);
        }
      }
      yield params;
    }
  }

  toArray(): Record<string, number>[] {
    return [...this];
  }
}

export interface SparseMatrix {
  data: Float64Array;
  indices: Int32Array;
  indptr: Int32Array;
  shape: [number, number];
  format: "csr" | "csc";
}

export function denseToSparse(X: Float64Array[], threshold = 0): SparseMatrix {
  const m = X.length;
  const n = X[0]?.length ?? 0;
  const data: number[] = [];
  const indices: number[] = [];
  const indptr: number[] = [0];
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (Math.abs(X[i]![j] ?? 0) > threshold) {
        data.push(X[i]![j] ?? 0);
        indices.push(j);
      }
    }
    indptr.push(data.length);
  }
  return {
    data: Float64Array.from(data),
    indices: Int32Array.from(indices),
    indptr: Int32Array.from(indptr),
    shape: [m, n],
    format: "csr",
  };
}

export function sparseToDense(mat: SparseMatrix): Float64Array[] {
  const [m, n] = mat.shape;
  return Array.from({ length: m }, (_, i) => {
    const row = new Float64Array(n);
    const start = mat.indptr[i] ?? 0;
    const end = mat.indptr[i + 1] ?? 0;
    for (let k = start; k < end; k++) row[mat.indices[k]!] = mat.data[k] ?? 0;
    return row;
  });
}

export function sparseDot(A: SparseMatrix, B: Float64Array[]): Float64Array[] {
  const [mA] = A.shape;
  const mB = B.length;
  const nB = B[0]?.length ?? 0;
  void mB;
  return Array.from({ length: mA }, (_, i) => {
    const row = new Float64Array(nB);
    const start = A.indptr[i] ?? 0;
    const end = A.indptr[i + 1] ?? 0;
    for (let k = start; k < end; k++) {
      const j = A.indices[k] ?? 0;
      const v = A.data[k] ?? 0;
      const bRow = B[j];
      if (!bRow) continue;
      for (let l = 0; l < nB; l++) row[l] = (row[l] ?? 0) + v * (bRow[l] ?? 0);
    }
    return row;
  });
}

export function computeClassWeight(
  classWeight: "balanced" | Record<number, number>,
  classes: Int32Array | number[],
  y: Int32Array | number[]
): Float64Array {
  const n = y.length;
  const nClasses = classes.length;
  if (classWeight === "balanced") {
    const weights = new Float64Array(nClasses);
    const counts = new Float64Array(nClasses);
    const classArr = Array.from(classes);
    for (const label of y) counts[classArr.indexOf(label as number)] = (counts[classArr.indexOf(label as number)] ?? 0) + 1;
    for (let k = 0; k < nClasses; k++) weights[k] = n / (nClasses * ((counts[k] ?? 1) + 1e-15));
    return weights;
  }
  return Float64Array.from(Array.from(classes).map(c => classWeight[c as number] ?? 1.0));
}

export function columnOrRow(X: Float64Array[], axis: 0 | 1, fn: (arr: Float64Array) => number): Float64Array {
  if (axis === 0) {
    const n = X[0]?.length ?? 0;
    return Float64Array.from({ length: n }, (_, j) => fn(Float64Array.from(X.map(xi => xi[j] ?? 0))));
  }
  return Float64Array.from(X.map(xi => fn(xi)));
}
