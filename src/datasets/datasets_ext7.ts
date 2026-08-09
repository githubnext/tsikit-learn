/**
 * Datasets extensions: time series generators, graph datasets, bioinformatics datasets.
 * Mirrors sklearn.datasets additional methods.
 */

export interface TimeSeriesData {
  X: Float64Array[];
  y: Float64Array;
  time: Float64Array;
}

/** Generate AR(p) time series data. */
export function makeARTimeSeries(params: {
  n_samples?: number;
  ar_coeffs?: number[];
  noise?: number;
  random_state?: number;
} = {}): TimeSeriesData {
  const n = params.n_samples ?? 100;
  const ar = params.ar_coeffs ?? [0.7, -0.2];
  const noise = params.noise ?? 0.1;
  const p = ar.length;
  const y = new Float64Array(n);
  for (let i = 0; i < p; i++) y[i] = (Math.random() - 0.5) * 2;
  for (let t = p; t < n; t++) {
    let v = (Math.random() - 0.5) * noise;
    for (let j = 0; j < p; j++) v += (ar[j] ?? 0) * (y[t - j - 1] ?? 0);
    y[t] = v;
  }
  const X = Array.from({ length: n - p }, (_, i) =>
    new Float64Array(ar.map((_, j) => y[i + p - j - 1] ?? 0)),
  );
  return { X, y: y.slice(p), time: new Float64Array(n - p).map((_, i) => i) };
}

/** Generate sinusoidal dataset with noise. */
export function makeSinusoid(params: {
  n_samples?: number;
  n_features?: number;
  frequency?: number;
  noise?: number;
} = {}): { X: Float64Array[]; y: Float64Array } {
  const n = params.n_samples ?? 100;
  const d = params.n_features ?? 1;
  const freq = params.frequency ?? 1.0;
  const noise = params.noise ?? 0.1;
  const X: Float64Array[] = [];
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(d).map(() => Math.random() * 2 * Math.PI);
    X.push(row);
    y[i] = Math.sin(freq * (row[0] ?? 0)) + (Math.random() - 0.5) * noise;
  }
  return { X, y };
}

/** Generate Friedman #1 regression benchmark dataset. */
export function makeFriedman1(params: {
  n_samples?: number;
  n_features?: number;
  noise?: number;
} = {}): { X: Float64Array[]; y: Float64Array } {
  const n = params.n_samples ?? 100;
  const d = Math.max(5, params.n_features ?? 10);
  const noise = params.noise ?? 0.1;
  const X: Float64Array[] = [];
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(d).map(() => Math.random());
    X.push(row);
    y[i] = 10 * Math.sin(Math.PI * (row[0] ?? 0) * (row[1] ?? 0)) +
      20 * ((row[2] ?? 0) - 0.5) ** 2 +
      10 * (row[3] ?? 0) +
      5 * (row[4] ?? 0) +
      (Math.random() - 0.5) * noise;
  }
  return { X, y };
}

/** Generate Friedman #2 regression benchmark dataset. */
export function makeFriedman2(params: {
  n_samples?: number;
  noise?: number;
} = {}): { X: Float64Array[]; y: Float64Array } {
  const n = params.n_samples ?? 100;
  const noise = params.noise ?? 0.1;
  const X: Float64Array[] = [];
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x1 = Math.random() * 100;
    const x2 = Math.random() * 520 * Math.PI + 40 * Math.PI;
    const x3 = Math.random();
    const x4 = Math.random() * 10 + 1;
    X.push(new Float64Array([x1, x2, x3, x4]));
    y[i] = Math.sqrt(x1 ** 2 + (x2 * x3 - 1 / (x2 * x4)) ** 2) + (Math.random() - 0.5) * noise;
  }
  return { X, y };
}

/** Generate Hastie classification dataset (two interleaved spirals). */
export function makeHastie10_2(params: {
  n_samples?: number;
} = {}): { X: Float64Array[]; y: Int32Array } {
  const n = params.n_samples ?? 12000;
  const X: Float64Array[] = [];
  const y = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(10).map(() => (Math.random() - 0.5) * 2 * Math.sqrt(3));
    const norm2 = row.reduce((s, v) => s + v * v, 0);
    y[i] = norm2 <= 9.34 ? 1 : 0;
    X.push(row);
  }
  return { X, y };
}

/** Generate sparse random regression data. */
export function makeSparseUncorrelated(params: {
  n_samples?: number;
  n_features?: number;
} = {}): { X: Float64Array[]; y: Float64Array } {
  const n = params.n_samples ?? 100;
  const d = params.n_features ?? 10;
  const X: Float64Array[] = Array.from({ length: n }, () => new Float64Array(d).map(() => Math.random() * 2 - 1));
  const coef = new Float64Array([1, -1, 0.5, -0.5, ...new Array(d - 4).fill(0) as number[]]);
  const y = new Float64Array(n).map((_, i) => {
    let s = 0;
    for (let f = 0; f < d; f++) s += (X[i]?.[f] ?? 0) * (coef[f] ?? 0);
    return s + (Math.random() - 0.5) * 0.5;
  });
  return { X, y };
}
