/**
 * Dataset generators — extended synthetic data generation.
 */

export interface DatasetResult {
  X: Float64Array[];
  y: Float64Array | Int32Array;
}

export function makeClusters(
  nSamples = 100,
  nFeatures = 2,
  nCenters = 3,
  clusterStd = 1.0,
  randomState = 42,
): { X: Float64Array[]; y: Int32Array; centers: Float64Array[] } {
  let seed = randomState;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 4294967296;
  };
  const randn = () => {
    const u1 = rand(), u2 = rand();
    return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-15))) * Math.cos(2 * Math.PI * u2);
  };

  const centers = Array.from({ length: nCenters }, () => Float64Array.from({ length: nFeatures }, () => randn() * 4));
  const nPerCluster = Math.floor(nSamples / nCenters);

  const X: Float64Array[] = [];
  const y: number[] = [];

  for (let c = 0; c < nCenters; c++) {
    const n = c === nCenters - 1 ? nSamples - X.length : nPerCluster;
    for (let i = 0; i < n; i++) {
      X.push(Float64Array.from({ length: nFeatures }, (_, d) => (centers[c] as Float64Array)[d] ?? 0 + clusterStd * randn()));
      y.push(c);
    }
  }
  return { X, y: Int32Array.from(y), centers };
}

export function makeMoons(nSamples = 100, noise = 0.1, randomState = 42): DatasetResult {
  let seed = randomState;
  const rand = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 4294967296; };
  const randn = () => {
    const u1 = rand(), u2 = rand();
    return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-15))) * Math.cos(2 * Math.PI * u2);
  };

  const half = Math.floor(nSamples / 2);
  const X: Float64Array[] = [];
  const y: number[] = [];

  for (let i = 0; i < half; i++) {
    const t = Math.PI * i / half;
    X.push(Float64Array.from([Math.cos(t) + noise * randn(), Math.sin(t) + noise * randn()]));
    y.push(0);
  }
  for (let i = 0; i < nSamples - half; i++) {
    const t = Math.PI * i / (nSamples - half);
    X.push(Float64Array.from([1 - Math.cos(t) + noise * randn(), 1 - Math.sin(t) - 0.5 + noise * randn()]));
    y.push(1);
  }
  return { X, y: Int32Array.from(y) };
}

export function makeCircles(nSamples = 100, noise = 0.05, factor = 0.8, randomState = 42): DatasetResult {
  let seed = randomState;
  const rand = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 4294967296; };
  const randn = () => {
    const u1 = rand(), u2 = rand();
    return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-15))) * Math.cos(2 * Math.PI * u2);
  };

  const half = Math.floor(nSamples / 2);
  const X: Float64Array[] = [];
  const y: number[] = [];

  for (let i = 0; i < half; i++) {
    const t = 2 * Math.PI * i / half;
    X.push(Float64Array.from([Math.cos(t) + noise * randn(), Math.sin(t) + noise * randn()]));
    y.push(0);
  }
  for (let i = 0; i < nSamples - half; i++) {
    const t = 2 * Math.PI * i / (nSamples - half);
    X.push(Float64Array.from([factor * Math.cos(t) + noise * randn(), factor * Math.sin(t) + noise * randn()]));
    y.push(1);
  }
  return { X, y: Int32Array.from(y) };
}

export function makeSpiral(nSamples = 100, nLoops = 2, noise = 0.3, randomState = 42): DatasetResult {
  let seed = randomState;
  const rand = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 4294967296; };

  const X: Float64Array[] = [];
  const y: number[] = [];
  for (let cls = 0; cls < 2; cls++) {
    const offset = cls * Math.PI;
    for (let i = 0; i < nSamples; i++) {
      const t = nLoops * Math.PI * i / nSamples;
      const r = t / (nLoops * Math.PI);
      X.push(Float64Array.from([
        r * Math.cos(t + offset) + noise * (rand() - 0.5),
        r * Math.sin(t + offset) + noise * (rand() - 0.5),
      ]));
      y.push(cls);
    }
  }
  return { X, y: Int32Array.from(y) };
}

export function makeRegression(
  nSamples = 100,
  nFeatures = 10,
  nInformative = 5,
  noise = 0.0,
  randomState = 42,
): { X: Float64Array[]; y: Float64Array; coef: Float64Array } {
  let seed = randomState;
  const rand = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 4294967296; };
  const randn = () => {
    const u1 = rand(), u2 = rand();
    return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-15))) * Math.cos(2 * Math.PI * u2);
  };

  const coef = Float64Array.from({ length: nFeatures }, (_, i) => i < nInformative ? randn() : 0);
  const X = Array.from({ length: nSamples }, () => Float64Array.from({ length: nFeatures }, () => randn()));
  const y = Float64Array.from(X, (row) => row.reduce((s, v, d) => s + v * (coef[d] ?? 0), 0) + noise * randn());
  return { X, y, coef };
}

export function loadIris(): { X: Float64Array[]; y: Int32Array; featureNames: string[]; targetNames: string[] } {
  // Built-in Iris-like synthetic data (4 features, 3 classes, 150 samples)
  const data: Array<[number, number, number, number, number]> = [];
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < 50; i++) {
      const sl = [5.1, 5.7, 6.3][c] ?? 5.1;
      const sw = [3.5, 2.8, 3.0][c] ?? 3.5;
      const pl = [1.4, 4.5, 5.9][c] ?? 1.4;
      const pw = [0.2, 1.3, 2.1][c] ?? 0.2;
      data.push([
        sl + (Math.random() - 0.5) * 1.2,
        sw + (Math.random() - 0.5) * 0.8,
        pl + (Math.random() - 0.5) * 1.5,
        pw + (Math.random() - 0.5) * 0.6,
        c,
      ]);
    }
  }
  return {
    X: data.map(([a, b, c, d]) => Float64Array.from([a, b, c, d])),
    y: Int32Array.from(data, (row) => row[4] ?? 0),
    featureNames: ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    targetNames: ["setosa", "versicolor", "virginica"],
  };
}

export function loadBreastCancer(): { X: Float64Array[]; y: Int32Array; featureNames: string[] } {
  const nSamples = 569;
  const nFeatures = 30;
  const featureNames = Array.from({ length: nFeatures }, (_, i) => `feature_${i}`);
  const X = Array.from({ length: nSamples }, (_, i) => {
    const cls = i < 357 ? 0 : 1;
    return Float64Array.from({ length: nFeatures }, (_, j) => Math.random() * 20 + cls * 5 + j * 0.1);
  });
  const y = Int32Array.from({ length: nSamples }, (_, i) => i < 357 ? 0 : 1);
  return { X, y, featureNames };
}
