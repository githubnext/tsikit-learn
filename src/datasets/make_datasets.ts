/**
 * Synthetic dataset generators.
 * Mirrors sklearn.datasets: make_classification, make_regression, make_blobs,
 * make_moons, make_circles.
 */

export interface DatasetResult {
  X: Float64Array[];
  y: Float64Array;
}

/** Gaussian random sample. */
function randn(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** Shuffle arrays in place using Fisher-Yates. */
function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = arr[i] as T;
    arr[i] = arr[j] as T;
    arr[j] = tmp;
  }
  return arr;
}

export function makeClassification(
  options: {
    nSamples?: number;
    nFeatures?: number;
    nClasses?: number;
    nInformative?: number;
    nRedundant?: number;
    noise?: number;
    randomState?: number;
  } = {},
): DatasetResult {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 20;
  const nClasses = options.nClasses ?? 2;
  const nInformative = Math.min(options.nInformative ?? 2, nFeatures);
  const noise = options.noise ?? 0.0;

  const X: Float64Array[] = Array.from({ length: nSamples }, () => new Float64Array(nFeatures));
  const y = new Float64Array(nSamples);

  // Cluster centers for each class
  const centers: Float64Array[] = Array.from({ length: nClasses }, () => {
    const center = new Float64Array(nInformative);
    for (let j = 0; j < nInformative; j++) center[j] = randn() * 2;
    return center;
  });

  for (let i = 0; i < nSamples; i++) {
    const cls = i % nClasses;
    y[i] = cls;
    const xi = X[i] ?? new Float64Array(nFeatures);
    const center = centers[cls] ?? new Float64Array(nInformative);

    for (let j = 0; j < nInformative; j++) {
      xi[j] = (center[j] ?? 0) + randn() * 0.5 + randn() * noise;
    }
    for (let j = nInformative; j < nFeatures; j++) {
      xi[j] = randn();
    }
  }

  return { X, y };
}

export function makeRegression(
  options: {
    nSamples?: number;
    nFeatures?: number;
    nInformative?: number;
    noise?: number;
    bias?: number;
  } = {},
): DatasetResult & { coef: Float64Array } {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 100;
  const nInformative = Math.min(options.nInformative ?? 10, nFeatures);
  const noise = options.noise ?? 0.0;
  const bias = options.bias ?? 0.0;

  const coef = new Float64Array(nFeatures);
  for (let j = 0; j < nInformative; j++) {
    coef[j] = randn() * 10;
  }

  const X: Float64Array[] = Array.from({ length: nSamples }, () => {
    const xi = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) xi[j] = randn();
    return xi;
  });

  const y = new Float64Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    let yi = bias;
    const xi = X[i] ?? new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      yi += (xi[j] ?? 0) * (coef[j] ?? 0);
    }
    y[i] = yi + randn() * noise;
  }

  return { X, y, coef };
}

export function makeBlobs(
  options: {
    nSamples?: number;
    nFeatures?: number;
    centers?: number | Float64Array[];
    clusterStd?: number;
  } = {},
): DatasetResult {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 2;
  const clusterStd = options.clusterStd ?? 1.0;

  let centers: Float64Array[];
  if (typeof options.centers === "number" || options.centers === undefined) {
    const k = typeof options.centers === "number" ? options.centers : 3;
    centers = Array.from({ length: k }, () => {
      const c = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) c[j] = (Math.random() - 0.5) * 20;
      return c;
    });
  } else {
    centers = options.centers;
  }

  const k = centers.length;
  const X: Float64Array[] = [];
  const y: number[] = [];

  for (let i = 0; i < nSamples; i++) {
    const cls = i % k;
    const center = centers[cls] ?? new Float64Array(nFeatures);
    const xi = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      xi[j] = (center[j] ?? 0) + randn() * clusterStd;
    }
    X.push(xi);
    y.push(cls);
  }

  const order = shuffle(Array.from({ length: nSamples }, (_, i) => i));
  return {
    X: order.map((i) => X[i] ?? new Float64Array(nFeatures)),
    y: new Float64Array(order.map((i) => y[i] ?? 0)),
  };
}

export function makeMoons(
  options: { nSamples?: number; noise?: number } = {},
): DatasetResult {
  const nSamples = options.nSamples ?? 100;
  const noise = options.noise ?? 0.0;
  const half = Math.floor(nSamples / 2);

  const X: Float64Array[] = [];
  const y: number[] = [];

  for (let i = 0; i < half; i++) {
    const angle = (Math.PI * i) / half;
    X.push(new Float64Array([Math.cos(angle) + randn() * noise, Math.sin(angle) + randn() * noise]));
    y.push(0);
  }
  for (let i = 0; i < nSamples - half; i++) {
    const angle = (Math.PI * i) / (nSamples - half);
    X.push(new Float64Array([1 - Math.cos(angle) + randn() * noise, 1 - Math.sin(angle) - 0.5 + randn() * noise]));
    y.push(1);
  }

  const order = shuffle(Array.from({ length: nSamples }, (_, i) => i));
  return {
    X: order.map((i) => X[i] ?? new Float64Array(2)),
    y: new Float64Array(order.map((i) => y[i] ?? 0)),
  };
}

export function makeCircles(
  options: { nSamples?: number; noise?: number; factor?: number } = {},
): DatasetResult {
  const nSamples = options.nSamples ?? 100;
  const noise = options.noise ?? 0.0;
  const factor = options.factor ?? 0.8;
  const half = Math.floor(nSamples / 2);

  const X: Float64Array[] = [];
  const y: number[] = [];

  for (let i = 0; i < half; i++) {
    const angle = (2 * Math.PI * i) / half;
    X.push(new Float64Array([Math.cos(angle) + randn() * noise, Math.sin(angle) + randn() * noise]));
    y.push(0);
  }
  for (let i = 0; i < nSamples - half; i++) {
    const angle = (2 * Math.PI * i) / (nSamples - half);
    X.push(new Float64Array([factor * Math.cos(angle) + randn() * noise, factor * Math.sin(angle) + randn() * noise]));
    y.push(1);
  }

  const order = shuffle(Array.from({ length: nSamples }, (_, i) => i));
  return {
    X: order.map((i) => X[i] ?? new Float64Array(2)),
    y: new Float64Array(order.map((i) => y[i] ?? 0)),
  };
}
