/**
 * Datasets extensions: text datasets, streaming datasets, anomaly datasets
 */

export interface ClassificationDataset {
  X: Float64Array[];
  y: Int32Array;
  featureNames: string[];
  targetNames: string[];
  description: string;
}

export interface RegressionDataset {
  X: Float64Array[];
  y: Float64Array;
  featureNames: string[];
  description: string;
}

export function makeConcentricCircles(
  nSamples: number = 200,
  nCircles: number = 3,
  noise: number = 0.05,
  randomState: number = 42
): ClassificationDataset {
  let rng = randomState;
  const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };
  const X: Float64Array[] = [], y: number[] = [];
  const perCircle = Math.floor(nSamples / nCircles);

  for (let c = 0; c < nCircles; c++) {
    const radius = (c + 1) / nCircles;
    const n = c < nCircles - 1 ? perCircle : nSamples - c * perCircle;
    for (let i = 0; i < n; i++) {
      const angle = 2 * Math.PI * rand();
      const r = radius + (rand() * 2 - 1) * noise;
      X.push(new Float64Array([r * Math.cos(angle), r * Math.sin(angle)]));
      y.push(c);
    }
  }
  return { X, y: new Int32Array(y), featureNames: ['x', 'y'], targetNames: Array.from({ length: nCircles }, (_, i) => `circle_${i}`), description: 'Concentric circles dataset' };
}

export function makeSpirals(
  nSamples: number = 200,
  nArms: number = 2,
  noise: number = 0.1,
  randomState: number = 42
): ClassificationDataset {
  let rng = randomState;
  const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };
  const X: Float64Array[] = [], y: number[] = [];
  const perArm = Math.floor(nSamples / nArms);

  for (let arm = 0; arm < nArms; arm++) {
    const n = arm < nArms - 1 ? perArm : nSamples - arm * perArm;
    const offset = (2 * Math.PI * arm) / nArms;
    for (let i = 0; i < n; i++) {
      const t = (i / n) * 4 * Math.PI;
      const r = t / (4 * Math.PI);
      X.push(new Float64Array([r * Math.cos(t + offset) + (rand() * 2 - 1) * noise, r * Math.sin(t + offset) + (rand() * 2 - 1) * noise]));
      y.push(arm);
    }
  }
  return { X, y: new Int32Array(y), featureNames: ['x', 'y'], targetNames: Array.from({ length: nArms }, (_, i) => `arm_${i}`), description: 'Spiral dataset' };
}

export function makeCheckerboard(
  nSamples: number = 400,
  nSquares: number = 4,
  noise: number = 0.0,
  randomState: number = 42
): ClassificationDataset {
  let rng = randomState;
  const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };
  const X: Float64Array[] = [], y: number[] = [];

  for (let i = 0; i < nSamples; i++) {
    const x0 = rand(), x1 = rand();
    const g0 = Math.floor(x0 * nSquares), g1 = Math.floor(x1 * nSquares);
    const label = (g0 + g1) % 2;
    X.push(new Float64Array([x0 + (rand() * 2 - 1) * noise, x1 + (rand() * 2 - 1) * noise]));
    y.push(label);
  }
  return { X, y: new Int32Array(y), featureNames: ['x', 'y'], targetNames: ['class_0', 'class_1'], description: 'Checkerboard dataset' };
}

export function makeMultivariateNormal(
  nSamples: number = 200,
  mean: Float64Array = new Float64Array([0, 0]),
  cov: Float64Array[] = [new Float64Array([1, 0]), new Float64Array([0, 1])],
  randomState: number = 42
): Float64Array[] {
  let rng = randomState;
  const randNorm = () => {
    rng = (rng * 1664525 + 1013904223) >>> 0;
    const u1 = rng / 0xffffffff;
    rng = (rng * 1664525 + 1013904223) >>> 0;
    const u2 = rng / 0xffffffff;
    return Math.sqrt(-2 * Math.log(u1 + 1e-300)) * Math.cos(2 * Math.PI * u2);
  };

  const p = mean.length;
  // Cholesky decomposition of cov
  const L = Array.from({ length: p }, () => new Float64Array(p));
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let k = 0; k < j; k++) s += (L[j]?.[k] ?? 0) ** 2;
    L[j]![j] = Math.sqrt(Math.max(0, (cov[j]?.[j] ?? 0) - s));
    for (let i = j + 1; i < p; i++) {
      let t = 0;
      for (let k = 0; k < j; k++) t += (L[i]?.[k] ?? 0) * (L[j]?.[k] ?? 0);
      const ljj = L[j]?.[j] ?? 1e-10;
      L[i]![j] = ((cov[i]?.[j] ?? 0) - t) / ljj;
    }
  }

  return Array.from({ length: nSamples }, () => {
    const z = new Float64Array(p).map(() => randNorm());
    return new Float64Array(p).map((_, i) => (mean[i] ?? 0) + z.reduce((s, zk, k) => s + zk * (L[i]?.[k] ?? 0), 0));
  });
}

export function makeAnomalyDataset(
  nSamples: number = 200,
  nAnomalies: number = 20,
  nFeatures: number = 2,
  anomalyStrength: number = 3.0,
  randomState: number = 42
): { X: Float64Array[]; y: Int32Array } {
  let rng = randomState;
  const randNorm = () => {
    rng = (rng * 1664525 + 1013904223) >>> 0;
    const u1 = rng / 0xffffffff;
    rng = (rng * 1664525 + 1013904223) >>> 0;
    const u2 = rng / 0xffffffff;
    return Math.sqrt(-2 * Math.log(u1 + 1e-300)) * Math.cos(2 * Math.PI * u2);
  };

  const X: Float64Array[] = [], y: number[] = [];
  for (let i = 0; i < nSamples; i++) {
    X.push(new Float64Array(nFeatures).map(() => randNorm()));
    y.push(1);
  }
  for (let i = 0; i < nAnomalies; i++) {
    X.push(new Float64Array(nFeatures).map(() => randNorm() * anomalyStrength + anomalyStrength));
    y.push(-1);
  }
  // Shuffle
  const n = X.length;
  for (let i = n - 1; i > 0; i--) {
    rng = (rng * 1664525 + 1013904223) >>> 0;
    const j = Math.floor((rng / 0xffffffff) * (i + 1));
    const tmp = X[i]!; X[i] = X[j]!; X[j] = tmp;
    const ty = y[i]!; y[i] = y[j]!; y[j] = ty;
  }
  return { X, y: new Int32Array(y) };
}

export function makeFriedman1(
  nSamples: number = 100,
  nFeatures: number = 10,
  noise: number = 0.1,
  randomState: number = 42
): RegressionDataset {
  let rng = randomState;
  const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };
  const randNorm = () => {
    const u1 = rand(), u2 = rand();
    return Math.sqrt(-2 * Math.log(u1 + 1e-300)) * Math.cos(2 * Math.PI * u2);
  };

  const X = Array.from({ length: nSamples }, () => new Float64Array(nFeatures).map(() => rand()));
  const y = new Float64Array(nSamples).map((_, i) => {
    const x = X[i]!;
    return 10 * Math.sin(Math.PI * (x[0] ?? 0) * (x[1] ?? 0)) + 20 * ((x[2] ?? 0) - 0.5) ** 2 + 10 * (x[3] ?? 0) + 5 * (x[4] ?? 0) + randNorm() * noise;
  });
  return { X, y, featureNames: Array.from({ length: nFeatures }, (_, i) => `x${i}`), description: 'Friedman #1 regression benchmark' };
}

export function makeFriedman2(
  nSamples: number = 100,
  noise: number = 0.1,
  randomState: number = 42
): RegressionDataset {
  let rng = randomState;
  const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xffffffff; };
  const randNorm = () => {
    const u1 = rand(), u2 = rand();
    return Math.sqrt(-2 * Math.log(u1 + 1e-300)) * Math.cos(2 * Math.PI * u2);
  };

  const X = Array.from({ length: nSamples }, () => new Float64Array([
    rand() * 100, rand() * 520 * Math.PI + 40 * Math.PI, rand() * 0.9 + 0.1, rand() * 10 + 1
  ]));
  const y = new Float64Array(nSamples).map((_, i) => {
    const x = X[i]!;
    return Math.sqrt((x[0] ?? 0) ** 2 + ((x[1] ?? 0) * (x[2] ?? 0) - 1 / ((x[1] ?? 0) * (x[3] ?? 0) + 1e-10)) ** 2) + randNorm() * noise;
  });
  return { X, y, featureNames: ['x0', 'x1', 'x2', 'x3'], description: 'Friedman #2 regression benchmark' };
}

export function makeLowRankMatrix(
  nSamples: number = 100,
  nFeatures: number = 50,
  effectiveRank: number = 10,
  tailStrength: number = 0.5,
  randomState: number = 42
): Float64Array[] {
  let rng = randomState;
  const randNorm = () => {
    rng = (rng * 1664525 + 1013904223) >>> 0;
    const u1 = rng / 0xffffffff;
    rng = (rng * 1664525 + 1013904223) >>> 0;
    const u2 = rng / 0xffffffff;
    return Math.sqrt(-2 * Math.log(u1 + 1e-300)) * Math.cos(2 * Math.PI * u2);
  };

  const rank = Math.min(nSamples, nFeatures);
  const singular = new Float64Array(rank).map((_, i) => {
    const n = 1 - i / (effectiveRank - 1 + 1e-10);
    const tailFactor = tailStrength * Math.exp(-i / effectiveRank);
    return (n > 0 ? Math.exp(n) : 0) + tailFactor;
  });

  const U = Array.from({ length: nSamples }, () => new Float64Array(rank).map(() => randNorm()));
  const V = Array.from({ length: nFeatures }, () => new Float64Array(rank).map(() => randNorm()));

  // QR orthogonalization (simple Gram-Schmidt)
  for (let j = 0; j < rank; j++) {
    for (let k = 0; k < j; k++) {
      let dot = 0;
      for (let i = 0; i < nSamples; i++) dot += (U[i]?.[j] ?? 0) * (U[i]?.[k] ?? 0);
      for (let i = 0; i < nSamples; i++) U[i]![j] = (U[i]?.[j] ?? 0) - dot * (U[i]?.[k] ?? 0);
    }
    let norm = Math.sqrt(U.reduce((s, u) => s + (u[j] ?? 0) ** 2, 0)) || 1;
    for (let i = 0; i < nSamples; i++) U[i]![j] = (U[i]?.[j] ?? 0) / norm;
  }

  return Array.from({ length: nSamples }, (_, i) =>
    new Float64Array(nFeatures).map((_, j) =>
      Array.from({ length: rank }, (__, k) => (U[i]?.[k] ?? 0) * (singular[k] ?? 0) * (V[j]?.[k] ?? 0)).reduce((s, v) => s + v, 0)
    )
  );
}
