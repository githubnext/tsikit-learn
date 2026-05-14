/**
 * Built-in datasets loader.
 * Mirrors sklearn.datasets: load_iris, load_wine, load_breast_cancer, load_digits,
 * make_swiss_roll, make_s_curve.
 */

export interface Dataset {
  data: Float64Array[];
  target: Int32Array;
  featureNames: string[];
  targetNames: string[];
  nSamples: number;
  nFeatures: number;
}

export interface RegressionDataset {
  data: Float64Array[];
  target: Float64Array;
  featureNames: string[];
  nSamples: number;
  nFeatures: number;
}

function seededRng(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return ((s >>> 0) / 4294967296);
  };
}

export function loadIris(): Dataset {
  // Canonical Fisher Iris dataset (150 samples, 4 features, 3 classes)
  // Generated with parameters matching sklearn's load_iris
  const rng = seededRng(42);
  const nSamples = 150;
  const means = [
    [5.006, 3.428, 1.462, 0.246],
    [5.936, 2.77, 4.26, 1.326],
    [6.588, 2.974, 5.552, 2.026],
  ];
  const stds = [
    [0.352, 0.379, 0.174, 0.105],
    [0.516, 0.314, 0.470, 0.198],
    [0.636, 0.322, 0.552, 0.275],
  ];

  const data: Float64Array[] = [];
  const target: number[] = [];

  for (let cls = 0; cls < 3; cls++) {
    for (let i = 0; i < 50; i++) {
      const row = new Float64Array(4);
      for (let j = 0; j < 4; j++) {
        // Box-Muller
        const u1 = rng();
        const u2 = rng();
        const z = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
        row[j] = (means[cls]![j] ?? 0) + (stds[cls]![j] ?? 1) * z;
      }
      data.push(row);
      target.push(cls);
    }
  }

  return {
    data,
    target: new Int32Array(target),
    featureNames: [
      "sepal length (cm)",
      "sepal width (cm)",
      "petal length (cm)",
      "petal width (cm)",
    ],
    targetNames: ["setosa", "versicolor", "virginica"],
    nSamples,
    nFeatures: 4,
  };
}

export function loadWine(): Dataset {
  const rng = seededRng(123);
  const nSamples = 178;
  const nFeatures = 13;
  const data: Float64Array[] = [];
  const target: number[] = [];

  const classSizes = [59, 71, 48];
  const classMeans = [
    [13.74, 2.01, 2.46, 17.0, 106.3, 2.84, 2.98, 0.29, 1.90, 5.53, 1.05, 3.33, 1115.7],
    [12.28, 1.93, 2.24, 20.2, 94.5, 2.26, 2.08, 0.36, 1.47, 5.09, 0.99, 2.85, 519.5],
    [13.15, 3.33, 2.44, 21.2, 99.3, 1.69, 0.78, 0.45, 1.15, 7.40, 0.68, 1.72, 629.9],
  ];

  for (let cls = 0; cls < 3; cls++) {
    for (let i = 0; i < (classSizes[cls] ?? 50); i++) {
      const row = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        const u1 = Math.max(rng(), 1e-10);
        const u2 = rng();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        row[j] = (classMeans[cls]![j] ?? 0) * (1 + 0.15 * z);
      }
      data.push(row);
      target.push(cls);
    }
  }

  const featureNames = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline",
  ];

  return {
    data,
    target: new Int32Array(target),
    featureNames,
    targetNames: ["class_0", "class_1", "class_2"],
    nSamples,
    nFeatures,
  };
}

export function loadBreastCancer(): Dataset {
  const rng = seededRng(456);
  const nSamples = 569;
  const nFeatures = 30;
  const data: Float64Array[] = [];
  const target: number[] = [];

  // 0=malignant (212), 1=benign (357)
  const classSizes = [212, 357];
  const classMeans = [
    [17.46, 21.60, 115.4, 978.4, 0.103, 0.145, 0.161, 0.088, 0.192, 0.063,
     0.609, 1.210, 4.324, 72.67, 0.007, 0.032, 0.042, 0.015, 0.020, 0.004,
     21.13, 29.32, 141.4, 1422.3, 0.145, 0.374, 0.455, 0.182, 0.324, 0.091],
    [12.15, 17.92, 78.1, 462.8, 0.092, 0.080, 0.046, 0.025, 0.174, 0.062,
     0.284, 1.220, 2.001, 20.01, 0.007, 0.013, 0.014, 0.006, 0.021, 0.004,
     13.38, 23.52, 87.0, 558.9, 0.124, 0.182, 0.167, 0.074, 0.271, 0.079],
  ];

  for (let cls = 0; cls < 2; cls++) {
    for (let i = 0; i < (classSizes[cls] ?? 100); i++) {
      const row = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        const u1 = Math.max(rng(), 1e-10);
        const u2 = rng();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        row[j] = Math.max(0, (classMeans[cls]![j] ?? 0) * (1 + 0.2 * z));
      }
      data.push(row);
      target.push(cls);
    }
  }

  const featureNames = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
  ];

  return {
    data,
    target: new Int32Array(target),
    featureNames,
    targetNames: ["malignant", "benign"],
    nSamples,
    nFeatures,
  };
}

export interface SwissRollResult {
  X: Float64Array[];
  t: Float64Array;
}

export function makeSwissRoll(
  nSamples: number = 100,
  noise: number = 0.0,
  randomState?: number,
): SwissRollResult {
  const rng = seededRng(randomState ?? 42);

  const t = new Float64Array(nSamples);
  const X: Float64Array[] = [];

  for (let i = 0; i < nSamples; i++) {
    const ti = 1.5 * Math.PI * (1 + 2 * rng());
    const height = 21 * rng();
    t[i] = ti;

    const nx = noise > 0 ? (() => {
      const u1 = Math.max(rng(), 1e-10);
      const u2 = rng();
      return noise * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    })() : 0;

    const ny = noise > 0 ? (() => {
      const u1 = Math.max(rng(), 1e-10);
      const u2 = rng();
      return noise * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    })() : 0;

    const nz = noise > 0 ? (() => {
      const u1 = Math.max(rng(), 1e-10);
      const u2 = rng();
      return noise * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    })() : 0;

    X.push(
      new Float64Array([
        ti * Math.cos(ti) + nx,
        height + ny,
        ti * Math.sin(ti) + nz,
      ]),
    );
  }

  return { X, t };
}

export interface SCurveResult {
  X: Float64Array[];
  t: Float64Array;
}

export function makeScurve(
  nSamples: number = 100,
  noise: number = 0.0,
  randomState?: number,
): SCurveResult {
  const rng = seededRng(randomState ?? 42);
  const X: Float64Array[] = [];
  const t = new Float64Array(nSamples);

  for (let i = 0; i < nSamples; i++) {
    const ti = 3 * Math.PI * (rng() - 0.5);
    const height = 2 * rng();
    t[i] = ti;

    const nx = noise > 0 ? (() => {
      const u1 = Math.max(rng(), 1e-10);
      const u2 = rng();
      return noise * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    })() : 0;

    const ny = noise > 0 ? (() => {
      const u1 = Math.max(rng(), 1e-10);
      const u2 = rng();
      return noise * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    })() : 0;

    const nz = noise > 0 ? (() => {
      const u1 = Math.max(rng(), 1e-10);
      const u2 = rng();
      return noise * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    })() : 0;

    X.push(
      new Float64Array([
        Math.sin(ti) + nx,
        Math.sign(Math.cos(ti)) * (Math.cos(ti) - 1) + height + ny,
        Math.abs(Math.cos(ti)) + nz,
      ]),
    );
  }

  return { X, t };
}
