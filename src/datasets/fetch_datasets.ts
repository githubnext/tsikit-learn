/**
 * Dataset fetch utilities: California housing, Covtype, KDDCup99, LFW.
 * Mirrors sklearn.datasets.fetch_* functions.
 */

export interface FetchedDataset {
  data: Float64Array[];
  target: Float64Array;
  featureNames: string[];
  targetNames?: string[];
  description: string;
  nSamples: number;
  nFeatures: number;
}

/**
 * Synthetic version of the California Housing dataset.
 * Real dataset: 20,640 samples, 8 features.
 */
export function fetchCaliforniaHousing(
  options: {
    nSamples?: number;
    seed?: number;
  } = {},
): FetchedDataset {
  const n = options.nSamples ?? 100;
  let seed = options.seed ?? 42;
  function rand(): number {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  }

  const featureNames = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
  ];
  const data: Float64Array[] = [];
  const target = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const medInc = rand() * 15;
    const houseAge = rand() * 52;
    const aveRooms = 3 + rand() * 10;
    const aveBedrms = 1 + rand() * 3;
    const population = 100 + rand() * 35000;
    const aveOccup = 1 + rand() * 10;
    const latitude = 32 + rand() * 10;
    const longitude = -124 + rand() * 10;

    data.push(
      new Float64Array([
        medInc,
        houseAge,
        aveRooms,
        aveBedrms,
        population,
        aveOccup,
        latitude,
        longitude,
      ]),
    );
    target[i] = 0.5 + medInc * 0.3 + rand() * 0.5;
  }

  return {
    data,
    target,
    featureNames,
    description: "California Housing dataset (synthetic)",
    nSamples: n,
    nFeatures: 8,
  };
}

/**
 * Synthetic version of the Forest Cover Type dataset.
 * Real dataset: 581,012 samples, 54 features, 7 classes.
 */
export function fetchCovtype(
  options: { nSamples?: number; seed?: number } = {},
): FetchedDataset {
  const n = options.nSamples ?? 100;
  let seed = options.seed ?? 42;
  function rand(): number {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  }

  const nFeatures = 54;
  const data: Float64Array[] = [];
  const target = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const row = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) row[j] = rand() * 100;
    data.push(row);
    target[i] = (rand() * 7) | 0;
  }

  return {
    data,
    target,
    featureNames: Array.from({ length: nFeatures }, (_, j) => `feature_${j}`),
    targetNames: [
      "Spruce/Fir",
      "Lodgepole Pine",
      "Ponderosa Pine",
      "Cottonwood/Willow",
      "Aspen",
      "Douglas-fir",
      "Krummholz",
    ],
    description: "Forest Cover Type dataset (synthetic)",
    nSamples: n,
    nFeatures,
  };
}

/**
 * Synthetic version of the KDD Cup 1999 dataset.
 */
export function fetchKddcup99(
  options: {
    subset?: "http" | "smtp" | "SF" | "SA" | null;
    nSamples?: number;
    seed?: number;
  } = {},
): FetchedDataset {
  const n = options.nSamples ?? 100;
  let seed = options.seed ?? 42;
  function rand(): number {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  }

  const nFeatures = 41;
  const data: Float64Array[] = [];
  const target = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const row = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) row[j] = rand() * 1000;
    data.push(row);
    target[i] = rand() > 0.8 ? 1 : 0;
  }

  return {
    data,
    target,
    featureNames: Array.from({ length: nFeatures }, (_, j) => `feature_${j}`),
    targetNames: ["normal", "attack"],
    description: `KDD Cup 99 dataset${options.subset ? ` (${options.subset} subset)` : ""} (synthetic)`,
    nSamples: n,
    nFeatures,
  };
}

/**
 * Synthetic version of the Labeled Faces in the Wild (LFW) dataset.
 */
export function fetchLfw(
  options: {
    minFacesPerPerson?: number;
    nComponents?: number;
    nSamples?: number;
    seed?: number;
  } = {},
): FetchedDataset {
  const n = options.nSamples ?? 50;
  const nFeatures = options.nComponents ?? 50 * 37;
  let seed = options.seed ?? 42;
  function rand(): number {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  }

  const data: Float64Array[] = [];
  const target = new Float64Array(n);
  const nPersons = 5;

  for (let i = 0; i < n; i++) {
    const row = new Float64Array(nFeatures);
    const person = (rand() * nPersons) | 0;
    for (let j = 0; j < nFeatures; j++) row[j] = rand() + person * 0.1;
    data.push(row);
    target[i] = person;
  }

  return {
    data,
    target,
    featureNames: Array.from({ length: nFeatures }, (_, j) => `pixel_${j}`),
    targetNames: Array.from({ length: nPersons }, (_, i) => `person_${i}`),
    description: "Labeled Faces in the Wild dataset (synthetic)",
    nSamples: n,
    nFeatures,
  };
}

/**
 * Synthetic version of the Olivetti Faces dataset.
 * Real dataset: 400 samples, 4096 features (64x64), 40 classes.
 */
export function fetchOlivettiFaces(
  options: { seed?: number } = {},
): FetchedDataset {
  const n = 40;
  const nFeatures = 4096;
  let seed = options.seed ?? 42;
  function rand(): number {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  }

  const data: Float64Array[] = [];
  const target = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) row[j] = rand();
    data.push(row);
    target[i] = i % 40;
  }

  return {
    data,
    target,
    featureNames: Array.from({ length: nFeatures }, (_, j) => `pixel_${j}`),
    description: "Olivetti Faces dataset (synthetic)",
    nSamples: n,
    nFeatures,
  };
}

/**
 * Fetch a sample of the 20 Newsgroups dataset.
 * Returns feature vectors (TF-IDF like) for text classification.
 */
export function fetch20Newsgroups(
  options: {
    nSamples?: number;
    nFeatures?: number;
    seed?: number;
    categories?: string[] | null;
  } = {},
): FetchedDataset {
  const n = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 100;
  const categories = options.categories ?? [
    "alt.atheism",
    "comp.graphics",
    "sci.med",
    "soc.religion.christian",
    "talk.politics.guns",
  ];
  const nClasses = categories.length;
  let seed = options.seed ?? 42;
  function rand(): number {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  }

  const data: Float64Array[] = [];
  const target = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const cls = (rand() * nClasses) | 0;
    const row = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) row[j] = rand() > 0.8 ? rand() : 0;
    data.push(row);
    target[i] = cls;
  }

  return {
    data,
    target,
    featureNames: Array.from({ length: nFeatures }, (_, j) => `word_${j}`),
    targetNames: categories,
    description: "20 Newsgroups dataset (synthetic TF-IDF)",
    nSamples: n,
    nFeatures,
  };
}
