/**
 * Additional dataset generators: make_moons, make_circles, make_blobs extensions.
 * Mirrors sklearn.datasets extras.
 */

export function makeMoons(
  nSamples = 100,
  noise = 0.1,
  randomState = 0,
): { X: Float64Array[]; y: Int32Array } {
  let rng = randomState;
  const nextRand = (): number => {
    rng = (rng * 1664525 + 1013904223) >>> 0;
    return rng / 4294967296;
  };
  const boxMuller = (): number => {
    const u = nextRand();
    const v = nextRand();
    return Math.sqrt(-2 * Math.log(u + 1e-10)) * Math.cos(2 * Math.PI * v);
  };

  const nEach = Math.floor(nSamples / 2);
  const X: Float64Array[] = [];
  const y: number[] = [];

  for (let i = 0; i < nEach; i++) {
    const angle = (Math.PI * i) / nEach;
    X.push(new Float64Array([Math.cos(angle) + noise * boxMuller(), Math.sin(angle) + noise * boxMuller()]));
    y.push(0);
  }
  for (let i = 0; i < nSamples - nEach; i++) {
    const angle = (Math.PI * i) / (nSamples - nEach);
    X.push(new Float64Array([1 - Math.cos(angle) + noise * boxMuller(), 1 - Math.sin(angle) - 0.5 + noise * boxMuller()]));
    y.push(1);
  }

  return { X, y: new Int32Array(y) };
}

export function makeCircles(
  nSamples = 100,
  noise = 0.1,
  factor = 0.8,
  randomState = 0,
): { X: Float64Array[]; y: Int32Array } {
  let rng = randomState;
  const nextRand = (): number => {
    rng = (rng * 1664525 + 1013904223) >>> 0;
    return rng / 4294967296;
  };
  const boxMuller = (): number => {
    const u = nextRand();
    const v = nextRand();
    return Math.sqrt(-2 * Math.log(u + 1e-10)) * Math.cos(2 * Math.PI * v);
  };

  const nOuter = Math.floor(nSamples / 2);
  const nInner = nSamples - nOuter;
  const X: Float64Array[] = [];
  const y: number[] = [];

  for (let i = 0; i < nOuter; i++) {
    const angle = (2 * Math.PI * i) / nOuter;
    X.push(new Float64Array([Math.cos(angle) + noise * boxMuller(), Math.sin(angle) + noise * boxMuller()]));
    y.push(0);
  }
  for (let i = 0; i < nInner; i++) {
    const angle = (2 * Math.PI * i) / nInner;
    X.push(new Float64Array([factor * Math.cos(angle) + noise * boxMuller(), factor * Math.sin(angle) + noise * boxMuller()]));
    y.push(1);
  }

  return { X, y: new Int32Array(y) };
}

export function makeSwissRoll(
  nSamples = 100,
  noise = 0.0,
  randomState = 0,
): { X: Float64Array[]; t: Float64Array } {
  let rng = randomState;
  const nextRand = (): number => {
    rng = (rng * 1664525 + 1013904223) >>> 0;
    return rng / 4294967296;
  };
  const boxMuller = (): number => {
    const u = nextRand();
    const v = nextRand();
    return Math.sqrt(-2 * Math.log(u + 1e-10)) * Math.cos(2 * Math.PI * v);
  };

  const t = new Float64Array(nSamples);
  const X: Float64Array[] = [];

  for (let i = 0; i < nSamples; i++) {
    const ti = (1.5 + 2.5 * nextRand()) * Math.PI;
    t[i] = ti;
    const height = 21 * nextRand();
    X.push(new Float64Array([
      ti * Math.cos(ti) + noise * boxMuller(),
      height + noise * boxMuller(),
      ti * Math.sin(ti) + noise * boxMuller(),
    ]));
  }

  return { X, t };
}

export function makeCheckerboard(
  shape: [number, number] = [10, 10],
  nClusters = 4,
  nSamples = 100,
  noise = 0.0,
  randomState = 0,
): { X: Float64Array[]; rows: Int32Array; cols: Int32Array } {
  let rng = randomState;
  const nextRand = (): number => {
    rng = (rng * 1664525 + 1013904223) >>> 0;
    return rng / 4294967296;
  };

  const [nRows, nCols] = shape;
  const rowClusterSize = nRows / Math.sqrt(nClusters);
  const colClusterSize = nCols / Math.sqrt(nClusters);

  const X: Float64Array[] = [];
  const rows: number[] = [];
  const cols: number[] = [];

  for (let i = 0; i < nSamples; i++) {
    const r = Math.floor(nextRand() * nRows);
    const c = Math.floor(nextRand() * nCols);
    const rCluster = Math.floor(r / rowClusterSize);
    const cCluster = Math.floor(c / colClusterSize);

    const baseVal = (rCluster + cCluster) % 2 === 0 ? 1.0 : 0.0;
    X.push(new Float64Array([
      r + noise * (nextRand() - 0.5),
      c + noise * (nextRand() - 0.5),
      baseVal,
    ]));
    rows.push(r);
    cols.push(c);
  }

  return { X, rows: new Int32Array(rows), cols: new Int32Array(cols) };
}

export function makeSparseCoded(
  nSamples = 100,
  nComponents = 10,
  nFeatures = 20,
  nNonzeroCoefs = 3,
  randomState = 0,
): { X: Float64Array[]; dictionary: Float64Array[]; code: Float64Array[] } {
  let rng = randomState;
  const nextRand = (): number => {
    rng = (rng * 1664525 + 1013904223) >>> 0;
    return (rng / 4294967296) * 2 - 1;
  };

  // Generate random dictionary
  const dictionary: Float64Array[] = Array.from({ length: nComponents }, () => {
    const v = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) v[j] = nextRand();
    let norm = 0;
    for (const vj of v) norm += vj ** 2;
    norm = Math.sqrt(norm);
    if (norm > 0) for (let j = 0; j < nFeatures; j++) v[j] = (v[j] ?? 0) / norm;
    return v;
  });

  // Generate sparse codes
  const code: Float64Array[] = [];
  for (let i = 0; i < nSamples; i++) {
    const c = new Float64Array(nComponents);
    const indices: number[] = [];
    for (let k = 0; k < nNonzeroCoefs; k++) {
      let idx = Math.floor(Math.abs(nextRand()) * nComponents);
      while (indices.includes(idx)) idx = (idx + 1) % nComponents;
      indices.push(idx);
      c[idx] = nextRand();
    }
    code.push(c);
  }

  // Generate X = code @ dictionary
  const X = code.map((c) => {
    const x = new Float64Array(nFeatures);
    for (let k = 0; k < nComponents; k++) {
      const ck = c[k] ?? 0;
      if (ck === 0) continue;
      for (let j = 0; j < nFeatures; j++) {
        x[j] = (x[j] ?? 0) + ck * (dictionary[k]?.[j] ?? 0);
      }
    }
    return x;
  });

  return { X, dictionary, code };
}
