/**
 * Clustering metrics.
 * Mirrors sklearn.metrics.cluster.
 */

export function silhouetteScore(X: Float64Array[], labels: Int32Array): number {
  const n = X.length;
  if (n === 0) return 0;

  function dist(a: Float64Array, b: Float64Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    }
    return Math.sqrt(s);
  }

  const scores = new Float64Array(n);
  const uniqueLabels = Array.from(new Set(Array.from(labels)));

  for (let i = 0; i < n; i++) {
    const li = labels[i] ?? 0;
    const xi = X[i] ?? new Float64Array(0);

    // a(i): mean distance to same cluster
    let aSumI = 0;
    let aCountI = 0;
    for (let j = 0; j < n; j++) {
      if (i !== j && labels[j] === li) {
        aSumI += dist(xi, X[j] ?? new Float64Array(0));
        aCountI++;
      }
    }
    const ai = aCountI > 0 ? aSumI / aCountI : 0;

    // b(i): min mean distance to other clusters
    let bi = Number.POSITIVE_INFINITY;
    for (const otherLabel of uniqueLabels) {
      if (otherLabel === li) continue;
      let bSum = 0;
      let bCount = 0;
      for (let j = 0; j < n; j++) {
        if (labels[j] === otherLabel) {
          bSum += dist(xi, X[j] ?? new Float64Array(0));
          bCount++;
        }
      }
      if (bCount > 0) {
        const bMean = bSum / bCount;
        if (bMean < bi) bi = bMean;
      }
    }
    if (!Number.isFinite(bi)) bi = 0;

    const maxAB = Math.max(ai, bi);
    scores[i] = maxAB > 0 ? (bi - ai) / maxAB : 0;
  }

  return Array.from(scores).reduce((a, b) => a + b, 0) / n;
}

export function adjustedRandScore(
  labelsTrue: Int32Array,
  labelsPred: Int32Array,
): number {
  const n = labelsTrue.length;
  const uniqueTrue = Array.from(new Set(Array.from(labelsTrue)));
  const uniquePred = Array.from(new Set(Array.from(labelsPred)));

  // Contingency table
  const contingency = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    const key = `${labelsTrue[i] ?? 0},${labelsPred[i] ?? 0}`;
    contingency.set(key, (contingency.get(key) ?? 0) + 1);
  }

  function comb2(x: number): number {
    return x < 2 ? 0 : (x * (x - 1)) / 2;
  }

  let sumComb = 0;
  for (const val of contingency.values()) {
    sumComb += comb2(val);
  }

  const rowSums = new Map<number, number>();
  const colSums = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const r = labelsTrue[i] ?? 0;
    const c = labelsPred[i] ?? 0;
    rowSums.set(r, (rowSums.get(r) ?? 0) + 1);
    colSums.set(c, (colSums.get(c) ?? 0) + 1);
  }

  let sumRowComb = 0;
  for (const v of rowSums.values()) sumRowComb += comb2(v);
  let sumColComb = 0;
  for (const v of colSums.values()) sumColComb += comb2(v);

  const total = comb2(n);
  const expected = (sumRowComb * sumColComb) / (total || 1);
  const maxVal = (sumRowComb + sumColComb) / 2;
  const denom = maxVal - expected;

  return denom === 0
    ? sumComb === expected
      ? 1
      : 0
    : (sumComb - expected) / denom;
}

export function homogeneityScore(
  labelsTrue: Int32Array,
  labelsPred: Int32Array,
): number {
  const n = labelsTrue.length;
  if (n === 0) return 1;

  function entropy(labels: Int32Array): number {
    const counts = new Map<number, number>();
    for (const l of labels) counts.set(l, (counts.get(l) ?? 0) + 1);
    let h = 0;
    for (const c of counts.values()) {
      const p = c / n;
      h -= p * Math.log(p);
    }
    return h;
  }

  const hC = entropy(labelsTrue);
  if (hC === 0) return 1;

  // Conditional entropy H(C|K)
  const contingency = new Map<number, Map<number, number>>();
  for (let i = 0; i < n; i++) {
    const k = labelsPred[i] ?? 0;
    const c = labelsTrue[i] ?? 0;
    if (!contingency.has(k)) contingency.set(k, new Map());
    const m = contingency.get(k) as Map<number, number>;
    m.set(c, (m.get(c) ?? 0) + 1);
  }

  const kCounts = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const k = labelsPred[i] ?? 0;
    kCounts.set(k, (kCounts.get(k) ?? 0) + 1);
  }

  let hCK = 0;
  for (const [k, cMap] of contingency) {
    const nK = kCounts.get(k) ?? 0;
    for (const cnt of cMap.values()) {
      const p = cnt / nK;
      hCK += (nK / n) * (-p * Math.log(p + 1e-15));
    }
  }

  return 1 - hCK / hC;
}
