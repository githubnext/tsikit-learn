/**
 * Information-theoretic and advanced regression metrics.
 */

export function mutualInfoScore(labels1: Int32Array, labels2: Int32Array): number {
  const n = labels1.length;
  const c1 = Array.from(new Set(Array.from(labels1)));
  const c2 = Array.from(new Set(Array.from(labels2)));
  let mi = 0;
  for (const a of c1) {
    for (const b of c2) {
      const nAB = Array.from(labels1).filter((v, i) => v === a && labels2[i] === b).length;
      if (nAB === 0) continue;
      const nA = Array.from(labels1).filter(v => v === a).length;
      const nB = Array.from(labels2).filter(v => v === b).length;
      mi += (nAB / n) * Math.log((n * nAB) / (nA * nB));
    }
  }
  return mi;
}

export function normalizedMutualInfoScore(labels1: Int32Array, labels2: Int32Array): number {
  const mi = mutualInfoScore(labels1, labels2);
  const n = labels1.length;
  const entropy = (labels: Int32Array) => {
    const counts: Record<number, number> = {};
    Array.from(labels).forEach(v => { counts[v] = (counts[v] ?? 0) + 1; });
    return Object.values(counts).reduce((s, c) => s - (c / n) * Math.log(c / n), 0);
  };
  const h1 = entropy(labels1), h2 = entropy(labels2);
  const denom = (h1 + h2) / 2;
  return denom > 0 ? mi / denom : 0;
}

export function adjustedMutualInfoScore(labels1: Int32Array, labels2: Int32Array): number {
  const mi = mutualInfoScore(labels1, labels2);
  const n = labels1.length;
  const c1 = Array.from(new Set(Array.from(labels1)));
  const c2 = Array.from(new Set(Array.from(labels2)));
  const r = c1.length, c = c2.length;
  // EMI approximation
  const expectedMI = (r * c) / (n * n);
  const maxMI = Math.log(Math.min(r, c));
  return maxMI > expectedMI ? (mi - expectedMI) / (maxMI - expectedMI) : 0;
}

export function dualityGap(primal: number, dual: number): number {
  return Math.abs(primal - dual);
}

export function informationCriteria(logLikelihood: number, nParams: number, n: number, criterion: 'aic' | 'bic' = 'bic'): number {
  if (criterion === 'aic') return 2 * nParams - 2 * logLikelihood;
  return nParams * Math.log(n) - 2 * logLikelihood;
}
