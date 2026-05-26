/**
 * Fast pairwise metrics: Wasserstein, Jensen-Shannon Divergence, Hausdorff distance.
 */

export function wassersteinDistance(u: Float64Array, v: Float64Array): number {
  const sorted_u = new Float64Array(u).sort();
  const sorted_v = new Float64Array(v).sort();
  const n = Math.min(sorted_u.length, sorted_v.length);
  let d = 0;
  for (let i = 0; i < n; i++) d += Math.abs((sorted_u[i] ?? 0) - (sorted_v[i] ?? 0));
  return d / Math.max(n, 1);
}

export function jensenShannonDivergence(p: Float64Array, q: Float64Array): number {
  const n = Math.min(p.length, q.length);
  const m = new Float64Array(n);
  let sumP = 0, sumQ = 0;
  for (let i = 0; i < n; i++) { sumP += p[i] ?? 0; sumQ += q[i] ?? 0; }
  for (let i = 0; i < n; i++) m[i] = ((p[i] ?? 0) / Math.max(sumP, 1e-10) + (q[i] ?? 0) / Math.max(sumQ, 1e-10)) / 2;
  let jsd = 0;
  for (let i = 0; i < n; i++) {
    const pi = (p[i] ?? 0) / Math.max(sumP, 1e-10);
    const qi = (q[i] ?? 0) / Math.max(sumQ, 1e-10);
    const mi = m[i] ?? 1e-10;
    if (pi > 1e-10) jsd += pi * Math.log(pi / mi) / 2;
    if (qi > 1e-10) jsd += qi * Math.log(qi / mi) / 2;
  }
  return jsd;
}

export function hausdorffDistance(X: Float64Array[], Y: Float64Array[]): number {
  const d1 = Math.max(...X.map((x) => Math.min(...Y.map((y) => _euclidean(x, y)))));
  const d2 = Math.max(...Y.map((y) => Math.min(...X.map((x) => _euclidean(x, y)))));
  return Math.max(d1, d2);
}

export function modifiedHausdorffDistance(X: Float64Array[], Y: Float64Array[]): number {
  const n = X.length, m = Y.length;
  const d1 = X.reduce((s, x) => s + Math.min(...Y.map((y) => _euclidean(x, y))), 0) / n;
  const d2 = Y.reduce((s, y) => s + Math.min(...X.map((x) => _euclidean(x, y))), 0) / m;
  return Math.max(d1, d2);
}

function _euclidean(a: Float64Array, b: Float64Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
  return Math.sqrt(d);
}

export function pairwiseWasserstein(X: Float64Array[], Y: Float64Array[]): Float64Array[] {
  return X.map((x) => new Float64Array(Y.map((y) => wassersteinDistance(x, y))));
}

export function pairwiseJSD(X: Float64Array[], Y: Float64Array[]): Float64Array[] {
  return X.map((x) => new Float64Array(Y.map((y) => jensenShannonDivergence(x, y))));
}

export function energyDistance(X: Float64Array[], Y: Float64Array[]): number {
  const n = X.length, m = Y.length;
  let eXY = 0, eXX = 0, eYY = 0;
  for (const x of X) for (const y of Y) eXY += _euclidean(x, y);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) eXX += _euclidean(X[i]!, X[j]!);
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) eYY += _euclidean(Y[i]!, Y[j]!);
  return 2 * eXY / (n * m) - eXX / (n * n) - eYY / (m * m);
}

export function bhattacharyyaDistance(p: Float64Array, q: Float64Array): number {
  const n = Math.min(p.length, q.length);
  let coef = 0;
  let sumP = 0, sumQ = 0;
  for (let i = 0; i < n; i++) { sumP += p[i] ?? 0; sumQ += q[i] ?? 0; }
  for (let i = 0; i < n; i++) coef += Math.sqrt((p[i] ?? 0) / Math.max(sumP, 1e-10) * (q[i] ?? 0) / Math.max(sumQ, 1e-10));
  return -Math.log(Math.max(coef, 1e-10));
}
