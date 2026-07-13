/**
 * Additional utility functions — numerical integration, statistics, sparse ops.
 */

export function trapz(y: Float64Array, x?: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < y.length - 1; i++) {
    const dx = x ? ((x[i + 1] ?? 0) - (x[i] ?? 0)) : 1;
    sum += dx * ((y[i] ?? 0) + (y[i + 1] ?? 0)) / 2;
  }
  return sum;
}

export function simpson(y: Float64Array, x?: Float64Array): number {
  if (y.length < 3) return trapz(y, x);
  let sum = 0;
  const n = y.length;
  for (let i = 0; i < n - 2; i += 2) {
    const dx = x ? ((x[i + 2] ?? 0) - (x[i] ?? 0)) / 6 : 1.0 / 3;
    sum += dx * ((y[i] ?? 0) + 4 * (y[i + 1] ?? 0) + (y[i + 2] ?? 0));
  }
  if (n % 2 === 0) {
    const dx = x ? ((x[n - 1] ?? 0) - (x[n - 2] ?? 0)) : 1;
    sum += dx * ((y[n - 2] ?? 0) + (y[n - 1] ?? 0)) / 2;
  }
  return sum;
}

export function histogramBins(data: Float64Array, nBins: number): { counts: Int32Array; edges: Float64Array } {
  const mn = Math.min(...Array.from(data)), mx = Math.max(...Array.from(data));
  const binWidth = (mx - mn) / nBins;
  const counts = new Int32Array(nBins);
  const edges = Float64Array.from({ length: nBins + 1 }, (_, i) => mn + i * binWidth);
  for (const v of data) {
    const bin = Math.min(Math.floor((v - mn) / Math.max(binWidth, 1e-12)), nBins - 1);
    if (bin >= 0) counts[bin]!++;
  }
  return { counts, edges };
}

export function bootstrapCI(
  data: Float64Array,
  statFn: (sample: Float64Array) => number,
  nBootstrap = 1000,
  alpha = 0.05,
): { lower: number; upper: number; estimate: number } {
  const n = data.length;
  const bootstrapStats = Float64Array.from({ length: nBootstrap }, () => {
    const sample = Float64Array.from({ length: n }, () => data[Math.floor(Math.random() * n)] ?? 0);
    return statFn(sample);
  });
  const sorted = Array.from(bootstrapStats).sort((a, b) => a - b);
  const lower = sorted[Math.floor(alpha / 2 * nBootstrap)] ?? 0;
  const upper = sorted[Math.floor((1 - alpha / 2) * nBootstrap)] ?? 0;
  return { lower, upper, estimate: statFn(data) };
}

export function jackknifeCI(data: Float64Array, statFn: (sample: Float64Array) => number): { lower: number; upper: number; bias: number; se: number } {
  const n = data.length;
  const fullStat = statFn(data);
  const jackStats = Float64Array.from({ length: n }, (_, i) => {
    const sample = Float64Array.from([...Array.from(data.slice(0, i)), ...Array.from(data.slice(i + 1))]);
    return statFn(sample);
  });
  const meanJack = jackStats.reduce((s, v) => s + v, 0) / n;
  const bias = (n - 1) * (meanJack - fullStat);
  const se = Math.sqrt((n - 1) / n * jackStats.reduce((s, v) => s + (v - meanJack) ** 2, 0));
  const z = 1.96;
  return { lower: fullStat - bias - z * se, upper: fullStat - bias + z * se, bias, se };
}

export function kolmogorovSmirnovTest(x: Float64Array, y: Float64Array): { statistic: number; pValue: number } {
  const xSorted = Array.from(x).sort((a, b) => a - b);
  const ySorted = Array.from(y).sort((a, b) => a - b);
  const n1 = x.length, n2 = y.length;
  let maxDiff = 0;
  for (let i = 0; i < n1; i++) {
    const xi = xSorted[i] ?? 0;
    const ecdfX = (i + 1) / n1;
    const ecdfY = ySorted.filter((v) => v <= xi).length / n2;
    maxDiff = Math.max(maxDiff, Math.abs(ecdfX - ecdfY));
  }
  const en = Math.sqrt((n1 * n2) / (n1 + n2));
  const pValue = Math.exp(-2 * (en * maxDiff) ** 2);
  return { statistic: maxDiff, pValue: Math.min(1, pValue) };
}

export function mannWhitneyU(x: Float64Array, y: Float64Array): { statistic: number; pValue: number } {
  const n1 = x.length, n2 = y.length;
  let u1 = 0;
  for (const xi of x) {
    for (const yi of y) {
      if (xi > yi) u1++;
      else if (xi === yi) u1 += 0.5;
    }
  }
  const u2 = n1 * n2 - u1;
  const u = Math.min(u1, u2);
  const meanU = n1 * n2 / 2;
  const stdU = Math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12);
  const z = (u - meanU) / Math.max(stdU, 1e-12);
  const pValue = 2 * (1 - 0.5 * (1 + Math.erf ? (Math as unknown as { erf: (x: number) => number }).erf(Math.abs(z) / Math.sqrt(2)) : 0.9999));
  return { statistic: u, pValue: Math.max(0, Math.min(1, pValue)) };
}

export function leveneTest(groups: Float64Array[]): { statistic: number; pValue: number } {
  const k = groups.length;
  const N = groups.reduce((s, g) => s + g.length, 0);
  const Z: Float64Array[] = groups.map((g) => {
    const med = Array.from(g).sort((a, b) => a - b)[Math.floor(g.length / 2)] ?? 0;
    return g.map((v) => Math.abs(v - med));
  });
  const Zdot = Z.map((z) => z.reduce((s, v) => s + v, 0) / z.length);
  const ZddotAll = Z.flatMap((z) => Array.from(z));
  const Zddot = ZddotAll.reduce((s, v) => s + v, 0) / N;

  let ssBetween = 0, ssWithin = 0;
  for (let i = 0; i < k; i++) {
    ssBetween += (Z[i] as Float64Array).length * ((Zdot[i] ?? 0) - Zddot) ** 2;
    for (const z of Z[i] as Float64Array) ssWithin += (z - (Zdot[i] ?? 0)) ** 2;
  }
  const statistic = ((N - k) / (k - 1)) * (ssBetween / Math.max(ssWithin, 1e-12));
  const pValue = Math.exp(-0.5 * statistic);
  return { statistic, pValue: Math.max(0, Math.min(1, pValue)) };
}

export function kroneckerDelta(i: number, j: number): number {
  return i === j ? 1 : 0;
}

export function softmax(x: Float64Array): Float64Array {
  const maxV = Math.max(...Array.from(x));
  const exps = x.map((v) => Math.exp(v - maxV));
  const sum = exps.reduce((s, v) => s + v, 0);
  return exps.map((v) => v / Math.max(sum, 1e-12));
}

export function logSumExp(x: Float64Array): number {
  const maxV = Math.max(...Array.from(x));
  return maxV + Math.log(x.reduce((s, v) => s + Math.exp(v - maxV), 0));
}

export function sparseDot(indices: Int32Array, values: Float64Array, dense: Float64Array): number {
  return Array.from(indices).reduce((s, idx, k) => s + (values[k] ?? 0) * (dense[idx] ?? 0), 0);
}

export function sparseMatVec(rowPtr: Int32Array, colIdx: Int32Array, data: Float64Array, x: Float64Array): Float64Array {
  const n = rowPtr.length - 1;
  return Float64Array.from({ length: n }, (_, i) => {
    let sum = 0;
    for (let j = rowPtr[i] ?? 0; j < (rowPtr[i + 1] ?? 0); j++) {
      sum += (data[j] ?? 0) * (x[colIdx[j] ?? 0] ?? 0);
    }
    return sum;
  });
}
