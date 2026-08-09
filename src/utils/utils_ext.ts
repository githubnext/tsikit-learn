/**
 * General utility functions — parallel and functional programming utilities.
 */

export function chunked<T>(array: T[], chunkSize: number): T[][] {
  const result: T[][] = [];
  for (let i = 0; i < array.length; i += chunkSize) {
    result.push(array.slice(i, i + chunkSize));
  }
  return result;
}

export function batched<T, R>(
  items: T[],
  batchSize: number,
  fn: (batch: T[]) => R[],
): R[] {
  const result: R[] = [];
  for (let i = 0; i < items.length; i += batchSize) {
    result.push(...fn(items.slice(i, i + batchSize)));
  }
  return result;
}

export function pairwiseApply<T, R>(
  a: T[],
  b: T[],
  fn: (x: T, y: T) => R,
): R[][] {
  return a.map((ai) => b.map((bj) => fn(ai, bj)));
}

export function argmax(arr: Float64Array | number[]): number {
  let best = 0;
  let bestVal = arr[0] ?? Number.NEGATIVE_INFINITY;
  for (let i = 1; i < arr.length; i++) {
    if ((arr[i] ?? Number.NEGATIVE_INFINITY) > bestVal) {
      bestVal = arr[i] as number;
      best = i;
    }
  }
  return best;
}

export function argmin(arr: Float64Array | number[]): number {
  let best = 0;
  let bestVal = arr[0] ?? Number.POSITIVE_INFINITY;
  for (let i = 1; i < arr.length; i++) {
    if ((arr[i] ?? Number.POSITIVE_INFINITY) < bestVal) {
      bestVal = arr[i] as number;
      best = i;
    }
  }
  return best;
}

export function argsort(arr: Float64Array | number[]): Int32Array {
  const indices = Array.from({ length: arr.length }, (_, i) => i);
  indices.sort((a, b) => (arr[a] ?? 0) - (arr[b] ?? 0));
  return Int32Array.from(indices);
}

export function unique<T>(arr: T[]): T[] {
  return Array.from(new Set(arr));
}

export function uniqueCounts(arr: Int32Array): Map<number, number> {
  const counts = new Map<number, number>();
  for (const v of arr) counts.set(v, (counts.get(v) ?? 0) + 1);
  return counts;
}

export function linspace(start: number, stop: number, num: number): Float64Array {
  if (num <= 0) return new Float64Array(0);
  if (num === 1) return new Float64Array([start]);
  const result = new Float64Array(num);
  const step = (stop - start) / (num - 1);
  for (let i = 0; i < num; i++) result[i] = start + i * step;
  return result;
}

export function arange(start: number, stop: number, step = 1): Float64Array {
  const n = Math.ceil((stop - start) / step);
  const result = new Float64Array(Math.max(0, n));
  for (let i = 0; i < result.length; i++) result[i] = start + i * step;
  return result;
}

export function meshgrid(x: Float64Array, y: Float64Array): [Float64Array[], Float64Array[]] {
  const nx = x.length, ny = y.length;
  const xx: Float64Array[] = Array.from({ length: ny }, () => new Float64Array(x));
  const yy: Float64Array[] = Array.from({ length: ny }, (_, i) => new Float64Array(nx).fill(y[i] ?? 0));
  return [xx, yy];
}

export function logspace(start: number, stop: number, num = 50, base = 10): Float64Array {
  return linspace(start, stop, num).map((v) => base ** v);
}

export function cumsum(arr: Float64Array): Float64Array {
  const result = new Float64Array(arr.length);
  let s = 0;
  for (let i = 0; i < arr.length; i++) {
    s += arr[i] ?? 0;
    result[i] = s;
  }
  return result;
}

export function cumprod(arr: Float64Array): Float64Array {
  const result = new Float64Array(arr.length);
  let p = 1;
  for (let i = 0; i < arr.length; i++) {
    p *= arr[i] ?? 1;
    result[i] = p;
  }
  return result;
}

export function diff(arr: Float64Array, n = 1): Float64Array {
  if (n <= 0 || arr.length === 0) return new Float64Array(arr);
  let result = arr;
  for (let d = 0; d < n; d++) {
    const next = new Float64Array(result.length - 1);
    for (let i = 0; i < next.length; i++) next[i] = (result[i + 1] ?? 0) - (result[i] ?? 0);
    result = next;
  }
  return result;
}

export function vectorize<T, R>(fn: (x: T) => R): (arr: T[]) => R[] {
  return (arr) => arr.map(fn);
}

export function where(condition: Uint8Array | boolean[], x: Float64Array, y: Float64Array): Float64Array {
  const n = condition.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = condition[i] ? (x[i] ?? 0) : (y[i] ?? 0);
  return out;
}

export function clip(arr: Float64Array, min: number, max: number): Float64Array {
  return arr.map((v) => Math.min(Math.max(v, min), max));
}

export function nanMean(arr: Float64Array): number {
  const valid = Array.from(arr).filter((v) => !Number.isNaN(v));
  return valid.length > 0 ? valid.reduce((a, b) => a + b, 0) / valid.length : Number.NaN;
}

export function nanStd(arr: Float64Array): number {
  const valid = Array.from(arr).filter((v) => !Number.isNaN(v));
  if (valid.length < 2) return 0;
  const m = valid.reduce((a, b) => a + b, 0) / valid.length;
  return Math.sqrt(valid.reduce((a, b) => a + (b - m) ** 2, 0) / (valid.length - 1));
}

export function percentile(arr: Float64Array | number[], q: number): number {
  const sorted = Array.from(arr).filter((v) => !Number.isNaN(v)).sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  const idx = (q / 100) * (sorted.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return (sorted[lo] ?? 0) * (hi - idx) + (sorted[hi] ?? 0) * (idx - lo);
}

export function iqr(arr: Float64Array): number {
  return percentile(arr, 75) - percentile(arr, 25);
}
