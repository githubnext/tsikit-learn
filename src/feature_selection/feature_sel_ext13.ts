/**
 * Feature selection extensions: MRMR, Gain Ratio, Information Value
 */

export function mrmrSelection(
  X: Float64Array[],
  y: Int32Array,
  nFeatures: number = 10
): { selectedFeatures: number[]; scores: Float64Array } {
  const n = X.length, p = X[0]?.length ?? 0;
  const selected: number[] = [];
  const remaining = Array.from({ length: p }, (_, i) => i);

  // Compute MI between each feature and y
  const miWithTarget = new Float64Array(p).map((_, j) => mutualInfoClassif(X.map(row => row[j] ?? 0), y));

  while (selected.length < Math.min(nFeatures, p)) {
    if (selected.length === 0) {
      // Pick highest MI with target
      let best = 0, bestScore = -1;
      for (const j of remaining) { if ((miWithTarget[j] ?? 0) > bestScore) { bestScore = miWithTarget[j] ?? 0; best = j; } }
      selected.push(best);
      remaining.splice(remaining.indexOf(best), 1);
    } else {
      // MRMR: max MI with target - mean MI with selected
      let best = remaining[0]!, bestScore = Number.NEGATIVE_INFINITY;
      for (const j of remaining) {
        const miTarget = miWithTarget[j] ?? 0;
        const miSelected = selected.reduce((s, s2) => s + mutualInfoContinuous(
          X.map(row => row[j] ?? 0), X.map(row => row[s2] ?? 0)
        ), 0) / selected.length;
        const score = miTarget - miSelected;
        if (score > bestScore) { bestScore = score; best = j; }
      }
      selected.push(best);
      remaining.splice(remaining.indexOf(best), 1);
    }
  }

  const scores = new Float64Array(p);
  for (const j of selected) scores[j] = miWithTarget[j] ?? 0;
  return { selectedFeatures: selected, scores };
}

export function mutualInfoClassif(feature: number[], y: Int32Array): number {
  const n = feature.length;
  const bins = 10;
  const minV = Math.min(...feature), maxV = Math.max(...feature);
  const binWidth = (maxV - minV) / bins + 1e-10;

  const xBins = feature.map(v => Math.min(Math.floor((v - minV) / binWidth), bins - 1));
  const classes = new Set(Array.from(y));

  let mi = 0;
  for (let b = 0; b < bins; b++) {
    const inBin = Array.from({ length: n }, (_, i) => i).filter(i => xBins[i] === b);
    if (inBin.length === 0) continue;
    const px = inBin.length / n;
    for (const c of classes) {
      const inBinClass = inBin.filter(i => (y[i] ?? 0) === c);
      if (inBinClass.length === 0) continue;
      const pxy = inBinClass.length / n;
      const py = Array.from(y).filter(v => v === c).length / n;
      mi += pxy * Math.log(pxy / (px * py + 1e-10) + 1e-10);
    }
  }
  return Math.max(0, mi);
}

export function mutualInfoContinuous(x: number[], y: number[]): number {
  const n = x.length;
  const bins = 5;
  const xMin = Math.min(...x), xMax = Math.max(...x);
  const yMin = Math.min(...y), yMax = Math.max(...y);
  const xBinW = (xMax - xMin) / bins + 1e-10;
  const yBinW = (yMax - yMin) / bins + 1e-10;
  const xBins = x.map(v => Math.min(Math.floor((v - xMin) / xBinW), bins - 1));
  const yBins = y.map(v => Math.min(Math.floor((v - yMin) / yBinW), bins - 1));

  const jointCounts = new Float64Array(bins * bins);
  for (let i = 0; i < n; i++) jointCounts[(xBins[i] ?? 0) * bins + (yBins[i] ?? 0)] = (jointCounts[(xBins[i] ?? 0) * bins + (yBins[i] ?? 0)] ?? 0) + 1;
  const xCounts = new Float64Array(bins), yCounts = new Float64Array(bins);
  for (let bx = 0; bx < bins; bx++) for (let by = 0; by < bins; by++) {
    xCounts[bx] = (xCounts[bx] ?? 0) + (jointCounts[bx * bins + by] ?? 0);
    yCounts[by] = (yCounts[by] ?? 0) + (jointCounts[bx * bins + by] ?? 0);
  }
  let mi = 0;
  for (let bx = 0; bx < bins; bx++) for (let by = 0; by < bins; by++) {
    const pxy = (jointCounts[bx * bins + by] ?? 0) / n;
    if (pxy === 0) continue;
    const px = (xCounts[bx] ?? 0) / n, py = (yCounts[by] ?? 0) / n;
    mi += pxy * Math.log(pxy / (px * py + 1e-10) + 1e-10);
  }
  return Math.max(0, mi);
}

export function gainRatioSelection(
  X: (string | number)[][],
  y: Int32Array,
  nFeatures: number = 10
): { selectedFeatures: number[]; gainRatios: Float64Array } {
  const p = X[0]?.length ?? 0;
  const gains = new Float64Array(p).map((_, j) => _gainRatio(X.map(row => row[j] ?? ''), y));
  const sorted = Array.from({ length: p }, (_, i) => i).sort((a, b) => (gains[b] ?? 0) - (gains[a] ?? 0));
  return { selectedFeatures: sorted.slice(0, nFeatures), gainRatios: gains };
}

function _gainRatio(feature: (string | number)[], y: Int32Array): number {
  const n = feature.length;
  const yEntropy = _entropy(y);
  const values = new Set(feature);
  let conditionalEntropy = 0, splitInfo = 0;
  for (const v of values) {
    const idx = Array.from({ length: n }, (_, i) => i).filter(i => feature[i] === v);
    const p = idx.length / n;
    conditionalEntropy += p * _entropy(new Int32Array(idx.map(i => y[i] ?? 0)));
    splitInfo -= p > 0 ? p * Math.log2(p) : 0;
  }
  const gain = yEntropy - conditionalEntropy;
  return splitInfo > 0 ? gain / splitInfo : 0;
}

function _entropy(y: Int32Array): number {
  const n = y.length;
  const counts = new Map<number, number>();
  for (const v of y) counts.set(v, (counts.get(v) ?? 0) + 1);
  let h = 0;
  for (const c of counts.values()) { const p = c / n; h -= p > 0 ? p * Math.log2(p) : 0; }
  return h;
}

export function informationValueFeatures(
  X: (string | number)[][],
  y: Int32Array
): { ivScores: Float64Array; woeTransform: (X: (string | number)[][], featureIdx: number) => Float64Array } {
  const p = X[0]?.length ?? 0;
  const nPos = Array.from(y).filter(v => v === 1).length;
  const nNeg = Array.from(y).filter(v => v === 0).length;
  const ivScores = new Float64Array(p);
  const woeTablesArr: Map<string | number, number>[] = [];

  for (let j = 0; j < p; j++) {
    const values = new Set(X.map(row => row[j] ?? ''));
    const woeMaps = new Map<string | number, number>();
    let iv = 0;
    for (const v of values) {
      const idx = Array.from({ length: X.length }, (_, i) => i).filter(i => X[i]?.[j] === v);
      const pos = idx.filter(i => (y[i] ?? 0) === 1).length;
      const neg = idx.filter(i => (y[i] ?? 0) === 0).length;
      const distPos = (pos + 0.5) / (nPos + 1);
      const distNeg = (neg + 0.5) / (nNeg + 1);
      const woe = Math.log(distPos / distNeg);
      woeMaps.set(v, woe);
      iv += (distPos - distNeg) * woe;
    }
    ivScores[j] = iv;
    woeTablesArr.push(woeMaps);
  }

  const woeTransform = (X2: (string | number)[][], featureIdx: number): Float64Array => {
    const woeMap = woeTablesArr[featureIdx];
    return new Float64Array(X2.map(row => woeMap?.get(row[featureIdx] ?? '') ?? 0));
  };

  return { ivScores, woeTransform };
}

export class SelectByInformationValueExt {
  private ivScores_: Float64Array = new Float64Array(0);
  private selectedFeatures_: number[] = [];
  private fitted_ = false;

  constructor(private threshold: number = 0.1) {}

  fit(X: (string | number)[][], y: Int32Array): this {
    const { ivScores } = informationValueFeatures(X, y);
    this.ivScores_ = ivScores;
    this.selectedFeatures_ = Array.from({ length: ivScores.length }, (_, i) => i).filter(i => (ivScores[i] ?? 0) >= this.threshold);
    this.fitted_ = true;
    return this;
  }

  transform(X: (string | number)[][]): (string | number)[][] {
    if (!this.fitted_) throw new Error('Not fitted');
    return X.map(row => this.selectedFeatures_.map(j => row[j] ?? ''));
  }

  get ivScores(): Float64Array { return this.ivScores_; }
  get selectedFeatures(): number[] { return this.selectedFeatures_; }
}

export function chiSquareFeatureSelection(
  X: Float64Array[],
  y: Int32Array,
  nFeatures: number = 10
): { selectedFeatures: number[]; chiScores: Float64Array; pValues: Float64Array } {
  const n = X.length, p = X[0]?.length ?? 0;
  const classes = new Set(Array.from(y));
  const nClasses = classes.size;
  const chiScores = new Float64Array(p);
  const pValues = new Float64Array(p);

  for (let j = 0; j < p; j++) {
    const vals = X.map(row => row[j] ?? 0);
    const bins = 5;
    const minV = Math.min(...vals), maxV = Math.max(...vals);
    const binW = (maxV - minV) / bins + 1e-10;
    const observed = Array.from({ length: bins }, () => new Float64Array(nClasses));
    const classArr = [...classes];

    for (let i = 0; i < n; i++) {
      const bin = Math.min(Math.floor(((vals[i] ?? 0) - minV) / binW), bins - 1);
      const classIdx = classArr.indexOf(y[i] ?? 0);
      if (classIdx >= 0) observed[bin]![classIdx] = (observed[bin]?.[classIdx] ?? 0) + 1;
    }

    const rowSums = observed.map(row => row.reduce((s, v) => s + v, 0));
    const colSums = new Float64Array(nClasses);
    for (const row of observed) for (let c = 0; c < nClasses; c++) colSums[c] = (colSums[c] ?? 0) + (row[c] ?? 0);

    let chi2 = 0;
    for (let b = 0; b < bins; b++) {
      for (let c = 0; c < nClasses; c++) {
        const expected = (rowSums[b] ?? 0) * (colSums[c] ?? 0) / n;
        if (expected > 0) chi2 += ((observed[b]?.[c] ?? 0) - expected) ** 2 / expected;
      }
    }
    chiScores[j] = chi2;
    const df = (bins - 1) * (nClasses - 1);
    pValues[j] = 1 - chi2CDF(chi2, df);
  }

  const sorted = Array.from({ length: p }, (_, i) => i).sort((a, b) => (chiScores[b] ?? 0) - (chiScores[a] ?? 0));
  return { selectedFeatures: sorted.slice(0, nFeatures), chiScores, pValues };
}

function chi2CDF(x: number, df: number): number {
  if (x <= 0) return 0;
  return gammaInc(df / 2, x / 2);
}

function gammaInc(a: number, x: number): number {
  let sum = 1 / a, term = 1 / a;
  for (let n = 1; n < 100; n++) {
    term *= x / (a + n); sum += term;
    if (term < 1e-10) break;
  }
  return Math.min(1, sum * Math.exp(-x + a * Math.log(x + 1e-300) - lgamma(a)));
}

function lgamma(x: number): number {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) { y++; ser += (c[j] ?? 0) / y; }
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}
