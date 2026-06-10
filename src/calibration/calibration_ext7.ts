/**
 * Platt scaling, temperature scaling, and reliability diagrams.
 */

export class PlattScaling {
  private coef_!: Float64Array; // [a, b] for sigmoid(a*x + b)
  private fitted_ = false;

  fit(scores: Float64Array, yTrue: Int32Array, maxIter = 100, lr = 0.01): this {
    // Platt scaling: fit logistic regression on raw scores
    this.coef_ = new Float64Array([1.0, 0.0]);
    const n = scores.length;
    // Gradient descent
    for (let iter = 0; iter < maxIter; iter++) {
      const a = this.coef_[0] ?? 1, b = this.coef_[1] ?? 0;
      let ga = 0, gb = 0;
      for (let i = 0; i < n; i++) {
        const s = scores[i] ?? 0;
        const p = 1 / (1 + Math.exp(-(a * s + b)));
        const err = p - (yTrue[i] ?? 0);
        ga += err * s;
        gb += err;
      }
      this.coef_[0] = a - lr * ga / n;
      this.coef_[1] = b - lr * gb / n;
    }
    this.fitted_ = true;
    return this;
  }

  calibrate(scores: Float64Array): Float64Array {
    if (!this.fitted_) throw new Error('Not fitted');
    const a = this.coef_[0] ?? 1, b = this.coef_[1] ?? 0;
    return new Float64Array(scores.map(s => 1 / (1 + Math.exp(-(a * s + b)))));
  }
}

export class TemperatureScaling {
  private temperature_ = 1.0;
  private fitted_ = false;

  fit(logits: Float64Array[], yTrue: Int32Array, maxIter = 50, lr = 0.01): this {
    let T = 1.0;
    const n = logits.length;
    const nClasses = logits[0]?.length ?? 2;
    for (let iter = 0; iter < maxIter; iter++) {
      let grad = 0;
      for (let i = 0; i < n; i++) {
        const row = logits[i]!;
        const scaled = new Float64Array(row.map(v => v / T));
        const maxL = Math.max(...scaled);
        const exps = new Float64Array(scaled.map(v => Math.exp(v - maxL)));
        const sumExp = exps.reduce((s, v) => s + v, 0);
        const proba = new Float64Array(exps.map(v => v / sumExp));
        const trueClass = yTrue[i] ?? 0;
        const pTrue = proba[trueClass] ?? 0;
        // NLL gradient w.r.t. T
        const weightedLogit = Array.from(row).reduce((s, v, c) => s + (proba[c] ?? 0) * v, 0);
        grad += ((row[trueClass] ?? 0) - weightedLogit) / (-T * T);
      }
      T = T - lr * grad / n;
      if (T <= 0) T = 0.01;
    }
    this.temperature_ = T;
    this.fitted_ = true;
    void nClasses;
    return this;
  }

  calibrate(logits: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new Error('Not fitted');
    const T = this.temperature_;
    return logits.map(row => {
      const scaled = new Float64Array(row.map(v => v / T));
      const maxL = Math.max(...scaled);
      const exps = new Float64Array(scaled.map(v => Math.exp(v - maxL)));
      const sumExp = exps.reduce((s, v) => s + v, 0);
      return new Float64Array(exps.map(v => v / sumExp));
    });
  }

  get temperature(): number { return this.temperature_; }
}

export interface CalibrationCurve {
  fractionOfPositives: Float64Array;
  meanPredictedValue: Float64Array;
  binCounts: Int32Array;
}

export function calibrationCurve(
  yTrue: Int32Array,
  yProb: Float64Array,
  nBins = 10,
  strategy: 'uniform' | 'quantile' = 'uniform'
): CalibrationCurve {
  let bins: Float64Array;
  if (strategy === 'uniform') {
    bins = new Float64Array(nBins + 1).map((_, k) => k / nBins);
  } else {
    const sorted = Array.from(yProb).sort((a, b) => a - b);
    bins = new Float64Array(nBins + 1).map((_, k) => sorted[Math.floor(k * sorted.length / nBins)] ?? (k === nBins ? 1 : 0));
  }
  const n = yTrue.length;
  const fracPos = new Float64Array(nBins);
  const meanPred = new Float64Array(nBins);
  const counts = new Int32Array(nBins);
  for (let i = 0; i < n; i++) {
    const p = yProb[i] ?? 0;
    let b = nBins - 1;
    for (let k = 0; k < nBins; k++) {
      if (p < (bins[k + 1] ?? 1)) { b = k; break; }
    }
    counts[b] = (counts[b] ?? 0) + 1;
    fracPos[b] = (fracPos[b] ?? 0) + (yTrue[i] ?? 0);
    meanPred[b] = (meanPred[b] ?? 0) + p;
  }
  const fractionOfPositives = new Float64Array(nBins).map((_, b) => (counts[b] ?? 0) > 0 ? (fracPos[b] ?? 0) / (counts[b] ?? 1) : 0);
  const meanPredictedValue = new Float64Array(nBins).map((_, b) => (counts[b] ?? 0) > 0 ? (meanPred[b] ?? 0) / (counts[b] ?? 1) : 0);
  return { fractionOfPositives, meanPredictedValue, binCounts: counts };
}

export function expectedCalibrationError(
  yTrue: Int32Array,
  yProb: Float64Array,
  nBins = 10
): number {
  const { fractionOfPositives, meanPredictedValue, binCounts } = calibrationCurve(yTrue, yProb, nBins);
  const n = yTrue.length;
  return Array.from(binCounts).reduce((s, cnt, b) =>
    s + (cnt / n) * Math.abs((fractionOfPositives[b] ?? 0) - (meanPredictedValue[b] ?? 0)), 0
  );
}
