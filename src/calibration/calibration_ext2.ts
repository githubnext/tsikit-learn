/**
 * TemperatureScaling and PlattScaling for calibration.
 */

export class TemperatureScaling {
  maxIter: number;
  lr: number;
  temperature_: number = 1.0;

  constructor(maxIter = 100, lr = 0.01) {
    this.maxIter = maxIter;
    this.lr = lr;
  }

  fit(logits: Float64Array[], y: Int32Array): this {
    let T = 1.0;
    const n = logits.length;
    for (let iter = 0; iter < this.maxIter; iter++) {
      let grad = 0, loss = 0;
      for (let i = 0; i < n; i++) {
        const logit = logits[i]?.[0] ?? 0;
        const scaledLogit = logit / T;
        const prob = 1 / (1 + Math.exp(-scaledLogit));
        const label = y[i] ?? 0;
        loss += -(label * Math.log(Math.max(prob, 1e-15)) + (1 - label) * Math.log(Math.max(1 - prob, 1e-15)));
        grad += -(label - prob) * logit / (T * T);
      }
      void loss;
      T = T - this.lr * grad / n;
      T = Math.max(0.05, T);
    }
    this.temperature_ = T;
    return this;
  }

  predictProba(logits: Float64Array[]): Float64Array[] {
    const T = this.temperature_;
    return logits.map((row) => {
      const k = row.length;
      const scaled = row.map((v) => v / T);
      const maxV = Math.max(...Array.from(scaled));
      const exps = scaled.map((v) => Math.exp(v - maxV));
      const sumExps = exps.reduce((s, v) => s + v, 0);
      return exps.map((v) => v / sumExps);
    });
  }
}

export class PlattScaling {
  maxIter: number;
  tol: number;
  a_: number = 0;
  b_: number = 0;

  constructor(maxIter = 100, tol = 1e-7) {
    this.maxIter = maxIter;
    this.tol = tol;
  }

  fit(decisionValues: Float64Array, y: Int32Array): this {
    const n = decisionValues.length;
    const nPos = Array.from(y).filter((v) => v === 1).length;
    const nNeg = n - nPos;
    const tPos = (nPos + 1) / (nPos + 2);
    const tNeg = 1 / (nNeg + 2);
    const t = Float64Array.from(y, (yi) => yi === 1 ? tPos : tNeg);

    let A = 0, B = Math.log((nNeg + 1) / (nPos + 1));
    const fVal = (fApB: number, ti: number) => ti > 0 ? ti * Math.log(Math.max(1e-15, 1 / (1 + Math.exp(fApB)))) : (ti - 1) * fApB - Math.log(1 + Math.exp(-fApB));

    for (let iter = 0; iter < this.maxIter; iter++) {
      let h11 = 0, h22 = 0, h21 = 0, g1 = 0, g2 = 0;
      for (let i = 0; i < n; i++) {
        const fApB = (decisionValues[i] ?? 0) * A + B;
        const p = Math.max(1e-15, 1 / (1 + Math.exp(-fApB)));
        const q = Math.max(1e-15, 1 - p);
        const d2 = p * q;
        h11 += (decisionValues[i] ?? 0) ** 2 * d2;
        h22 += d2;
        h21 += (decisionValues[i] ?? 0) * d2;
        const d1 = (t[i] ?? 0) - p;
        g1 -= (decisionValues[i] ?? 0) * d1;
        g2 -= d1;
      }
      if (Math.abs(g1) < this.tol && Math.abs(g2) < this.tol) break;
      const det = h11 * h22 - h21 ** 2;
      if (Math.abs(det) < 1e-12) break;
      const dA = -(h22 * g1 - h21 * g2) / det;
      const dB = -(-h21 * g1 + h11 * g2) / det;
      let stepSize = 1;
      const fOld = Array.from({ length: n }, (_, i) => fVal((decisionValues[i] ?? 0) * A + B, t[i] ?? 0)).reduce((s, v) => s + v, 0);
      for (let ls = 0; ls < 10; ls++) {
        const fNew = Array.from({ length: n }, (_, i) => fVal((decisionValues[i] ?? 0) * (A + stepSize * dA) + (B + stepSize * dB), t[i] ?? 0)).reduce((s, v) => s + v, 0);
        if (fNew < fOld) break;
        stepSize /= 2;
      }
      A += stepSize * dA;
      B += stepSize * dB;
    }
    this.a_ = A;
    this.b_ = B;
    return this;
  }

  predictProba(decisionValues: Float64Array): Float64Array {
    return Float64Array.from(decisionValues, (v) => 1 / (1 + Math.exp(v * this.a_ + this.b_)));
  }
}

export class IsotonicCalibration {
  private _xThresholds: Float64Array | null = null;
  private _yMappings: Float64Array | null = null;

  fit(scores: Float64Array, y: Int32Array): this {
    const n = scores.length;
    const sorted = Array.from({ length: n }, (_, i) => ({ s: scores[i] ?? 0, y: y[i] ?? 0 })).sort((a, b) => a.s - b.s);

    // Pool Adjacent Violators Algorithm
    const blocks: Array<{ sum: number; count: number }> = [];
    for (const { y: yi } of sorted) {
      blocks.push({ sum: yi, count: 1 });
      while (blocks.length > 1) {
        const last = blocks[blocks.length - 1] as { sum: number; count: number };
        const prev = blocks[blocks.length - 2] as { sum: number; count: number };
        if (last.sum / last.count < prev.sum / prev.count) {
          prev.sum += last.sum;
          prev.count += last.count;
          blocks.pop();
        } else break;
      }
    }

    this._xThresholds = Float64Array.from(sorted, (e) => e.s);
    const isotonicY: number[] = [];
    let idx = 0;
    for (const block of blocks) {
      const mean = block.sum / block.count;
      for (let k = 0; k < block.count; k++) isotonicY.push(mean);
      idx += block.count;
    }
    void idx;
    this._yMappings = Float64Array.from(isotonicY);
    return this;
  }

  predictProba(scores: Float64Array): Float64Array {
    if (!this._xThresholds || !this._yMappings) throw new Error("Not fitted");
    return Float64Array.from(scores, (s) => {
      const xs = this._xThresholds as Float64Array;
      const ys = this._yMappings as Float64Array;
      if (s <= (xs[0] ?? 0)) return ys[0] ?? 0;
      if (s >= (xs[xs.length - 1] ?? 0)) return ys[ys.length - 1] ?? 0;
      // Linear interpolation
      let lo = 0, hi = xs.length - 1;
      while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if ((xs[mid] ?? 0) <= s) lo = mid; else hi = mid;
      }
      const t = ((s - (xs[lo] ?? 0)) / ((xs[hi] ?? 1) - (xs[lo] ?? 0)));
      return (ys[lo] ?? 0) + t * ((ys[hi] ?? 0) - (ys[lo] ?? 0));
    });
  }
}
