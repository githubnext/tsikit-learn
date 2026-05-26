/**
 * Calibration extensions: TemperatureScaling, PlattScaling, BetaCalibration.
 */

export class TemperatureScaling {
  private temperature = 1.0;

  fit(logits: Float64Array[], y: Int32Array, maxIter = 100): this {
    let T = 1.0;
    const lr = 0.01;
    for (let iter = 0; iter < maxIter; iter++) {
      let grad = 0;
      for (let i = 0; i < logits.length; i++) {
        const scaled = (logits[i]![0] ?? 0) / T;
        const p = 1 / (1 + Math.exp(-scaled));
        const yi = y[i] ?? 0;
        grad += (p - yi) * (-scaled / T);
      }
      T = Math.max(0.01, T - lr * grad / Math.max(logits.length, 1));
    }
    this.temperature = T;
    return this;
  }

  calibrate(logits: Float64Array[]): Float64Array {
    return new Float64Array(logits.map((l) => {
      const scaled = (l[0] ?? 0) / this.temperature;
      return 1 / (1 + Math.exp(-scaled));
    }));
  }

  getTemperature(): number { return this.temperature; }
}

export class PlattScaling {
  private a = 0;
  private b = 0;

  fit(scores: Float64Array, y: Int32Array, maxIter = 100): this {
    const n = scores.length;
    const hiTarget = (n + 1) / (n + 2);
    const loTarget = 1 / (n + 2);
    let a = 0;
    let b = Math.log((n + 1) / n);
    let fApB: number;
    for (let iter = 0; iter < maxIter; iter++) {
      let h11 = 0, h22 = 0, h21 = 0, g1 = 0, g2 = 0;
      for (let i = 0; i < n; i++) {
        const s = scores[i] ?? 0;
        const ti = (y[i] ?? 0) === 1 ? hiTarget : loTarget;
        fApB = s * a + b;
        let p: number, q: number;
        if (fApB >= 0) {
          p = Math.exp(-fApB) / (1 + Math.exp(-fApB));
          q = 1 / (1 + Math.exp(-fApB));
        } else {
          p = 1 / (1 + Math.exp(fApB));
          q = Math.exp(fApB) / (1 + Math.exp(fApB));
        }
        const d2 = p * q;
        h11 += s * s * d2;
        h22 += d2;
        h21 += s * d2;
        const d1 = ti - p;
        g1 += s * d1;
        g2 += d1;
      }
      const det = h11 * h22 - h21 * h21;
      if (Math.abs(det) < 1e-10) break;
      const dA = -(h22 * g1 - h21 * g2) / det;
      const dB = -(-h21 * g1 + h11 * g2) / det;
      let stepsize = 1.0;
      while (stepsize >= 1e-10) {
        const newA = a + stepsize * dA;
        const newB = b + stepsize * dB;
        let newF = 0;
        for (let i = 0; i < n; i++) {
          const s = scores[i] ?? 0;
          const ti = (y[i] ?? 0) === 1 ? hiTarget : loTarget;
          fApB = s * newA + newB;
          newF += fApB >= 0
            ? ti * fApB + Math.log(1 + Math.exp(-fApB))
            : (ti - 1) * fApB + Math.log(1 + Math.exp(fApB));
        }
        if (newF < 1e-10) { a = newA; b = newB; break; }
        stepsize /= 2;
      }
    }
    this.a = a;
    this.b = b;
    return this;
  }

  calibrate(scores: Float64Array): Float64Array {
    return new Float64Array(scores.map((s) => {
      const fApB = s * this.a + this.b;
      return fApB >= 0
        ? Math.exp(-fApB) / (1 + Math.exp(-fApB))
        : 1 / (1 + Math.exp(fApB));
    }));
  }
}

export class BetaCalibration {
  private a = 1.0;
  private b = 1.0;
  private c = 0.0;

  fit(scores: Float64Array, y: Int32Array): this {
    const eps = 1e-7;
    let sumA = 0, sumB = 0, sumC = 0;
    const n = scores.length;
    for (let i = 0; i < n; i++) {
      const s = Math.max(eps, Math.min(1 - eps, scores[i] ?? 0));
      const yi = y[i] ?? 0;
      sumA += yi * Math.log(s);
      sumB += yi * Math.log(1 - s);
      sumC += yi;
    }
    this.a = Math.max(0.01, sumA / Math.max(n, 1));
    this.b = Math.max(0.01, -sumB / Math.max(n, 1));
    this.c = sumC / Math.max(n, 1);
    return this;
  }

  calibrate(scores: Float64Array): Float64Array {
    const eps = 1e-7;
    return new Float64Array(scores.map((s) => {
      const sc = Math.max(eps, Math.min(1 - eps, s));
      const logOdds = this.a * Math.log(sc) - this.b * Math.log(1 - sc) + this.c;
      return 1 / (1 + Math.exp(-logOdds));
    }));
  }
}

export class IsotonicCalibration {
  private xs: Float64Array = new Float64Array(0);
  private ys: Float64Array = new Float64Array(0);

  fit(scores: Float64Array, y: Int32Array): this {
    const n = scores.length;
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => (scores[a] ?? 0) - (scores[b] ?? 0));
    const sortedX = new Float64Array(idx.map((i) => scores[i] ?? 0));
    const sortedY = new Float64Array(idx.map((i) => y[i] ?? 0));
    // Pool adjacent violators
    const pooled = Array.from({ length: n }, (_, i) => ({ x: sortedX[i] ?? 0, y: sortedY[i] ?? 0, cnt: 1 }));
    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < pooled.length - 1; i++) {
        const a = pooled[i];
        const b = pooled[i + 1];
        if (a !== undefined && b !== undefined && a.y > b.y) {
          const newY = (a.y * a.cnt + b.y * b.cnt) / (a.cnt + b.cnt);
          a.y = newY;
          a.cnt += b.cnt;
          pooled.splice(i + 1, 1);
          changed = true;
        }
      }
    }
    this.xs = new Float64Array(pooled.map((p) => p.x));
    this.ys = new Float64Array(pooled.map((p) => p.y));
    return this;
  }

  calibrate(scores: Float64Array): Float64Array {
    return new Float64Array(scores.map((s) => {
      if (this.xs.length === 0) return s;
      if (s <= (this.xs[0] ?? 0)) return this.ys[0] ?? 0;
      if (s >= (this.xs[this.xs.length - 1] ?? 0)) return this.ys[this.ys.length - 1] ?? 0;
      for (let i = 0; i < this.xs.length - 1; i++) {
        if (s >= (this.xs[i] ?? 0) && s <= (this.xs[i + 1] ?? 0)) {
          const dx = (this.xs[i + 1] ?? 0) - (this.xs[i] ?? 0);
          if (Math.abs(dx) < 1e-10) return this.ys[i] ?? 0;
          const t = (s - (this.xs[i] ?? 0)) / dx;
          return (this.ys[i] ?? 0) + t * ((this.ys[i + 1] ?? 0) - (this.ys[i] ?? 0));
        }
      }
      return s;
    }));
  }
}
