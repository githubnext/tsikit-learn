/**
 * Radius-based neighbors classifier and regressor.
 * Mirrors scikit-learn's neighbors.RadiusNeighborsClassifier and RadiusNeighborsRegressor.
 */

export interface RadiusNeighborsOptions {
  radius?: number;
  weights?: "uniform" | "distance";
  algorithm?: "brute";
  metric?: "euclidean" | "manhattan" | "minkowski";
  p?: number;
  outlierLabel?: number;
}

function dist(
  a: Float64Array,
  b: Float64Array,
  metric: "euclidean" | "manhattan" | "minkowski",
  p: number,
): number {
  switch (metric) {
    case "manhattan": {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
      return s;
    }
    case "minkowski": {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += Math.abs((a[i] ?? 0) - (b[i] ?? 0)) ** p;
      return s ** (1 / p);
    }
    default: {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
      return Math.sqrt(s);
    }
  }
}

export class RadiusNeighborsClassifier {
  readonly radius: number;
  readonly weights: "uniform" | "distance";
  readonly metric: "euclidean" | "manhattan" | "minkowski";
  readonly p: number;
  readonly outlierLabel: number;

  private _XFit: Float64Array[] | null = null;
  private _yFit: Int32Array | null = null;

  constructor(options: RadiusNeighborsOptions = {}) {
    this.radius = options.radius ?? 1.0;
    this.weights = options.weights ?? "uniform";
    this.metric = options.metric ?? "euclidean";
    this.p = options.p ?? 2;
    this.outlierLabel = options.outlierLabel ?? -1;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    this._XFit = X;
    this._yFit = y;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (this._XFit === null || this._yFit === null) {
      throw new Error("RadiusNeighborsClassifier must be fitted first");
    }
    const yFit = this._yFit;
    const XFit = this._XFit;
    return Int32Array.from(X, (xi) => {
      const neighbors: Array<{ label: number; d: number }> = [];
      for (let i = 0; i < XFit.length; i++) {
        const d = dist(xi, XFit[i]!, this.metric, this.p);
        if (d <= this.radius) {
          neighbors.push({ label: yFit[i]!, d });
        }
      }
      if (neighbors.length === 0) return this.outlierLabel;
      const votes = new Map<number, number>();
      for (const { label, d } of neighbors) {
        const w = this.weights === "uniform" ? 1 : (d < 1e-10 ? 1e10 : 1 / d);
        votes.set(label, (votes.get(label) ?? 0) + w);
      }
      let best = this.outlierLabel;
      let bestW = -1;
      for (const [label, w] of votes) {
        if (w > bestW) { bestW = w; best = label; }
      }
      return best;
    });
  }

  radiusNeighbors(X: Float64Array[]): Array<{ indices: Int32Array; distances: Float64Array }> {
    if (this._XFit === null) throw new Error("Not fitted");
    const XFit = this._XFit;
    return X.map((xi) => {
      const indices: number[] = [];
      const distances: number[] = [];
      for (let i = 0; i < XFit.length; i++) {
        const d = dist(xi, XFit[i]!, this.metric, this.p);
        if (d <= this.radius) {
          indices.push(i);
          distances.push(d);
        }
      }
      return { indices: Int32Array.from(indices), distances: new Float64Array(distances) };
    });
  }
}

export class RadiusNeighborsRegressor {
  readonly radius: number;
  readonly weights: "uniform" | "distance";
  readonly metric: "euclidean" | "manhattan" | "minkowski";
  readonly p: number;

  private _XFit: Float64Array[] | null = null;
  private _yFit: Float64Array | null = null;

  constructor(options: RadiusNeighborsOptions = {}) {
    this.radius = options.radius ?? 1.0;
    this.weights = options.weights ?? "uniform";
    this.metric = options.metric ?? "euclidean";
    this.p = options.p ?? 2;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    this._XFit = X;
    this._yFit = y;
    return this;
  }

  predict(X: Float64Array[]): Float64Array {
    if (this._XFit === null || this._yFit === null) {
      throw new Error("RadiusNeighborsRegressor must be fitted first");
    }
    const yFit = this._yFit;
    const XFit = this._XFit;
    return Float64Array.from(X, (xi) => {
      const neighbors: Array<{ val: number; d: number }> = [];
      for (let i = 0; i < XFit.length; i++) {
        const d = dist(xi, XFit[i]!, this.metric, this.p);
        if (d <= this.radius) {
          neighbors.push({ val: yFit[i]!, d });
        }
      }
      if (neighbors.length === 0) return 0;
      let sumW = 0, sumWY = 0;
      for (const { val, d } of neighbors) {
        const w = this.weights === "uniform" ? 1 : (d < 1e-10 ? 1e10 : 1 / d);
        sumW += w;
        sumWY += w * val;
      }
      return sumW < 1e-10 ? 0 : sumWY / sumW;
    });
  }
}
