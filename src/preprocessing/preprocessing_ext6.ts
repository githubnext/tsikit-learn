/**
 * CyclicalEncoder and DatetimeFeatureExtractor — temporal feature preprocessing.
 */

export class CyclicalEncoder {
  periods: Record<string, number>;
  private featureNames_: string[] | null = null;
  nFeaturesIn_: number = 0;
  nFeaturesOut_: number = 0;

  constructor(periods: Record<string, number> = { hour: 24, day_of_week: 7, month: 12 }) {
    this.periods = periods;
  }

  fit(X: Float64Array[], featureNames: string[]): this {
    this.nFeaturesIn_ = X[0]?.length ?? 0;
    this.featureNames_ = featureNames;
    this.nFeaturesOut_ = this.nFeaturesIn_ * 2;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const names = this.featureNames_ ?? [];
    return X.map((row) => {
      const parts: number[] = [];
      for (let j = 0; j < row.length; j++) {
        const name = names[j] ?? "";
        const period = this.periods[name] ?? 1;
        const v = (row[j] ?? 0);
        parts.push(Math.sin(2 * Math.PI * v / period));
        parts.push(Math.cos(2 * Math.PI * v / period));
      }
      return Float64Array.from(parts);
    });
  }

  fitTransform(X: Float64Array[], featureNames: string[]): Float64Array[] {
    return this.fit(X, featureNames).transform(X);
  }

  getFeatureNames(): string[] {
    const names = this.featureNames_ ?? [];
    const out: string[] = [];
    for (const name of names) {
      out.push(`${name}_sin`);
      out.push(`${name}_cos`);
    }
    return out;
  }
}

export class DatetimeFeatureExtractor {
  features: Array<"year" | "month" | "day" | "hour" | "minute" | "second" | "day_of_week" | "day_of_year" | "week_of_year">;
  cyclicalEncode: boolean;
  nFeaturesOut_: number = 0;

  constructor(
    features: Array<"year" | "month" | "day" | "hour" | "minute" | "second" | "day_of_week" | "day_of_year" | "week_of_year"> = ["year", "month", "day", "hour", "day_of_week"],
    cyclicalEncode = false,
  ) {
    this.features = features;
    this.cyclicalEncode = cyclicalEncode;
  }

  fit(_timestamps: number[]): this {
    this.nFeaturesOut_ = this.cyclicalEncode
      ? this.features.filter((f) => f !== "year").length * 2 + (this.features.includes("year") ? 1 : 0)
      : this.features.length;
    return this;
  }

  transform(timestamps: number[]): Float64Array[] {
    return timestamps.map((ts) => {
      const d = new Date(ts);
      const parts: number[] = [];
      const dayOfYear = Math.floor((ts - new Date(d.getFullYear(), 0, 0).getTime()) / 86400000);
      const weekOfYear = Math.ceil(dayOfYear / 7);
      for (const feat of this.features) {
        const v = (() => {
          switch (feat) {
            case "year": return d.getFullYear();
            case "month": return d.getMonth() + 1;
            case "day": return d.getDate();
            case "hour": return d.getHours();
            case "minute": return d.getMinutes();
            case "second": return d.getSeconds();
            case "day_of_week": return d.getDay();
            case "day_of_year": return dayOfYear;
            case "week_of_year": return weekOfYear;
            default: return 0;
          }
        })();
        if (this.cyclicalEncode && feat !== "year") {
          const period = { month: 12, day: 31, hour: 24, minute: 60, second: 60, day_of_week: 7, day_of_year: 365, week_of_year: 52 }[feat] ?? 1;
          parts.push(Math.sin(2 * Math.PI * v / period));
          parts.push(Math.cos(2 * Math.PI * v / period));
        } else {
          parts.push(v);
        }
      }
      return Float64Array.from(parts);
    });
  }

  fitTransform(timestamps: number[]): Float64Array[] {
    return this.fit(timestamps).transform(timestamps);
  }

  getFeatureNames(): string[] {
    const out: string[] = [];
    for (const feat of this.features) {
      if (this.cyclicalEncode && feat !== "year") {
        out.push(`${feat}_sin`, `${feat}_cos`);
      } else {
        out.push(feat);
      }
    }
    return out;
  }
}

export class InteractionTransformer {
  degree: number;
  includeOriginal: boolean;
  nFeaturesIn_: number = 0;
  nFeaturesOut_: number = 0;

  constructor(degree = 2, includeOriginal = true) {
    this.degree = degree;
    this.includeOriginal = includeOriginal;
  }

  fit(X: Float64Array[]): this {
    const p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    let count = this.includeOriginal ? p : 0;
    // Count interaction terms
    for (let i = 0; i < p; i++) {
      for (let j = i; j < p; j++) {
        count++;
      }
    }
    this.nFeaturesOut_ = count;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const p = this.nFeaturesIn_ || (X[0]?.length ?? 0);
    return X.map((row) => {
      const parts: number[] = this.includeOriginal ? Array.from(row) : [];
      for (let i = 0; i < p; i++) {
        for (let j = i; j < p; j++) {
          parts.push((row[i] ?? 0) * (row[j] ?? 0));
        }
      }
      return Float64Array.from(parts);
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}

export class TargetMeanEncoder {
  smoothing: number;
  private meansByCategory_: Map<string, Map<number, number>> | null = null;
  private globalMean_: number = 0;
  nFeaturesIn_: number = 0;

  constructor(smoothing = 10) {
    this.smoothing = smoothing;
  }

  fit(X: Float64Array[], y: Float64Array): this {
    const n = X.length;
    const p = X[0]?.length ?? 0;
    this.nFeaturesIn_ = p;
    this.globalMean_ = y.reduce((a, b) => a + b, 0) / n;
    this.meansByCategory_ = new Map();

    for (let j = 0; j < p; j++) {
      const featureMap = new Map<number, { sum: number; count: number }>();
      for (let i = 0; i < n; i++) {
        const cat = Math.round(X[i]?.[j] ?? 0);
        if (!featureMap.has(cat)) featureMap.set(cat, { sum: 0, count: 0 });
        const entry = featureMap.get(cat)!;
        entry.sum += y[i] ?? 0;
        entry.count++;
      }
      const smoothed = new Map<number, number>();
      for (const [cat, { sum, count }] of featureMap) {
        smoothed.set(cat, (sum + this.smoothing * this.globalMean_) / (count + this.smoothing));
      }
      this.meansByCategory_.set(`feature_${j}`, smoothed);
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    const p = X[0]?.length ?? 0;
    return X.map((row) => {
      const out = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        const cat = Math.round(row[j] ?? 0);
        const featureKey = `feature_${j}`;
        out[j] = this.meansByCategory_?.get(featureKey)?.get(cat) ?? this.globalMean_;
      }
      return out;
    });
  }

  fitTransform(X: Float64Array[], y: Float64Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}
