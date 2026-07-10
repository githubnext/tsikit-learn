/**
 * Pipeline extensions: CachedPipeline, TransformerPipeline, FeatureEngineeringPipeline
 * Port of sklearn.pipeline extensions
 */

import { NotFittedError } from "../exceptions.js";

export interface Transformer {
  fit(X: Float64Array[], y?: Int32Array): this;
  transform(X: Float64Array[]): Float64Array[];
  fitTransform?(X: Float64Array[], y?: Int32Array): Float64Array[];
}

export interface Estimator {
  fit(X: Float64Array[], y: Int32Array): this;
  predict(X: Float64Array[]): Int32Array;
  score?(X: Float64Array[], y: Int32Array): number;
}

export class CachedPipeline {
  steps: Array<[string, Transformer]>;
  estimator: Estimator | null;
  private cache_: Map<string, Float64Array[]> = new Map();
  private fitted_ = false;

  constructor(opts: {
    steps?: Array<[string, Transformer]>;
    estimator?: Estimator;
  } = {}) {
    this.steps = opts.steps ?? [];
    this.estimator = opts.estimator ?? null;
  }

  private _cacheKey(X: Float64Array[], stepIdx: number): string {
    const fingerprint = X.slice(0, 3).map(row => row.slice(0, 3).join(",")).join("|");
    return `step${stepIdx}:${fingerprint}`;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    let current = X;
    for (let i = 0; i < this.steps.length; i++) {
      const [, transformer] = this.steps[i]!;
      if (transformer.fitTransform) {
        current = transformer.fitTransform(current, y);
      } else {
        transformer.fit(current, y);
        current = transformer.transform(current);
      }
      const key = this._cacheKey(current, i);
      this.cache_.set(key, current.map(row => row.slice()));
    }
    if (this.estimator) this.estimator.fit(current, y);
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new NotFittedError("CachedPipeline not fitted.");
    let current = X;
    for (let i = 0; i < this.steps.length; i++) {
      const [, transformer] = this.steps[i]!;
      const key = this._cacheKey(current, i);
      if (this.cache_.has(key)) {
        current = this.cache_.get(key)!;
      } else {
        current = transformer.transform(current);
      }
    }
    return current;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.fitted_ || !this.estimator) throw new NotFittedError("CachedPipeline not fitted or has no estimator.");
    const transformed = this.transform(X);
    return this.estimator.predict(transformed);
  }

  score(X: Float64Array[], y: Int32Array): number {
    if (!this.fitted_ || !this.estimator) throw new NotFittedError("CachedPipeline not fitted or has no estimator.");
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) correct++;
    return correct / y.length;
  }

  clearCache(): void {
    this.cache_.clear();
  }
}

export class TransformerPipeline {
  steps: Array<[string, Transformer]>;
  private fitted_ = false;

  constructor(opts: { steps?: Array<[string, Transformer]> } = {}) {
    this.steps = opts.steps ?? [];
  }

  fit(X: Float64Array[], y?: Int32Array): this {
    let current = X;
    for (const [, t] of this.steps) {
      if (t.fitTransform) {
        current = t.fitTransform(current, y);
      } else {
        t.fit(current, y);
        current = t.transform(current);
      }
    }
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new NotFittedError("TransformerPipeline not fitted.");
    let current = X;
    for (const [, t] of this.steps) current = t.transform(current);
    return current;
  }

  fitTransform(X: Float64Array[], y?: Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

export class FeatureUnionExt3 {
  }

  fit(X: Float64Array[], y?: Int32Array): this {
    for (const [, t] of this.transformerList) {
      if (t.fitTransform) t.fitTransform(X, y);
      else { t.fit(X, y); }
    }
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new NotFittedError("FeatureUnionExt3 not fitted.");
    const outputs = this.transformerList.map(([, t]) => t.transform(X));
    return X.map((_, i) => {
      const parts = outputs.map(out => out[i]!);
      const totalLen = parts.reduce((s, p) => s + p.length, 0);
      const combined = new Float64Array(totalLen);
      let offset = 0;
      for (const part of parts) {
        for (let j = 0; j < part.length; j++) combined[offset + j] = part[j] ?? 0;
        offset += part.length;
      }
      return combined;
    });
  }

  fitTransform(X: Float64Array[], y?: Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}

export class SelectiveColumnTransformer {
  transformers: Array<{ name: string; transformer: Transformer; columns: number[] }>;
  remainder: "drop" | "passthrough";
  private fitted_ = false;
  private nFeatures_ = 0;

  constructor(opts: {
    transformers?: Array<{ name: string; transformer: Transformer; columns: number[] }>;
    remainder?: "drop" | "passthrough";
  } = {}) {
    this.transformers = opts.transformers ?? [];
    this.remainder = opts.remainder ?? "drop";
  }

  fit(X: Float64Array[], y?: Int32Array): this {
    this.nFeatures_ = X[0]?.length ?? 0;
    for (const { transformer, columns } of this.transformers) {
      const subX = X.map(xi => Float64Array.from(columns.map(c => xi[c] ?? 0)));
      if (transformer.fitTransform) transformer.fitTransform(subX, y);
      else transformer.fit(subX, y);
    }
    this.fitted_ = true;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (!this.fitted_) throw new NotFittedError("SelectiveColumnTransformer not fitted.");
    const usedCols = new Set(this.transformers.flatMap(t => t.columns));
    const outputs: Float64Array[][] = this.transformers.map(({ transformer, columns }) => {
      const subX = X.map(xi => Float64Array.from(columns.map(c => xi[c] ?? 0)));
      return transformer.transform(subX);
    });
    return X.map((xi, i) => {
      const parts = outputs.map(out => out[i]!);
      const passthroughCols = this.remainder === "passthrough"
        ? Array.from({ length: this.nFeatures_ }, (_, j) => j).filter(j => !usedCols.has(j))
        : [];
      const passthrough = Float64Array.from(passthroughCols.map(j => xi[j] ?? 0));
      const totalLen = parts.reduce((s, p) => s + p.length, 0) + passthrough.length;
      const combined = new Float64Array(totalLen);
      let offset = 0;
      for (const part of parts) {
        for (let j = 0; j < part.length; j++) combined[offset + j] = part[j] ?? 0;
        offset += part.length;
      }
      for (let j = 0; j < passthrough.length; j++) combined[offset + j] = passthrough[j] ?? 0;
      return combined;
    });
  }

  fitTransform(X: Float64Array[], y?: Int32Array): Float64Array[] {
    return this.fit(X, y).transform(X);
  }
}
