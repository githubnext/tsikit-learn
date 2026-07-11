/**
 * ColumnTransformer: applies transformers to columns of an array.
 * Mirrors sklearn.compose.ColumnTransformer.
 */

import { NotFittedError } from "../exceptions.js";

export interface Transformer {
  fit(X: Float64Array[]): this;
  transform(X: Float64Array[]): Float64Array[];
  fitTransform?(X: Float64Array[]): Float64Array[];
}

export type ColumnSpec = number | number[] | "all";

export class ColumnTransformer {
  transformers: [string, Transformer | "passthrough" | "drop", ColumnSpec][];
  remainder: "passthrough" | "drop";

  transformers_: [string, Transformer | "passthrough", ColumnSpec][] = [];
  private _nFeatures = 0;
  private _allCols = new Set<number>();

  constructor(
    transformers: [string, Transformer | "passthrough" | "drop", ColumnSpec][],
    options: { remainder?: "passthrough" | "drop" } = {},
  ) {
    this.transformers = transformers;
    this.remainder = options.remainder ?? "drop";
  }

  private _getCols(spec: ColumnSpec, nFeatures: number): number[] {
    if (spec === "all") return Array.from({ length: nFeatures }, (_, i) => i);
    if (typeof spec === "number") return [spec];
    return spec;
  }

  fit(X: Float64Array[]): this {
    const n = (X[0] ?? new Float64Array(0)).length;
    this._nFeatures = n;
    this._allCols.clear();

    this.transformers_ = [];
    for (const [name, t, spec] of this.transformers) {
      if (t === "drop") continue;
      const cols = this._getCols(spec, n);
      for (const c of cols) this._allCols.add(c);

      if (t === "passthrough") {
        this.transformers_.push([name, "passthrough", spec]);
      } else {
        const Xsub = X.map(
          (row) => new Float64Array(cols.map((c) => row[c] ?? 0)),
        );
        t.fit(Xsub);
        this.transformers_.push([name, t, spec]);
      }
    }
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.transformers_.length === 0)
      throw new NotFittedError("ColumnTransformer");
    const n = (X[0] ?? new Float64Array(0)).length;
    const parts: Float64Array[][] = [];

    for (const [, t, spec] of this.transformers_) {
      const cols = this._getCols(spec, n);
      const Xsub = X.map(
        (row) => new Float64Array(cols.map((c) => row[c] ?? 0)),
      );
      if (t === "passthrough") {
        parts.push(Xsub);
      } else {
        parts.push(t.transform(Xsub));
      }
    }

    if (this.remainder === "passthrough") {
      const remainderCols: number[] = [];
      for (let c = 0; c < n; c++) {
        if (!this._allCols.has(c)) remainderCols.push(c);
      }
      if (remainderCols.length > 0) {
        parts.push(
          X.map(
            (row) => new Float64Array(remainderCols.map((c) => row[c] ?? 0)),
          ),
        );
      }
    }

    // Horizontally concatenate
    return X.map((_, i) => {
      const rowParts = parts.map((p) => p[i] ?? new Float64Array(0));
      const total = rowParts.reduce((s, r) => s + r.length, 0);
      const result = new Float64Array(total);
      let offset = 0;
      for (const part of rowParts) {
        result.set(part, offset);
        offset += part.length;
      }
      return result;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
