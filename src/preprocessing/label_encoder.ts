/**
 * LabelEncoder — encode target labels with value between 0 and n_classes-1.
 * Mirrors sklearn.preprocessing.LabelEncoder.
 */

import { BaseEstimator } from "../base.js";
import { ValueError } from "../exceptions.js";

export class LabelEncoder extends BaseEstimator {
  classes_?: Int32Array;

  fit(y: Float64Array | Int32Array): this {
    const unique = new Set<number>();
    for (const v of y) unique.add(v);
    this.classes_ = new Int32Array([...unique].sort((a, b) => a - b));
    return this;
  }

  transform(y: Float64Array | Int32Array): Int32Array {
    this._check_is_fitted(["classes_"]);
    const classes = this.classes_ as Int32Array;
    const classMap = new Map<number, number>();
    for (let i = 0; i < classes.length; i++) {
      classMap.set(classes[i] ?? 0, i);
    }
    const result = new Int32Array(y.length);
    for (let i = 0; i < y.length; i++) {
      const encoded = classMap.get(y[i] ?? 0);
      if (encoded === undefined) {
        throw new ValueError(
          `y contains previously unseen labels: ${String(y[i])}`,
        );
      }
      result[i] = encoded;
    }
    return result;
  }

  inverse_transform(y: Int32Array): Int32Array {
    this._check_is_fitted(["classes_"]);
    const classes = this.classes_ as Int32Array;
    const result = new Int32Array(y.length);
    for (let i = 0; i < y.length; i++) {
      const idx = y[i] ?? 0;
      if (idx < 0 || idx >= classes.length) {
        throw new ValueError("y contains values not in the fitted classes");
      }
      result[i] = classes[idx] ?? 0;
    }
    return result;
  }

  fit_transform(y: Float64Array | Int32Array): Int32Array {
    return this.fit(y).transform(y);
  }
}
