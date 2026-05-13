/**
 * Polynomial features transformer.
 * Mirrors sklearn.preprocessing.PolynomialFeatures.
 */

import { NotFittedError } from "../exceptions.js";

export class PolynomialFeatures {
  degree: number;
  interactionOnly: boolean;
  includeBias: boolean;

  nOutputFeatures_: number = 0;
  powers_: number[][] | null = null;

  constructor(
    options: {
      degree?: number;
      interactionOnly?: boolean;
      includeBias?: boolean;
    } = {},
  ) {
    this.degree = options.degree ?? 2;
    this.interactionOnly = options.interactionOnly ?? false;
    this.includeBias = options.includeBias ?? true;
  }

  private _generatePowers(nFeatures: number): number[][] {
    const includeBias = this.includeBias;
    const interactionOnly = this.interactionOnly;
    const degree = this.degree;
    const powers: number[][] = [];

    const gen = (fi: number, rem: number, cur: number[], targetDeg: number): void => {
      if (fi === nFeatures) {
        const sum = cur.reduce((a, b) => a + b, 0);
        if (sum !== targetDeg) return;
        if (!includeBias && sum === 0) return;
        if (interactionOnly && cur.some((d) => d > 1)) return;
        powers.push([...cur]);
        return;
      }
      for (let d = 0; d <= rem; d++) {
        cur.push(d);
        gen(fi + 1, rem - d, cur, targetDeg);
        cur.pop();
      }
    };

    for (let deg = 0; deg <= degree; deg++) {
      gen(0, deg, [], deg);
    }

    // Remove duplicates and sort
    const seen = new Set<string>();
    const unique: number[][] = [];
    for (const p of powers) {
      const key = p.join(",");
      if (!seen.has(key)) {
        seen.add(key);
        unique.push(p);
      }
    }

    return unique.sort((a, b) => {
      const sumA = a.reduce((s, v) => s + v, 0);
      const sumB = b.reduce((s, v) => s + v, 0);
      if (sumA !== sumB) return sumA - sumB;
      for (let i = 0; i < a.length; i++) {
        if ((a[i] ?? 0) !== (b[i] ?? 0)) return (a[i] ?? 0) - (b[i] ?? 0);
      }
      return 0;
    });
  }

  fit(X: Float64Array[]): this {
    const nFeatures = (X[0] ?? new Float64Array(0)).length;
    this.powers_ = this._generatePowers(nFeatures);
    this.nOutputFeatures_ = this.powers_.length;
    return this;
  }

  transform(X: Float64Array[]): Float64Array[] {
    if (this.powers_ === null) throw new NotFittedError("PolynomialFeatures");
    const powers = this.powers_;
    const nOut = powers.length;

    return X.map((xi) => {
      const result = new Float64Array(nOut);
      for (let k = 0; k < nOut; k++) {
        const power = powers[k] ?? [];
        let val = 1;
        for (let j = 0; j < power.length; j++) {
          const exp = power[j] ?? 0;
          if (exp !== 0) val *= (xi[j] ?? 0) ** exp;
        }
        result[k] = val;
      }
      return result;
    });
  }

  fitTransform(X: Float64Array[]): Float64Array[] {
    return this.fit(X).transform(X);
  }
}
