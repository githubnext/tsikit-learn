/**
 * Additional multioutput regression/classification utilities.
 * Port of sklearn.multioutput extensions.
 */

import { NotFittedError } from "../exceptions.js";

type Regressor = {
  fit(X: Float64Array[], y: Float64Array): Regressor;
  predict(X: Float64Array[]): Float64Array;
};

/**
 * Multi-output regressor with correlation-aware prediction using ECC.
 */
export class EnsembleRegressorChain {
  private nChains: number;
  private chains_: Array<Array<Regressor>> = [];
  private orders_: Int32Array[] = [];
  private fitted = false;

  constructor(
    private baseEstimator: new () => Regressor,
    options: { nChains?: number } = {}
  ) {
    this.nChains = options.nChains ?? 10;
  }

  fit(X: Float64Array[], Y: Float64Array[]): this {
    const nOut = Y[0]?.length ?? 1;
    this.chains_ = [];
    this.orders_ = [];

    for (let chain = 0; chain < this.nChains; chain++) {
      const order = Int32Array.from({ length: nOut }, (_, i) => i);
      for (let i = nOut - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const tmp = order[i] ?? 0; order[i] = order[j] ?? 0; order[j] = tmp;
      }
      this.orders_.push(order);

      const chainRegressors: Regressor[] = [];
      const augX = X.map(row => Float64Array.from(row));

      for (let k = 0; k < nOut; k++) {
        const targetIdx = order[k] ?? k;
        const yTarget = Float64Array.from(Y, row => row[targetIdx] ?? 0);
        const reg = new this.baseEstimator();
        reg.fit(augX, yTarget);
        chainRegressors.push(reg);

        const preds = reg.predict(augX);
        for (let i = 0; i < augX.length; i++) {
          const newRow = new Float64Array(augX[i]!.length + 1);
          newRow.set(augX[i]!);
          newRow[augX[i]!.length] = preds[i] ?? 0;
          augX[i] = newRow;
        }
      }

      this.chains_.push(chainRegressors);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Float64Array[]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("EnsembleRegressorChain not fitted");
    const nOut = this.orders_[0]?.length ?? 1;
    const sumPreds: Float64Array[] = Array.from({ length: X.length }, () => new Float64Array(nOut));

    for (let chain = 0; chain < this.nChains; chain++) {
      const order = this.orders_[chain]!;
      const chainRegs = this.chains_[chain]!;
      const augX = X.map(row => Float64Array.from(row));

      for (let k = 0; k < nOut; k++) {
        const preds = chainRegs[k]!.predict(augX);
        const targetIdx = order[k] ?? k;
        for (let i = 0; i < X.length; i++) {
          sumPreds[i]![targetIdx] = (sumPreds[i]?.[targetIdx] ?? 0) + (preds[i] ?? 0);
          const newRow = new Float64Array(augX[i]!.length + 1);
          newRow.set(augX[i]!);
          newRow[augX[i]!.length] = preds[i] ?? 0;
          augX[i] = newRow;
        }
      }
    }

    return sumPreds.map(row => Float64Array.from(row, v => v / this.nChains));
  }
}

/**
 * Multi-output regression score (average R²).
 */
export function multiOutputR2Score(
  yTrue: Float64Array[],
  yPred: Float64Array[],
): number {
  const nOut = yTrue[0]?.length ?? 0;
  let totalR2 = 0;

  for (let o = 0; o < nOut; o++) {
    const trueVals = Float64Array.from(yTrue, row => row[o] ?? 0);
    const predVals = Float64Array.from(yPred, row => row[o] ?? 0);
    const mean = trueVals.reduce((s, v) => s + v, 0) / trueVals.length;
    let ss_tot = 0; let ss_res = 0;
    for (let i = 0; i < trueVals.length; i++) {
      ss_tot += ((trueVals[i] ?? 0) - mean) ** 2;
      ss_res += ((trueVals[i] ?? 0) - (predVals[i] ?? 0)) ** 2;
    }
    totalR2 += ss_tot > 0 ? 1 - ss_res / ss_tot : 0;
  }

  return totalR2 / nOut;
}

/**
 * Multi-output classification accuracy.
 */
export function multiOutputAccuracy(
  yTrue: Int32Array[],
  yPred: Int32Array[],
): number {
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i]!;
    const p = yPred[i]!;
    let allCorrect = true;
    for (let j = 0; j < t.length; j++) {
      if ((t[j] ?? 0) !== (p[j] ?? 0)) { allCorrect = false; break; }
    }
    if (allCorrect) correct++;
  }
  return correct / yTrue.length;
}
