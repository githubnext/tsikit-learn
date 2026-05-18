/**
 * LogisticRegressionCV: cross-validated logistic regression.
 * Mirrors sklearn.linear_model.LogisticRegressionCV.
 */

import { NotFittedError } from "../exceptions.js";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function fitLogisticL2(
  X: Float64Array[],
  y: Int32Array,
  C: number,
  maxIter: number,
  tol: number,
): { coef: Float64Array; intercept: number } {
  const n = X.length;
  const p = (X[0] ?? new Float64Array(0)).length;
  const coef = new Float64Array(p);
  let intercept = 0;

  // Convert labels to {0, 1}
  const classes = Array.from(new Set(Array.from(y))).sort();
  const yBin = new Float64Array(n).map((_, i) =>
    (y[i] ?? 0) === (classes[1] ?? 1) ? 1 : 0,
  );

  const lr = 0.1;
  for (let iter = 0; iter < maxIter; iter++) {
    const gradCoef = new Float64Array(p);
    let gradBias = 0;
    let loss = 0;

    for (let i = 0; i < n; i++) {
      let dot = intercept;
      for (let j = 0; j < p; j++) dot += (coef[j] ?? 0) * (X[i]![j] ?? 0);
      const prob = sigmoid(dot);
      const err = prob - (yBin[i] ?? 0);
      gradBias += err;
      for (let j = 0; j < p; j++) gradCoef[j]! += err * (X[i]![j] ?? 0);
      loss += -((yBin[i] ?? 0) * Math.log(prob + 1e-15) + (1 - (yBin[i] ?? 0)) * Math.log(1 - prob + 1e-15));
    }

    // L2 regularization gradient
    let regLoss = 0;
    for (let j = 0; j < p; j++) {
      regLoss += (coef[j] ?? 0) ** 2;
      gradCoef[j]! += (coef[j] ?? 0) / C;
    }
    loss = loss / n + regLoss / (2 * C);

    const maxGrad = Math.max(
      Math.abs(gradBias) / n,
      Math.max(...Array.from(gradCoef).map(g => Math.abs(g / n))),
    );
    if (maxGrad < tol) break;

    intercept -= lr * gradBias / n;
    for (let j = 0; j < p; j++) coef[j]! -= lr * (gradCoef[j]! / n);
    void loss;
  }
  return { coef, intercept };
}

export interface LogisticRegressionCVOptions {
  Cs?: number | number[];
  cv?: number;
  penalty?: "l2";
  scoring?: "accuracy" | "neg_log_loss";
  fitIntercept?: boolean;
  maxIter?: number;
  tol?: number;
  refit?: boolean;
}

/**
 * Logistic Regression CV — selects best regularization strength via cross-validation.
 * Mirrors sklearn.linear_model.LogisticRegressionCV.
 */
export class LogisticRegressionCV {
  Cs: number[];
  cv: number;
  penalty: "l2";
  scoring: "accuracy" | "neg_log_loss";
  fitIntercept: boolean;
  maxIter: number;
  tol: number;
  refit: boolean;

  coef_: Float64Array | null = null;
  intercept_: number = 0;
  classes_: Int32Array | null = null;
  C_: number | null = null;
  scores_: Map<number, number[]> | null = null;
  Cs_: Float64Array | null = null;

  constructor(options: LogisticRegressionCVOptions = {}) {
    const rawCs = options.Cs ?? 10;
    this.Cs =
      typeof rawCs === "number"
        ? Array.from({ length: rawCs }, (_, i) =>
            Math.pow(10, -4 + (8 / (rawCs - 1)) * i),
          )
        : rawCs;
    this.cv = options.cv ?? 5;
    this.penalty = options.penalty ?? "l2";
    this.scoring = options.scoring ?? "accuracy";
    this.fitIntercept = options.fitIntercept ?? true;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-4;
    this.refit = options.refit ?? true;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const classes = Int32Array.from(new Set(Array.from(y))).sort();
    this.classes_ = classes;

    const scoresByC = new Map<number, number[]>();

    // K-fold cross-validation for each C
    const foldSize = Math.floor(n / this.cv);
    for (const C of this.Cs) {
      const foldScores: number[] = [];
      for (let fold = 0; fold < this.cv; fold++) {
        const valStart = fold * foldSize;
        const valEnd = fold === this.cv - 1 ? n : valStart + foldSize;

        const XTrain: Float64Array[] = [];
        const yTrain: number[] = [];
        const XVal: Float64Array[] = [];
        const yVal: number[] = [];

        for (let i = 0; i < n; i++) {
          if (i >= valStart && i < valEnd) {
            XVal.push(X[i]!);
            yVal.push(y[i] ?? 0);
          } else {
            XTrain.push(X[i]!);
            yTrain.push(y[i] ?? 0);
          }
        }

        const { coef, intercept } = fitLogisticL2(
          XTrain,
          new Int32Array(yTrain),
          C,
          this.maxIter,
          this.tol,
        );

        // Score on validation fold
        let score = 0;
        for (let i = 0; i < XVal.length; i++) {
          let dot = intercept;
          for (let j = 0; j < coef.length; j++)
            dot += (coef[j] ?? 0) * (XVal[i]![j] ?? 0);
          const prob = sigmoid(dot);
          const pred = prob >= 0.5 ? (classes[1] ?? 1) : (classes[0] ?? 0);
          if (this.scoring === "accuracy") {
            if (pred === yVal[i]) score++;
          } else {
            // neg_log_loss
            const p = yVal[i] === (classes[1] ?? 1) ? prob : 1 - prob;
            score -= Math.log(p + 1e-15);
          }
        }
        foldScores.push(score / (XVal.length || 1));
      }
      scoresByC.set(C, foldScores);
    }

    this.scores_ = scoresByC;

    // Select best C
    let bestC = this.Cs[0] ?? 1;
    let bestScore = -Number.POSITIVE_INFINITY;
    for (const [C, scores] of scoresByC) {
      const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
      if (mean > bestScore) {
        bestScore = mean;
        bestC = C;
      }
    }
    this.C_ = bestC;
    this.Cs_ = new Float64Array(this.Cs);

    // Refit on all data with best C
    if (this.refit) {
      const { coef, intercept } = fitLogisticL2(
        X,
        y,
        bestC,
        this.maxIter,
        this.tol,
      );
      this.coef_ = coef;
      this.intercept_ = intercept;
    }

    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.coef_ || !this.classes_) throw new NotFittedError("LogisticRegressionCV");
    const c0 = this.classes_[0] ?? 0;
    const c1 = this.classes_[1] ?? 1;
    return new Int32Array(
      X.map(xi => {
        let dot = this.intercept_;
        for (let j = 0; j < this.coef_!.length; j++)
          dot += (this.coef_![j] ?? 0) * (xi[j] ?? 0);
        return sigmoid(dot) >= 0.5 ? c1 : c0;
      }),
    );
  }

  predictProba(X: Float64Array[]): Float64Array[] {
    if (!this.coef_ || !this.classes_) throw new NotFittedError("LogisticRegressionCV");
    return X.map(xi => {
      let dot = this.intercept_;
      for (let j = 0; j < this.coef_!.length; j++)
        dot += (this.coef_![j] ?? 0) * (xi[j] ?? 0);
      const p1 = sigmoid(dot);
      return new Float64Array([1 - p1, p1]);
    });
  }

  score(X: Float64Array[], y: Int32Array): number {
    const pred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.length; i++)
      if ((pred[i] ?? 0) === (y[i] ?? 0)) correct++;
    return correct / y.length;
  }
}
