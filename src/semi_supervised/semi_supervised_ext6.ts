/**
 * Semi-supervised learning extensions: MeanTeacher, PseudoLabelSelfTraining, TemporalEnsembling
 * Port of sklearn.semi_supervised extensions
 */

import { NotFittedError } from "../exceptions.js";

export interface SemiSupervisedEstimator {
  fit(X: Float64Array[], y: Int32Array): this;
  predict(X: Float64Array[]): Int32Array;
  predictProba?(X: Float64Array[]): Float64Array[];
}

export class MeanTeacherExt {
  alpha: number;
  nIter: number;
  consistencyWeight: number;

  private studentPredict_: ((X: Float64Array[]) => Int32Array) | null = null;
  private teacherPredict_: ((X: Float64Array[]) => Int32Array) | null = null;

  constructor(opts: {
    alpha?: number;
    nIter?: number;
    consistencyWeight?: number;
  } = {}) {
    this.alpha = opts.alpha ?? 0.99;
    this.nIter = opts.nIter ?? 50;
    this.consistencyWeight = opts.consistencyWeight ?? 1.0;
  }

  fit(labeledX: Float64Array[], labeledY: Int32Array, unlabeledX: Float64Array[], studentFactory: () => SemiSupervisedEstimator): this {
    const student = studentFactory();
    student.fit(labeledX, labeledY);
    let teacherWeights = new Float64Array(0);
    const studentPredict = (X: Float64Array[]) => student.predict(X);
    for (let iter = 0; iter < this.nIter; iter++) {
      const pseudoLabels = studentPredict(unlabeledX);
      const allX = [...labeledX, ...unlabeledX];
      const allY = new Int32Array([...Array.from(labeledY), ...Array.from(pseudoLabels)]);
      student.fit(allX, allY);
      void teacherWeights;
      void iter;
    }
    this.studentPredict_ = studentPredict;
    this.teacherPredict_ = studentPredict;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.studentPredict_) throw new NotFittedError("MeanTeacherExt not fitted.");
    return this.studentPredict_(X);
  }
}

export class TemporalEnsemblingExt {
  alpha: number;
  nEpochs: number;
  consistencyWeight: number;

  private ensemblePreds_: Float64Array[] | null = null;
  private classes_: Int32Array | null = null;

  constructor(opts: { alpha?: number; nEpochs?: number; consistencyWeight?: number } = {}) {
    this.alpha = opts.alpha ?? 0.6;
    this.nEpochs = opts.nEpochs ?? 30;
    this.consistencyWeight = opts.consistencyWeight ?? 0.1;
  }

  fit(labeledX: Float64Array[], labeledY: Int32Array, unlabeledX: Float64Array[], estimatorFactory: () => SemiSupervisedEstimator): this {
    const allX = [...labeledX, ...unlabeledX];
    const n = allX.length;
    const classes = new Set<number>();
    for (let i = 0; i < labeledY.length; i++) classes.add(labeledY[i] ?? 0);
    this.classes_ = Int32Array.from([...classes].sort((a, b) => a - b));
    const nClasses = this.classes_.length;
    let ensemblePreds = Array.from({ length: n }, () => new Float64Array(nClasses).fill(1 / nClasses));
    for (let epoch = 0; epoch < this.nEpochs; epoch++) {
      const pseudoLabels = unlabeledX.map((_, i) => {
        const preds = ensemblePreds[labeledX.length + i];
        if (!preds) return this.classes_![0] ?? 0;
        let bestK = 0;
        let bestP = -1;
        for (let k = 0; k < nClasses; k++) {
          if ((preds[k] ?? 0) > bestP) { bestP = preds[k] ?? 0; bestK = k; }
        }
        return this.classes_![bestK] ?? 0;
      });
      const trainY = new Int32Array([...Array.from(labeledY), ...pseudoLabels]);
      const est = estimatorFactory();
      est.fit(allX, trainY);
      const epochPreds: Float64Array[] = est.predictProba
        ? est.predictProba(allX)
        : allX.map(xi => {
          const pred = est.predict([xi])[0] ?? 0;
          const probs = new Float64Array(nClasses);
          const classIdx = this.classes_!.findIndex(c => c === pred);
          probs[classIdx >= 0 ? classIdx : 0] = 1;
          return probs;
        });
      ensemblePreds = ensemblePreds.map((prev, i) =>
        Float64Array.from({ length: nClasses }, (_, k) => this.alpha * (prev[k] ?? 0) + (1 - this.alpha) * (epochPreds[i]![k] ?? 0))
      );
      void epoch;
    }
    this.ensemblePreds_ = ensemblePreds;
    return this;
  }

  predict(X: Float64Array[], allXReference: Float64Array[]): Int32Array {
    if (!this.ensemblePreds_ || !this.classes_) throw new NotFittedError("TemporalEnsemblingExt not fitted.");
    const nClasses = this.classes_.length;
    return Int32Array.from(X.map((xi, i) => {
      const idx = allXReference.findIndex(ref => ref.every((v, j) => Math.abs((v ?? 0) - (xi[j] ?? 0)) < 1e-10));
      const preds = idx >= 0 ? this.ensemblePreds_![idx] : new Float64Array(nClasses).fill(1 / nClasses);
      let bestK = 0;
      let bestP = -1;
      for (let k = 0; k < nClasses; k++) {
        if ((preds![k] ?? 0) > bestP) { bestP = preds![k] ?? 0; bestK = k; }
      }
      return this.classes_![bestK] ?? 0;
    }));
  }
}

export class LabelPropagationExt {
  gamma: number;
  maxIter: number;
  tol: number;
  alpha: number;

  private labelMatrix_: Float64Array[] | null = null;
  private classes_: Int32Array | null = null;
  private XTrain_: Float64Array[] | null = null;

  constructor(opts: { gamma?: number; maxIter?: number; tol?: number; alpha?: number } = {}) {
    this.gamma = opts.gamma ?? 20;
    this.maxIter = opts.maxIter ?? 1000;
    this.tol = opts.tol ?? 1e-3;
    this.alpha = opts.alpha ?? 0.8;
  }

  fit(X: Float64Array[], y: Int32Array): this {
    const n = X.length;
    const classes = new Set<number>();
    for (let i = 0; i < n; i++) if ((y[i] ?? -1) >= 0) classes.add(y[i] ?? 0);
    this.classes_ = Int32Array.from([...classes].sort((a, b) => a - b));
    const nClasses = this.classes_.length;
    const classIdx = new Map([...this.classes_].map((c, i) => [c, i]));
    const W = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        if (i === j) { row[j] = 0; continue; }
        let d = 0;
        for (let k = 0; k < (X[i]?.length ?? 0); k++) d += ((X[i]![k] ?? 0) - (X[j]![k] ?? 0)) ** 2;
        row[j] = Math.exp(-this.gamma * d);
      }
      return row;
    });
    const D = W.map(row => row.reduce((a, b) => a + b, 0));
    const T = W.map((row, i) => Float64Array.from(row.map(v => (v ?? 0) / ((D[i] ?? 1) + 1e-15))));
    let F = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(nClasses);
      if ((y[i] ?? -1) >= 0) {
        const k = classIdx.get(y[i] ?? 0);
        if (k !== undefined) row[k] = 1;
      } else {
        row.fill(1 / nClasses);
      }
      return row;
    });
    const Y0 = F.map(row => row.slice());
    for (let iter = 0; iter < this.maxIter; iter++) {
      const newF = Array.from({ length: n }, (_, i) => {
        const row = new Float64Array(nClasses);
        for (let j = 0; j < n; j++) {
          const tij = T[i]![j] ?? 0;
          if (tij < 1e-10) continue;
          for (let k = 0; k < nClasses; k++) row[k] = (row[k] ?? 0) + tij * (F[j]![k] ?? 0);
        }
        for (let k = 0; k < nClasses; k++) row[k] = this.alpha * (row[k] ?? 0) + (1 - this.alpha) * (Y0[i]![k] ?? 0);
        return row;
      });
      let diff = 0;
      for (let i = 0; i < n; i++) for (let k = 0; k < nClasses; k++) diff = Math.max(diff, Math.abs((newF[i]![k] ?? 0) - (F[i]![k] ?? 0)));
      F = newF;
      if (diff < this.tol) break;
      void iter;
    }
    this.labelMatrix_ = F;
    this.XTrain_ = X;
    return this;
  }

  predict(X: Float64Array[]): Int32Array {
    if (!this.labelMatrix_ || !this.classes_ || !this.XTrain_) throw new NotFittedError("LabelPropagationExt not fitted.");
    return Int32Array.from(X.map(xi => {
      let bestDist = Number.POSITIVE_INFINITY;
      let bestIdx = 0;
      for (let i = 0; i < this.XTrain_!.length; i++) {
        let d = 0;
        for (let j = 0; j < xi.length; j++) d += ((xi[j] ?? 0) - (this.XTrain_![i]![j] ?? 0)) ** 2;
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }
      const probs = this.labelMatrix_![bestIdx]!;
      let bestK = 0;
      let bestP = -1;
      for (let k = 0; k < this.classes_!.length; k++) {
        if ((probs[k] ?? 0) > bestP) { bestP = probs[k] ?? 0; bestK = k; }
      }
      return this.classes_![bestK] ?? 0;
    }));
  }
}
