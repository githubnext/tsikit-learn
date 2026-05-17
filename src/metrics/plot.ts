/**
 * Display classes for metrics visualization.
 * Mirrors sklearn.metrics.ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
 * DetCurveDisplay, CalibrationDisplay.
 */

export interface ConfusionMatrixDisplayOptions {
  confusionMatrix: number[][];
  displayLabels?: string[];
  colormap?: string;
  includeValues?: boolean;
  valuesFormat?: string;
  textKw?: Record<string, unknown>;
  imKw?: Record<string, unknown>;
  axisKw?: Record<string, unknown>;
}

/**
 * Visualization of a confusion matrix.
 */
export class ConfusionMatrixDisplay {
  confusionMatrix: number[][];
  displayLabels: string[];

  constructor(options: ConfusionMatrixDisplayOptions) {
    this.confusionMatrix = options.confusionMatrix;
    this.displayLabels =
      options.displayLabels ??
      options.confusionMatrix.map((_, i) => String(i));
  }

  /**
   * Compute from estimator predictions.
   */
  static fromEstimator(
    estimator: { predict(X: Float64Array[]): Int32Array },
    X: Float64Array[],
    y: Int32Array,
    labels?: number[]
  ): ConfusionMatrixDisplay {
    const yPred = estimator.predict(X);
    return ConfusionMatrixDisplay.fromPredictions(y, yPred, labels);
  }

  /**
   * Compute from true and predicted labels.
   */
  static fromPredictions(
    yTrue: Int32Array,
    yPred: Int32Array,
    labels?: number[]
  ): ConfusionMatrixDisplay {
    const uniqueLabels =
      labels ??
      [...new Set([...yTrue, ...yPred])].sort((a, b) => a - b);
    const n = uniqueLabels.length;
    const labelIdx = new Map(uniqueLabels.map((l, i) => [l, i]));
    const cm = Array.from({ length: n }, () => new Array(n).fill(0) as number[]);
    for (let i = 0; i < yTrue.length; i++) {
      const ti = labelIdx.get(yTrue[i] ?? 0) ?? 0;
      const pi = labelIdx.get(yPred[i] ?? 0) ?? 0;
      cm[ti]![pi]! += 1;
    }
    return new ConfusionMatrixDisplay({
      confusionMatrix: cm,
      displayLabels: uniqueLabels.map(String),
    });
  }

  /**
   * Return ASCII text representation of the confusion matrix.
   */
  toText(): string {
    const n = this.confusionMatrix.length;
    const maxLen = Math.max(
      ...this.displayLabels.map((l) => l.length),
      ...this.confusionMatrix.flat().map((v) => String(v).length)
    );
    const pad = (s: string, w: number) => s.padStart(w);
    const header = " ".repeat(maxLen + 2) +
      this.displayLabels.map((l) => pad(l, maxLen + 1)).join("");
    const rows = this.confusionMatrix.map(
      (row, i) =>
        pad(this.displayLabels[i] ?? String(i), maxLen) + " |" +
        row.map((v) => pad(String(v), maxLen + 1)).join("")
    );
    return [header, ...rows].join("\n");
  }
}

export interface RocCurveDisplayOptions {
  fpr: Float64Array;
  tpr: Float64Array;
  rocAuc?: number;
  estimatorName?: string;
  pos_label?: number;
}

/**
 * ROC Curve visualization.
 */
export class RocCurveDisplay {
  fpr: Float64Array;
  tpr: Float64Array;
  rocAuc: number;
  estimatorName: string;

  constructor(options: RocCurveDisplayOptions) {
    this.fpr = options.fpr;
    this.tpr = options.tpr;
    this.rocAuc = options.rocAuc ?? Number.NaN;
    this.estimatorName = options.estimatorName ?? "";
  }

  /**
   * Compute from estimator predict_proba.
   */
  static fromEstimator(
    estimator: { predictProba(X: Float64Array[]): Float64Array[] },
    X: Float64Array[],
    y: Int32Array,
    posLabel = 1
  ): RocCurveDisplay {
    const probas = estimator.predictProba(X);
    const scores = new Float64Array(probas.map((p) => p[posLabel] ?? 0));
    return RocCurveDisplay.fromPredictions(y, scores, posLabel);
  }

  /**
   * Compute from predictions.
   */
  static fromPredictions(
    yTrue: Int32Array,
    yScore: Float64Array,
    posLabel = 1
  ): RocCurveDisplay {
    const { fpr, tpr, auc } = computeRocCurve(yTrue, yScore, posLabel);
    return new RocCurveDisplay({ fpr, tpr, rocAuc: auc });
  }

  /** SVG representation */
  toSvg(width = 300, height = 300): string {
    const m = 40;
    const w = width - 2 * m;
    const h = height - 2 * m;
    const pts = Array.from(this.fpr).map((x, i) => {
      const px = m + x * w;
      const py = m + (1 - (this.tpr[i] ?? 0)) * h;
      return `${px},${py}`;
    });
    const line = `<polyline points="${pts.join(" ")}" fill="none" stroke="steelblue" stroke-width="2"/>`;
    const diag = `<line x1="${m}" y1="${m + h}" x2="${m + w}" y2="${m}" stroke="gray" stroke-dasharray="4"/>`;
    const label = `<text x="${width / 2}" y="${height - 5}" text-anchor="middle" font-size="12">AUC = ${this.rocAuc.toFixed(3)}</text>`;
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${diag}${line}${label}</svg>`;
  }
}

function computeRocCurve(
  yTrue: Int32Array,
  yScore: Float64Array,
  posLabel: number
): { fpr: Float64Array; tpr: Float64Array; auc: number } {
  const n = yTrue.length;
  const indices = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0)
  );
  const nPos = Array.from(yTrue).filter((v) => v === posLabel).length;
  const nNeg = n - nPos;
  let tp = 0, fp = 0;
  const tpr: number[] = [0];
  const fpr: number[] = [0];
  let prevScore = Number.POSITIVE_INFINITY;
  for (const idx of indices) {
    const s = yScore[idx] ?? 0;
    if (s !== prevScore && prevScore !== Number.POSITIVE_INFINITY) {
      tpr.push(tp / Math.max(nPos, 1));
      fpr.push(fp / Math.max(nNeg, 1));
    }
    if ((yTrue[idx] ?? 0) === posLabel) tp++;
    else fp++;
    prevScore = s;
  }
  tpr.push(tp / Math.max(nPos, 1));
  fpr.push(fp / Math.max(nNeg, 1));

  // Compute AUC via trapezoidal rule
  let auc = 0;
  for (let i = 1; i < fpr.length; i++) {
    auc += ((fpr[i]! - fpr[i - 1]!) * ((tpr[i]! + tpr[i - 1]!) / 2));
  }

  return { fpr: new Float64Array(fpr), tpr: new Float64Array(tpr), auc };
}

export interface PrecisionRecallDisplayOptions {
  precision: Float64Array;
  recall: Float64Array;
  averagePrecision?: number;
  estimatorName?: string;
  posLabel?: number;
}

/**
 * Precision-Recall Curve visualization.
 */
export class PrecisionRecallDisplay {
  precision: Float64Array;
  recall: Float64Array;
  averagePrecision: number;
  estimatorName: string;

  constructor(options: PrecisionRecallDisplayOptions) {
    this.precision = options.precision;
    this.recall = options.recall;
    this.averagePrecision = options.averagePrecision ?? Number.NaN;
    this.estimatorName = options.estimatorName ?? "";
  }

  static fromPredictions(
    yTrue: Int32Array,
    probaPos: Float64Array,
    posLabel = 1
  ): PrecisionRecallDisplay {
    const n = yTrue.length;
    const indices = Array.from({ length: n }, (_, i) => i).sort(
      (a, b) => (probaPos[b] ?? 0) - (probaPos[a] ?? 0)
    );
    const nPos = Array.from(yTrue).filter((v) => v === posLabel).length;
    let tp = 0, fp = 0;
    const prec: number[] = [];
    const rec: number[] = [];
    for (const idx of indices) {
      if ((yTrue[idx] ?? 0) === posLabel) tp++;
      else fp++;
      prec.push(tp / (tp + fp));
      rec.push(tp / Math.max(nPos, 1));
    }
    prec.push(1);
    rec.push(0);

    // Average precision
    let ap = 0;
    for (let i = 1; i < rec.length; i++) {
      ap += (rec[i - 1]! - rec[i]!) * prec[i - 1]!;
    }

    return new PrecisionRecallDisplay({
      precision: new Float64Array(prec),
      recall: new Float64Array(rec),
      averagePrecision: ap,
    });
  }

  toSvg(width = 300, height = 300): string {
    const m = 40;
    const w = width - 2 * m;
    const h = height - 2 * m;
    const pts = Array.from(this.recall).map((r, i) => {
      const px = m + r * w;
      const py = m + (1 - (this.precision[i] ?? 0)) * h;
      return `${px},${py}`;
    });
    const line = `<polyline points="${pts.join(" ")}" fill="none" stroke="darkorange" stroke-width="2"/>`;
    const label = `<text x="${width / 2}" y="${height - 5}" text-anchor="middle" font-size="12">AP = ${this.averagePrecision.toFixed(3)}</text>`;
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${line}${label}</svg>`;
  }
}

export interface DetCurveDisplayOptions {
  fpr: Float64Array;
  fnr: Float64Array;
  estimatorName?: string;
}

/**
 * Detection Error Tradeoff (DET) curve visualization.
 */
export class DetCurveDisplay {
  fpr: Float64Array;
  fnr: Float64Array;
  estimatorName: string;

  constructor(options: DetCurveDisplayOptions) {
    this.fpr = options.fpr;
    this.fnr = options.fnr;
    this.estimatorName = options.estimatorName ?? "";
  }

  static fromPredictions(
    yTrue: Int32Array,
    yScore: Float64Array,
    posLabel = 1
  ): DetCurveDisplay {
    const { fpr, tpr } = computeRocCurve(yTrue, yScore, posLabel);
    const fnr = new Float64Array(tpr.map((t) => 1 - t));
    return new DetCurveDisplay({ fpr, fnr });
  }

  toSvg(width = 300, height = 300): string {
    const m = 40;
    const w = width - 2 * m;
    const h = height - 2 * m;
    const pts = Array.from(this.fpr).map((x, i) => {
      const px = m + x * w;
      const py = m + (this.fnr[i] ?? 0) * h;
      return `${px},${py}`;
    });
    const line = `<polyline points="${pts.join(" ")}" fill="none" stroke="crimson" stroke-width="2"/>`;
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${line}</svg>`;
  }
}

export interface CalibrationDisplayOptions {
  fractionOfPositives: Float64Array;
  meanPredictedValue: Float64Array;
  estimatorName?: string;
  nBins?: number;
}

/**
 * Calibration curve visualization.
 */
export class CalibrationDisplay {
  fractionOfPositives: Float64Array;
  meanPredictedValue: Float64Array;
  estimatorName: string;
  nBins: number;

  constructor(options: CalibrationDisplayOptions) {
    this.fractionOfPositives = options.fractionOfPositives;
    this.meanPredictedValue = options.meanPredictedValue;
    this.estimatorName = options.estimatorName ?? "";
    this.nBins = options.nBins ?? 5;
  }

  static fromEstimator(
    estimator: {
      predictProba(X: Float64Array[]): Float64Array[];
      fitted_?: boolean;
    },
    X: Float64Array[],
    y: Int32Array,
    nBins = 5,
    posLabel = 1
  ): CalibrationDisplay {
    const probas = estimator.predictProba(X);
    const scores = new Float64Array(probas.map((p) => p[posLabel] ?? 0));
    return CalibrationDisplay.fromPredictions(y, scores, nBins, posLabel);
  }

  static fromPredictions(
    yTrue: Int32Array,
    yProba: Float64Array,
    nBins = 5,
    posLabel = 1
  ): CalibrationDisplay {
    const binEdges = Array.from({ length: nBins + 1 }, (_, i) => i / nBins);
    const fracPos = new Float64Array(nBins);
    const meanPred = new Float64Array(nBins);
    const binCounts = new Int32Array(nBins);

    for (let i = 0; i < yTrue.length; i++) {
      const p = yProba[i] ?? 0;
      const binIdx = Math.min(
        Math.floor(p * nBins),
        nBins - 1
      );
      fracPos[binIdx]! += (yTrue[i] ?? 0) === posLabel ? 1 : 0;
      meanPred[binIdx]! += p;
      binCounts[binIdx]! += 1;
    }

    for (let b = 0; b < nBins; b++) {
      const cnt = binCounts[b] ?? 1;
      if (cnt > 0) {
        fracPos[b]! /= cnt;
        meanPred[b]! /= cnt;
      } else {
        fracPos[b] = (binEdges[b]! + binEdges[b + 1]!) / 2;
        meanPred[b] = (binEdges[b]! + binEdges[b + 1]!) / 2;
      }
    }

    return new CalibrationDisplay({ fractionOfPositives: fracPos, meanPredictedValue: meanPred, nBins });
  }

  toSvg(width = 300, height = 300): string {
    const m = 40;
    const w = width - 2 * m;
    const h = height - 2 * m;
    const diag = `<line x1="${m}" y1="${m + h}" x2="${m + w}" y2="${m}" stroke="gray" stroke-dasharray="4"/>`;
    const pts = Array.from(this.meanPredictedValue).map((x, i) => {
      const px = m + x * w;
      const py = m + (1 - (this.fractionOfPositives[i] ?? 0)) * h;
      return `${px},${py}`;
    });
    const line = `<polyline points="${pts.join(" ")}" fill="none" stroke="steelblue" stroke-width="2" marker-end="url(#dot)"/>`;
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${diag}${line}</svg>`;
  }
}
