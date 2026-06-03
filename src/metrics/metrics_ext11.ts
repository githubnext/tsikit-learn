/**
 * Metrics extensions: calibration metrics, distance metrics, fairness metrics.
 * Mirrors sklearn.metrics additional methods.
 */

/** Expected calibration error (ECE) for probability calibration assessment. */
export function expectedCalibrationError(
  y_true: Int32Array,
  y_prob: Float64Array,
  n_bins = 10,
): number {
  const n = y_true.length;
  const binEdges = Array.from({ length: n_bins + 1 }, (_, i) => i / n_bins);
  let ece = 0;

  for (let b = 0; b < n_bins; b++) {
    const lo = binEdges[b] ?? 0;
    const hi = binEdges[b + 1] ?? 1;
    const idx: number[] = [];
    for (let i = 0; i < n; i++) {
      const p = y_prob[i] ?? 0;
      if (p >= lo && (b === n_bins - 1 ? p <= hi : p < hi)) idx.push(i);
    }
    if (idx.length === 0) continue;
    const accMean = idx.filter(i => (y_true[i] ?? 0) === 1).length / idx.length;
    const confMean = idx.reduce((s, i) => s + (y_prob[i] ?? 0), 0) / idx.length;
    ece += (idx.length / n) * Math.abs(accMean - confMean);
  }
  return ece;
}

/** Maximum calibration error (MCE). */
export function maxCalibrationError(
  y_true: Int32Array,
  y_prob: Float64Array,
  n_bins = 10,
): number {
  const n = y_true.length;
  const binEdges = Array.from({ length: n_bins + 1 }, (_, i) => i / n_bins);
  let mce = 0;

  for (let b = 0; b < n_bins; b++) {
    const lo = binEdges[b] ?? 0;
    const hi = binEdges[b + 1] ?? 1;
    const idx: number[] = [];
    for (let i = 0; i < n; i++) {
      const p = y_prob[i] ?? 0;
      if (p >= lo && (b === n_bins - 1 ? p <= hi : p < hi)) idx.push(i);
    }
    if (idx.length === 0) continue;
    const accMean = idx.filter(i => (y_true[i] ?? 0) === 1).length / idx.length;
    const confMean = idx.reduce((s, i) => s + (y_prob[i] ?? 0), 0) / idx.length;
    mce = Math.max(mce, Math.abs(accMean - confMean));
  }
  return mce;
}

/** Demographic parity difference (fairness metric). */
export function demographicParityDifference(
  y_pred: Int32Array,
  sensitive: Int32Array,
): number {
  const groups = [...new Set(Array.from(sensitive))];
  const rates = groups.map(g => {
    const idx = Array.from({ length: y_pred.length }, (_, i) => i).filter(i => sensitive[i] === g);
    return idx.filter(i => (y_pred[i] ?? 0) === 1).length / (idx.length || 1);
  });
  return Math.max(...rates) - Math.min(...rates);
}

/** Equalized odds difference (fairness metric). */
export function equalizedOddsDifference(
  y_true: Int32Array,
  y_pred: Int32Array,
  sensitive: Int32Array,
): number {
  const groups = [...new Set(Array.from(sensitive))];
  const tprByGroup = groups.map(g => {
    const idx = Array.from({ length: y_true.length }, (_, i) => i)
      .filter(i => sensitive[i] === g && y_true[i] === 1);
    return idx.filter(i => (y_pred[i] ?? 0) === 1).length / (idx.length || 1);
  });
  const fprByGroup = groups.map(g => {
    const idx = Array.from({ length: y_true.length }, (_, i) => i)
      .filter(i => sensitive[i] === g && y_true[i] === 0);
    return idx.filter(i => (y_pred[i] ?? 0) === 1).length / (idx.length || 1);
  });
  const tprDiff = Math.max(...tprByGroup) - Math.min(...tprByGroup);
  const fprDiff = Math.max(...fprByGroup) - Math.min(...fprByGroup);
  return Math.max(tprDiff, fprDiff);
}

/** Balanced accuracy score accounting for class imbalance. */
export function balancedAccuracyScore(y_true: Int32Array, y_pred: Int32Array): number {
  const classes = [...new Set(Array.from(y_true))];
  let sum = 0;
  for (const cls of classes) {
    const truePositives = Array.from(y_true).filter((v, i) => v === cls && (y_pred[i] ?? -1) === cls).length;
    const total = Array.from(y_true).filter(v => v === cls).length;
    sum += truePositives / (total || 1);
  }
  return sum / (classes.length || 1);
}

/** Compute Jaccard similarity coefficient. */
export function jaccardScore(
  y_true: Int32Array,
  y_pred: Int32Array,
  average: "macro" | "micro" = "macro",
): number {
  const classes = [...new Set([...Array.from(y_true), ...Array.from(y_pred)])];
  if (average === "micro") {
    let tp = 0, fp = 0, fn = 0;
    for (const cls of classes) {
      for (let i = 0; i < y_true.length; i++) {
        const t = y_true[i] === cls;
        const p = (y_pred[i] ?? -1) === cls;
        if (t && p) tp++;
        else if (!t && p) fp++;
        else if (t && !p) fn++;
      }
    }
    return tp / (tp + fp + fn || 1);
  }
  // macro
  const scores = classes.map(cls => {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < y_true.length; i++) {
      const t = y_true[i] === cls;
      const p = (y_pred[i] ?? -1) === cls;
      if (t && p) tp++;
      else if (!t && p) fp++;
      else if (t && !p) fn++;
    }
    return tp / (tp + fp + fn || 1);
  });
  return scores.reduce((a, b) => a + b, 0) / (scores.length || 1);
}

/** Hamming loss for multi-label classification. */
export function hammingLoss(y_true: Int32Array[], y_pred: Int32Array[]): number {
  const n = y_true.length;
  const k = y_true[0]?.length ?? 0;
  let errors = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      if ((y_true[i]?.[j] ?? 0) !== (y_pred[i]?.[j] ?? 0)) errors++;
    }
  }
  return errors / (n * k || 1);
}

/** Zero-one loss (fraction of misclassified samples). */
export function zeroOneLoss(y_true: Int32Array, y_pred: Int32Array, normalize = true): number {
  let wrong = 0;
  for (let i = 0; i < y_true.length; i++) {
    if ((y_true[i] ?? 0) !== (y_pred[i] ?? 0)) wrong++;
  }
  return normalize ? wrong / (y_true.length || 1) : wrong;
}

/** Compute average precision at k for ranking. */
export function averagePrecisionAtK(
  y_true: Int32Array,
  y_score: Float64Array,
  k: number,
): number {
  const n = y_true.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (y_score[b] ?? 0) - (y_score[a] ?? 0));
  let hits = 0;
  let ap = 0;
  for (let rank = 0; rank < Math.min(k, n); rank++) {
    if ((y_true[order[rank] ?? 0] ?? 0) === 1) {
      hits++;
      ap += hits / (rank + 1);
    }
  }
  return ap / (hits || 1);
}
