/**
 * Additional metrics: zero_one_loss, hamming_loss, jaccard_score, balanced_accuracy.
 * Mirrors sklearn.metrics extras.
 */

export function zeroOneLoss(
  yTrue: Int32Array,
  yPred: Int32Array,
  normalize = true,
): number {
  let wrong = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if ((yTrue[i] ?? 0) !== (yPred[i] ?? 0)) wrong++;
  }
  return normalize ? wrong / yTrue.length : wrong;
}

export function hammingLoss(
  yTrue: Int32Array,
  yPred: Int32Array,
): number {
  let diff = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if ((yTrue[i] ?? 0) !== (yPred[i] ?? 0)) diff++;
  }
  return diff / yTrue.length;
}

export function jaccardScore(
  yTrue: Int32Array,
  yPred: Int32Array,
  average: "binary" | "macro" | "micro" | "weighted" = "binary",
): number {
  const classes = Array.from(new Set([...Array.from(yTrue), ...Array.from(yPred)])).sort((a, b) => a - b);

  const scores = classes.map((c) => {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const t = (yTrue[i] ?? 0) === c;
      const p = (yPred[i] ?? 0) === c;
      if (t && p) tp++;
      else if (!t && p) fp++;
      else if (t && !p) fn++;
    }
    return tp + fp + fn > 0 ? tp / (tp + fp + fn) : 0;
  });

  if (average === "micro") {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    for (const c of classes) {
      for (let i = 0; i < yTrue.length; i++) {
        const t = (yTrue[i] ?? 0) === c;
        const p = (yPred[i] ?? 0) === c;
        if (t && p) tp++;
        else if (!t && p) fp++;
        else if (t && !p) fn++;
      }
    }
    return tp + fp + fn > 0 ? tp / (tp + fp + fn) : 0;
  }
  if (average === "binary") return scores[1] ?? scores[0] ?? 0;
  if (average === "macro") return scores.reduce((a, b) => a + b, 0) / scores.length;

  // weighted
  const support = classes.map((c) => Array.from(yTrue).filter((v) => v === c).length);
  const total = support.reduce((a, b) => a + b, 0);
  return total === 0
    ? 0
    : scores.reduce((acc, s, i) => acc + s * (support[i] ?? 0), 0) / total;
}

export function balancedAccuracyScore(
  yTrue: Int32Array,
  yPred: Int32Array,
  adjusted = false,
): number {
  const classes = Array.from(new Set(Array.from(yTrue))).sort((a, b) => a - b);
  const recalls: number[] = [];

  for (const c of classes) {
    let tp = 0;
    let fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if ((yTrue[i] ?? 0) === c) {
        if ((yPred[i] ?? 0) === c) tp++;
        else fn++;
      }
    }
    recalls.push(tp + fn > 0 ? tp / (tp + fn) : 0);
  }

  const score = recalls.reduce((a, b) => a + b, 0) / recalls.length;
  if (adjusted) {
    const k = classes.length;
    return (score - 1 / k) / (1 - 1 / k);
  }
  return score;
}

export function topKAccuracyScore(
  yTrue: Int32Array,
  yScore: Float64Array[],
  k = 2,
): number {
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const scores = Array.from(yScore[i] ?? []);
    const topK = scores
      .map((s, j) => ({ s, j }))
      .sort((a, b) => b.s - a.s)
      .slice(0, k)
      .map((x) => x.j);
    if (topK.includes(yTrue[i] ?? 0)) correct++;
  }
  return correct / yTrue.length;
}
