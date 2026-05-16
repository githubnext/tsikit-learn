/**
 * Additional classification metrics not in classification.ts
 * Ports: balanced_accuracy_score, fbeta_score, brier_score_loss,
 *        matthews_corrcoef, cohen_kappa_score, hinge_loss, zero_one_loss
 */

/**
 * Balanced accuracy — average of recall per class.
 * For binary classification this equals (sensitivity + specificity) / 2.
 */
export function balancedAccuracyScore(
  yTrue: Int32Array | number[],
  yPred: Int32Array | number[],
  adjusted = false,
): number {
  const classes = new Set<number>();
  for (const v of yTrue) classes.add(v);
  const sorted = [...classes].sort((a, b) => a - b);
  let sum = 0;
  for (const c of sorted) {
    let tp = 0;
    let fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if ((yTrue[i] ?? 0) === c) {
        if ((yPred[i] ?? 0) === c) tp++;
        else fn++;
      }
    }
    const support = tp + fn;
    if (support > 0) sum += tp / support;
  }
  const score = sum / sorted.length;
  if (adjusted) {
    const chance = 1 / sorted.length;
    return (score - chance) / (1 - chance);
  }
  return score;
}

/**
 * F-beta score — weighted harmonic mean of precision and recall.
 * beta < 1 favours precision, beta > 1 favours recall.
 */
export function fbetaScore(
  yTrue: Int32Array | number[],
  yPred: Int32Array | number[],
  beta: number,
  average: "binary" | "macro" | "micro" | "weighted" = "binary",
): number {
  const b2 = beta * beta;
  const classes = [...new Set<number>([...yTrue, ...yPred])].sort(
    (a, b) => a - b,
  );

  const perClass = classes.map((c) => {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const t = yTrue[i] ?? 0;
      const p = yPred[i] ?? 0;
      if (t === c && p === c) tp++;
      else if (t !== c && p === c) fp++;
      else if (t === c && p !== c) fn++;
    }
    return { tp, fp, fn };
  });

  if (average === "micro") {
    const tp = perClass.reduce((s, x) => s + x.tp, 0);
    const fp = perClass.reduce((s, x) => s + x.fp, 0);
    const fn = perClass.reduce((s, x) => s + x.fn, 0);
    const denom = (1 + b2) * tp + b2 * fn + fp;
    return denom === 0 ? 0 : ((1 + b2) * tp) / denom;
  }

  const scores = perClass.map(({ tp, fp, fn }) => {
    const denom = (1 + b2) * tp + b2 * fn + fp;
    return denom === 0 ? 0 : ((1 + b2) * tp) / denom;
  });

  if (average === "macro") {
    return scores.reduce((s, x) => s + x, 0) / scores.length;
  }
  if (average === "weighted") {
    const support = classes.map((c) => {
      let cnt = 0;
      for (const v of yTrue) if (v === c) cnt++;
      return cnt;
    });
    const total = support.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;
    return (
      scores.reduce((s, sc, i) => s + sc * (support[i] ?? 0), 0) / total
    );
  }
  // binary: use second class
  return scores[1] ?? scores[0] ?? 0;
}

/**
 * Brier score loss — mean squared difference between probability predictions
 * and true binary outcomes.
 */
export function brierScoreLoss(
  yTrue: Int32Array | number[],
  yProba: Float64Array | number[],
  posLabel = 1,
): number {
  let sum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const t = (yTrue[i] ?? 0) === posLabel ? 1 : 0;
    const p = yProba[i] ?? 0;
    sum += (t - p) ** 2;
  }
  return sum / yTrue.length;
}

/**
 * Matthews correlation coefficient — balanced metric for binary and multi-class.
 */
export function matthewsCorrCoef(
  yTrue: Int32Array | number[],
  yPred: Int32Array | number[],
): number {
  const classes = [...new Set<number>([...yTrue, ...yPred])].sort(
    (a, b) => a - b,
  );
  const K = classes.length;
  const idx = new Map(classes.map((c, i) => [c, i]));
  // confusion matrix
  const cm: number[][] = Array.from({ length: K }, () => new Array(K).fill(0));
  for (let i = 0; i < yTrue.length; i++) {
    const ti = idx.get(yTrue[i] ?? 0) ?? 0;
    const pi = idx.get(yPred[i] ?? 0) ?? 0;
    (cm[ti] as number[])[pi] = ((cm[ti] as number[])[pi] ?? 0) + 1;
  }
  const n = yTrue.length;
  let sumDiag = 0;
  for (let k = 0; k < K; k++) sumDiag += (cm[k] as number[])[k] ?? 0;
  let dotSums = 0;
  for (let k = 0; k < K; k++) {
    const rk = (cm[k] as number[]).reduce((a, b) => a + b, 0);
    const ck = cm.reduce((s, row) => s + ((row[k] ?? 0) as number), 0);
    dotSums += rk * ck;
  }
  let rowSq = 0;
  let colSq = 0;
  for (let k = 0; k < K; k++) {
    const rk = (cm[k] as number[]).reduce((a, b) => a + b, 0);
    const ck = cm.reduce((s, row) => s + ((row[k] ?? 0) as number), 0);
    rowSq += rk * rk;
    colSq += ck * ck;
  }
  const num = n * sumDiag - dotSums;
  const denom = Math.sqrt((n * n - rowSq) * (n * n - colSq));
  return denom === 0 ? 0 : num / denom;
}

/**
 * Cohen's kappa statistic — measures inter-rater agreement.
 */
export function cohenKappaScore(
  y1: Int32Array | number[],
  y2: Int32Array | number[],
): number {
  const classes = [...new Set<number>([...y1, ...y2])].sort((a, b) => a - b);
  const K = classes.length;
  const idx = new Map(classes.map((c, i) => [c, i]));
  const n = y1.length;
  const cm: number[][] = Array.from({ length: K }, () => new Array(K).fill(0));
  for (let i = 0; i < n; i++) {
    const ti = idx.get(y1[i] ?? 0) ?? 0;
    const pi = idx.get(y2[i] ?? 0) ?? 0;
    (cm[ti] as number[])[pi] = ((cm[ti] as number[])[pi] ?? 0) + 1;
  }
  let po = 0;
  for (let k = 0; k < K; k++) po += (cm[k] as number[])[k] ?? 0;
  po /= n;
  let pe = 0;
  for (let k = 0; k < K; k++) {
    const rk = (cm[k] as number[]).reduce((a, b) => a + b, 0) / n;
    const ck = cm.reduce((s, row) => s + ((row[k] ?? 0) as number), 0) / n;
    pe += rk * ck;
  }
  return pe === 1 ? 1 : (po - pe) / (1 - pe);
}

/**
 * Hinge loss — used by SVMs for classification.
 */
export function hingeLoss(
  yTrue: Int32Array | number[],
  predDecision: Float64Array | number[],
  labels?: number[],
): number {
  // binary: map labels to {-1, +1}
  const classes = labels ?? [...new Set<number>([...yTrue])].sort((a, b) => a - b);
  if (classes.length === 2) {
    const neg = classes[0] ?? -1;
    let sum = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const t = (yTrue[i] ?? 0) === neg ? -1 : 1;
      sum += Math.max(0, 1 - t * (predDecision[i] ?? 0));
    }
    return sum / yTrue.length;
  }
  // multiclass: OvA
  let sum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const trueIdx = classes.indexOf(yTrue[i] ?? 0);
    let maxOther = Number.NEGATIVE_INFINITY;
    for (let k = 0; k < classes.length; k++) {
      if (k !== trueIdx) {
        const v = Array.isArray(predDecision)
          ? (predDecision[i] ?? 0)
          : (predDecision[i] ?? 0);
        if (v > maxOther) maxOther = v;
      }
    }
    const trueScore = Array.isArray(predDecision)
      ? (predDecision[i] ?? 0)
      : (predDecision[i] ?? 0);
    sum += Math.max(0, 1 + maxOther - trueScore);
  }
  return sum / yTrue.length;
}

/**
 * Zero-one loss — fraction (or count) of misclassifications.
 */
export function zeroOneLoss(
  yTrue: Int32Array | number[],
  yPred: Int32Array | number[],
  normalize = true,
): number {
  let wrong = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if ((yTrue[i] ?? 0) !== (yPred[i] ?? 0)) wrong++;
  }
  return normalize ? wrong / yTrue.length : wrong;
}
