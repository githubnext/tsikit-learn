/**
 * classification_report and precision_recall_fscore_support.
 * Mirrors sklearn.metrics classification_report.
 */

export interface ClassificationReportOptions {
  labels?: Int32Array;
  targetNames?: string[];
  outputDict?: boolean;
  digits?: number;
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1Score: number;
  support: number;
}

export interface ClassificationReportResult {
  classes: Record<string, ClassMetrics>;
  accuracy: number;
  macroAvg: ClassMetrics;
  weightedAvg: ClassMetrics;
}

function computeClassMetrics(
  yTrue: Int32Array,
  yPred: Int32Array,
  label: number,
): ClassMetrics {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let support = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i] ?? 0;
    const p = yPred[i] ?? 0;
    if (t === label) {
      support++;
      if (p === label) tp++;
      else fn++;
    } else if (p === label) {
      fp++;
    }
  }
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1Score =
    precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : 0;
  return { precision, recall, f1Score, support };
}

export function classificationReport(
  yTrue: Int32Array,
  yPred: Int32Array,
  opts: ClassificationReportOptions = {},
): ClassificationReportResult {
  const classSet = new Set<number>();
  for (let i = 0; i < yTrue.length; i++) classSet.add(yTrue[i] ?? 0);
  const labels =
    opts.labels ?? Int32Array.from(Array.from(classSet).sort((a, b) => a - b));

  const classes: Record<string, ClassMetrics> = {};
  for (let li = 0; li < labels.length; li++) {
    const label = labels[li] ?? 0;
    const name = opts.targetNames?.[li] ?? String(label);
    classes[name] = computeClassMetrics(yTrue, yPred, label);
  }

  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) if (yTrue[i] === yPred[i]) correct++;
  const accuracy = yTrue.length > 0 ? correct / yTrue.length : 0;

  const allMetrics = Object.values(classes);
  const totalSupport = allMetrics.reduce((s, m) => s + m.support, 0);

  const macroAvg: ClassMetrics = {
    precision:
      allMetrics.reduce((s, m) => s + m.precision, 0) / allMetrics.length,
    recall: allMetrics.reduce((s, m) => s + m.recall, 0) / allMetrics.length,
    f1Score: allMetrics.reduce((s, m) => s + m.f1Score, 0) / allMetrics.length,
    support: totalSupport,
  };

  const weightedAvg: ClassMetrics = {
    precision:
      allMetrics.reduce((s, m) => s + m.precision * m.support, 0) /
      totalSupport,
    recall:
      allMetrics.reduce((s, m) => s + m.recall * m.support, 0) / totalSupport,
    f1Score:
      allMetrics.reduce((s, m) => s + m.f1Score * m.support, 0) / totalSupport,
    support: totalSupport,
  };

  return { classes, accuracy, macroAvg, weightedAvg };
}

export function precisionRecallFscoreSupport(
  yTrue: Int32Array,
  yPred: Int32Array,
  opts: {
    average?: "macro" | "weighted" | "micro" | null;
    labels?: Int32Array;
  } = {},
):
  | { precision: number; recall: number; fScore: number; support: number }
  | {
      precisions: Float64Array;
      recalls: Float64Array;
      fScores: Float64Array;
      supports: Int32Array;
    } {
  const classSet = new Set<number>();
  for (let i = 0; i < yTrue.length; i++) classSet.add(yTrue[i] ?? 0);
  const labels =
    opts.labels ?? Int32Array.from(Array.from(classSet).sort((a, b) => a - b));

  const metrics = Array.from({ length: labels.length }, (_, li) =>
    computeClassMetrics(yTrue, yPred, labels[li] ?? 0),
  );

  if (opts.average === null || opts.average === undefined) {
    return {
      precisions: Float64Array.from(metrics, (m) => m.precision),
      recalls: Float64Array.from(metrics, (m) => m.recall),
      fScores: Float64Array.from(metrics, (m) => m.f1Score),
      supports: Int32Array.from(metrics, (m) => m.support),
    };
  }

  const totalSupport = metrics.reduce((s, m) => s + m.support, 0);

  if (opts.average === "micro") {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    for (let li = 0; li < labels.length; li++) {
      const label = labels[li] ?? 0;
      for (let i = 0; i < yTrue.length; i++) {
        const t = yTrue[i] ?? 0;
        const p = yPred[i] ?? 0;
        if (t === label && p === label) tp++;
        else if (t !== label && p === label) fp++;
        else if (t === label && p !== label) fn++;
      }
    }
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const fScore =
      precision + recall > 0
        ? (2 * precision * recall) / (precision + recall)
        : 0;
    return { precision, recall, fScore, support: totalSupport };
  }

  if (opts.average === "weighted") {
    return {
      precision:
        metrics.reduce((s, m) => s + m.precision * m.support, 0) / totalSupport,
      recall:
        metrics.reduce((s, m) => s + m.recall * m.support, 0) / totalSupport,
      fScore:
        metrics.reduce((s, m) => s + m.f1Score * m.support, 0) / totalSupport,
      support: totalSupport,
    };
  }

  // macro average
  return {
    precision: metrics.reduce((s, m) => s + m.precision, 0) / metrics.length,
    recall: metrics.reduce((s, m) => s + m.recall, 0) / metrics.length,
    fScore: metrics.reduce((s, m) => s + m.f1Score, 0) / metrics.length,
    support: totalSupport,
  };
}
