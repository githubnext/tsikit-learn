/**
 * Metrics extensions: ranking metrics, detection metrics, survival metrics
 */

export function averagePrecisionScore(yTrue: Int32Array, yScore: Float64Array): number {
  const n = yTrue.length;
  const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  let tp = 0, fp = 0;
  let ap = 0;
  let prevPrec = 0;
  for (let i = 0; i < n; i++) {
    if ((yTrue[indices[i]!] ?? 0) === 1) {
      tp++;
      const prec = tp / (tp + fp);
      ap += prec;
      prevPrec = prec;
    } else {
      fp++;
    }
  }
  return tp > 0 ? ap / tp : 0;
}

export function rocAucScore(yTrue: Int32Array, yScore: Float64Array, multiClass?: 'ovo' | 'ovr'): number {
  const n = yTrue.length;
  const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  let tp = 0, fp = 0;
  let auc = 0;
  let prevFp = 0, prevTp = 0;
  const nPos = Array.from(yTrue).filter(v => v === 1).length;
  const nNeg = n - nPos;

  for (let i = 0; i < n; i++) {
    if ((yTrue[indices[i]!] ?? 0) === 1) tp++;
    else {
      fp++;
      auc += (tp + prevTp) * (fp - prevFp) / 2;
      prevFp = fp; prevTp = tp;
    }
  }
  auc += (tp + prevTp) * (nNeg - prevFp) / 2;
  return nPos > 0 && nNeg > 0 ? auc / (nPos * nNeg) : 0;
}

export function precisionRecallCurve(yTrue: Int32Array, yScore: Float64Array): { precision: Float64Array; recall: Float64Array; thresholds: Float64Array } {
  const n = yTrue.length;
  const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  const nPos = Array.from(yTrue).filter(v => v === 1).length;
  const precision: number[] = [1], recall: number[] = [0], thresholds: number[] = [];
  let tp = 0, fp = 0;

  for (let i = 0; i < n; i++) {
    if ((yTrue[indices[i]!] ?? 0) === 1) tp++; else fp++;
    precision.push(tp / (tp + fp));
    recall.push(tp / nPos);
    thresholds.push(yScore[indices[i]!] ?? 0);
  }
  return { precision: new Float64Array(precision), recall: new Float64Array(recall), thresholds: new Float64Array(thresholds) };
}

export function brierScore(yTrue: Int32Array, yProb: Float64Array): number {
  let score = 0;
  for (let i = 0; i < yTrue.length; i++) score += ((yTrue[i] ?? 0) - (yProb[i] ?? 0)) ** 2;
  return score / yTrue.length;
}

export function calibrationCurve(yTrue: Int32Array, yProb: Float64Array, nBins: number = 10): { fracPos: Float64Array; meanPredProb: Float64Array } {
  const fracPos: number[] = [], meanPredProb: number[] = [];
  for (let b = 0; b < nBins; b++) {
    const lo = b / nBins, hi = (b + 1) / nBins;
    const inBin = Array.from({ length: yTrue.length }, (_, i) => i).filter(i => (yProb[i] ?? 0) >= lo && (yProb[i] ?? 0) < hi);
    if (inBin.length > 0) {
      fracPos.push(inBin.filter(i => (yTrue[i] ?? 0) === 1).length / inBin.length);
      meanPredProb.push(inBin.reduce((s, i) => s + (yProb[i] ?? 0), 0) / inBin.length);
    }
  }
  return { fracPos: new Float64Array(fracPos), meanPredProb: new Float64Array(meanPredProb) };
}

export function detectionErrorTradeoff(yTrue: Int32Array, yScore: Float64Array): { fpr: Float64Array; fnr: Float64Array; thresholds: Float64Array } {
  const n = yTrue.length;
  const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => (yScore[b] ?? 0) - (yScore[a] ?? 0));
  const nPos = Array.from(yTrue).filter(v => v === 1).length;
  const nNeg = n - nPos;
  const fpr: number[] = [], fnr: number[] = [], thresholds: number[] = [];
  let tp = 0, fp = 0;

  for (let i = 0; i < n; i++) {
    if ((yTrue[indices[i]!] ?? 0) === 1) tp++; else fp++;
    fpr.push(fp / nNeg);
    fnr.push((nPos - tp) / nPos);
    thresholds.push(yScore[indices[i]!] ?? 0);
  }
  return { fpr: new Float64Array(fpr), fnr: new Float64Array(fnr), thresholds: new Float64Array(thresholds) };
}

export function concordanceIndex(eventTimes: Float64Array, predictedRisk: Float64Array, events: Int32Array): number {
  let concordant = 0, discordant = 0, tied = 0;
  const n = eventTimes.length;
  for (let i = 0; i < n; i++) {
    if ((events[i] ?? 0) === 0) continue;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      if ((eventTimes[j] ?? 0) <= (eventTimes[i] ?? 0)) continue;
      const ri = predictedRisk[i] ?? 0, rj = predictedRisk[j] ?? 0;
      if (ri > rj) concordant++;
      else if (ri < rj) discordant++;
      else tied++;
    }
  }
  const total = concordant + discordant + tied;
  return total > 0 ? concordant / total : 0;
}

export function topKAccuracy(yTrue: Int32Array, yScore: Float64Array[], k: number = 5): number {
  const n = yTrue.length;
  let correct = 0;
  for (let i = 0; i < n; i++) {
    const scores = yScore.map((col, c) => ({ c, s: col[i] ?? 0 })).sort((a, b) => b.s - a.s);
    const topK = scores.slice(0, k).map(e => e.c);
    if (topK.includes(yTrue[i] ?? 0)) correct++;
  }
  return correct / n;
}

export function balancedAccuracyScore(yTrue: Int32Array, yPred: Int32Array, adjusted: boolean = false): number {
  const classes = new Set(Array.from(yTrue));
  let recall = 0;
  for (const c of classes) {
    let tp = 0, fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if ((yTrue[i] ?? 0) === c) { if ((yPred[i] ?? 0) === c) tp++; else fn++; }
    }
    recall += tp / (tp + fn + 1e-10);
  }
  const ba = recall / classes.size;
  if (adjusted) return (ba - 1 / classes.size) / (1 - 1 / classes.size);
  return ba;
}

export function geometricMeanScore(yTrue: Int32Array, yPred: Int32Array): number {
  const classes = new Set(Array.from(yTrue));
  let logSum = 0;
  for (const c of classes) {
    let tp = 0, fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if ((yTrue[i] ?? 0) === c) { if ((yPred[i] ?? 0) === c) tp++; else fn++; }
    }
    logSum += Math.log(tp / (tp + fn + 1e-10));
  }
  return Math.exp(logSum / classes.size);
}

export function classLikelihoodRatios(yTrue: Int32Array, yPred: Int32Array): { lrPlus: number; lrMinus: number } {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i] ?? 0, p = yPred[i] ?? 0;
    if (t === 1 && p === 1) tp++;
    else if (t === 0 && p === 1) fp++;
    else if (t === 0 && p === 0) tn++;
    else fn++;
  }
  const sensitivity = tp / (tp + fn + 1e-10);
  const specificity = tn / (tn + fp + 1e-10);
  return {
    lrPlus: sensitivity / (1 - specificity + 1e-10),
    lrMinus: (1 - sensitivity) / (specificity + 1e-10)
  };
}

export function dPrime(hits: number, misses: number, falseAlarms: number, correctRejections: number): number {
  const hitRate = (hits + 0.5) / (hits + misses + 1);
  const faRate = (falseAlarms + 0.5) / (falseAlarms + correctRejections + 1);
  return normInv(hitRate) - normInv(faRate);
}

function normInv(p: number): number {
  // Rational approximation
  if (p <= 0) return -8;
  if (p >= 1) return 8;
  if (p < 0.5) return -normInv(1 - p);
  const t = Math.sqrt(-2 * Math.log(1 - p));
  const c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
  const d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
  return t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3);
}
