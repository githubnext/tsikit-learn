/**
 * Individual Conditional Expectation (ICE) utilities.
 * Extends partial dependence with per-sample ICE curves.
 */

export interface ICEResult {
  gridValues: Float64Array[];
  averages: Float64Array[];
  individual: Float64Array[][];
}

export interface ICEEstimator {
  predict(X: Float64Array[]): Float64Array | Int32Array;
}

/**
 * Compute ICE curves and partial dependence averages for the given features.
 *
 * @param estimator - Fitted estimator with a `predict` method.
 * @param X - Training data [n_samples × n_features].
 * @param features - Feature indices to compute ICE/PD for.
 * @param gridResolution - Number of grid points per feature (default 100).
 */
export function computeICE(
  estimator: ICEEstimator,
  X: Float64Array[],
  features: number[],
  gridResolution = 100,
): ICEResult {
  const n = X.length;
  const gridValues: Float64Array[] = [];
  const averages: Float64Array[] = [];
  const individual: Float64Array[][] = [];

  for (const feat of features) {
    const colVals = Float64Array.from(
      { length: n },
      (_, i) => (X[i] ?? new Float64Array(0))[feat] ?? 0,
    );
    const sorted = colVals.slice().sort();
    const gridSize = Math.min(gridResolution, n);
    const grid = new Float64Array(gridSize);
    for (let g = 0; g < gridSize; g++) {
      const idx = Math.round((g / (gridSize - 1 || 1)) * (sorted.length - 1));
      grid[g] = sorted[idx] ?? 0;
    }
    gridValues.push(grid);

    const avg = new Float64Array(gridSize);
    const indiv: Float64Array[] = Array.from(
      { length: n },
      () => new Float64Array(gridSize),
    );

    for (let g = 0; g < gridSize; g++) {
      const Xmod: Float64Array[] = X.map((row) => {
        const r = row.slice();
        r[feat]! = grid[g] ?? 0;
        return r;
      });
      const preds = estimator.predict(Xmod);
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const p = Number(preds[i] ?? 0);
        (indiv[i] as Float64Array)[g] = p;
        sum += p;
      }
      avg[g] = sum / (n || 1);
    }

    averages.push(avg);
    individual.push(indiv);
  }

  return { gridValues, averages, individual };
}

/** Stores ICE/PD results and provides a simple SVG plot. */
export class PartialDependenceDisplay {
  result: ICEResult;
  featureNames: string[];

  constructor(result: ICEResult, featureNames: string[] = []) {
    this.result = result;
    this.featureNames = featureNames;
  }

  /** Returns a minimal SVG string visualising the partial dependence curves. */
  plot(width = 400, height = 300): string {
    const { gridValues, averages } = this.result;
    const margin = 40;
    const plotW = width - 2 * margin;
    const plotH = height - 2 * margin;

    const paths = gridValues
      .map((grid, fi) => {
        const avg = averages[fi] ?? new Float64Array(0);
        if (grid.length === 0) return "";

        let minX = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        let minY = Number.POSITIVE_INFINITY;
        let maxY = Number.NEGATIVE_INFINITY;
        for (let g = 0; g < grid.length; g++) {
          const x = grid[g] ?? 0;
          const y = avg[g] ?? 0;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
        const xRange = maxX - minX || 1;
        const yRange = maxY - minY || 1;

        const pts = Array.from({ length: grid.length }, (_, g) => {
          const px = margin + (((grid[g] ?? 0) - minX) / xRange) * plotW;
          const py = margin + plotH - (((avg[g] ?? 0) - minY) / yRange) * plotH;
          return `${px.toFixed(1)},${py.toFixed(1)}`;
        }).join(" ");

        const label = this.featureNames[fi] ?? `feature ${fi}`;
        return `<polyline points="${pts}" fill="none" stroke="steelblue" stroke-width="2"/><text x="${margin}" y="${margin - 8 + fi * 14}" font-size="10">${label}</text>`;
      })
      .join("");

    return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="${width}" height="${height}" fill="white"/>${paths}</svg>`;
  }
}
