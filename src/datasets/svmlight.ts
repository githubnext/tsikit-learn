/**
 * SVMLight format loading and saving utilities.
 * Ports: load_svmlight_file, dump_svmlight_file
 */

export interface SVMLightDataset {
  data: Float64Array[];
  target: Float64Array;
  nFeatures: number;
}

/**
 * Parse SVMLight / LibSVM format text.
 * Format: <label> <index>:<value> <index>:<value> ...
 */
export function loadSvmlightString(
  text: string,
  nFeatures?: number,
  multilabel = false,
): SVMLightDataset {
  const lines = text.split("\n").filter((l) => l.trim() && !l.startsWith("#"));
  const targets: number[] = [];
  const rows: Map<number, number>[] = [];
  let maxFeature = 0;

  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    const labelStr = parts[0] ?? "0";
    // support multilabel: "1,2" or "1"
    if (!multilabel) {
      targets.push(Number.parseFloat(labelStr));
    } else {
      targets.push(Number.parseFloat(labelStr.split(",")[0] ?? "0"));
    }
    const row = new Map<number, number>();
    for (let i = 1; i < parts.length; i++) {
      const pair = parts[i] ?? "";
      const colon = pair.indexOf(":");
      if (colon < 0) continue;
      const idx = Number.parseInt(pair.slice(0, colon), 10);
      const val = Number.parseFloat(pair.slice(colon + 1));
      if (!Number.isNaN(idx) && !Number.isNaN(val)) {
        row.set(idx, val);
        if (idx > maxFeature) maxFeature = idx;
      }
    }
    rows.push(row);
  }

  const numFeatures = nFeatures ?? maxFeature;
  const data: Float64Array[] = rows.map((row) => {
    const arr = new Float64Array(numFeatures);
    for (const [idx, val] of row) {
      // SVMLight uses 1-based indexing
      if (idx >= 1 && idx <= numFeatures) {
        arr[idx - 1] = val;
      }
    }
    return arr;
  });

  return {
    data,
    target: new Float64Array(targets),
    nFeatures: numFeatures,
  };
}

/**
 * Serialize a dataset to SVMLight format string.
 */
export function dumpSvmlightString(
  data: Float64Array[],
  target: Float64Array | Int32Array | number[],
  zeroBaseIndex = false,
): string {
  const lines: string[] = [];
  for (let i = 0; i < data.length; i++) {
    const row = data[i] ?? new Float64Array(0);
    const label = target[i] ?? 0;
    const pairs: string[] = [];
    for (let j = 0; j < row.length; j++) {
      const v = row[j] ?? 0;
      if (v !== 0) {
        const idx = zeroBaseIndex ? j : j + 1;
        pairs.push(`${idx}:${v}`);
      }
    }
    lines.push(`${label} ${pairs.join(" ")}`.trim());
  }
  return lines.join("\n");
}

/**
 * Simple in-memory file loading from an SVMLight format string.
 * In a browser/Node environment, pass the file content as a string.
 */
export function loadSvmlightFile(
  content: string,
  nFeatures?: number,
): SVMLightDataset {
  return loadSvmlightString(content, nFeatures);
}

/**
 * Serialize dataset to SVMLight format (alias for dumpSvmlightString).
 */
export function dumpSvmlightFile(
  data: Float64Array[],
  target: Float64Array | Int32Array | number[],
): string {
  return dumpSvmlightString(data, target);
}
