/**
 * Toy datasets: loadDigits and loadLinnerud — analogous to sklearn.datasets._base.
 */

/** A single 8×8 hand-written digit image dataset entry. */
export interface DigitsDataset {
  /** Pixel data: nSamples × 64 (flattened 8×8 images, values 0–16). */
  data: Float64Array;
  /** Target digit labels (0–9). */
  target: Int32Array;
  /** Number of samples. */
  nSamples: number;
  /** Feature names: "pixel_0_0" … "pixel_7_7". */
  featureNames: string[];
  /** Target names: ["0","1",…,"9"]. */
  targetNames: string[];
  /** Description string. */
  DESCR: string;
}

/** The Linnerud multivariate exercise dataset. */
export interface LinnerudDataset {
  /** Exercise data: 20 × 3 (Chins, Situps, Jumps). */
  data: Float64Array;
  /** Physiological measurements: 20 × 3 (Weight, Waist, Pulse). */
  target: Float64Array;
  nSamples: number;
  featureNames: string[];
  targetNames: string[];
  DESCR: string;
}

/**
 * Generates a minimal synthetic digits dataset.
 * Returns nSamples per class (default 10 per digit) arranged as 8×8 pixel blocks.
 */
export function loadDigits(
  options: { nClass?: number; samplesPerClass?: number } = {},
): DigitsDataset {
  const nClass = options.nClass ?? 10;
  const samplesPerClass = options.samplesPerClass ?? 10;
  const nSamples = nClass * samplesPerClass;
  const nFeatures = 64;
  const data = new Float64Array(nSamples * nFeatures);
  const target = new Int32Array(nSamples);
  const rng = mulberry32(42);

  for (let cls = 0; cls < nClass; cls++) {
    // Build a prototype 8×8 pattern for this digit using a seeded pattern
    const proto = new Float64Array(nFeatures);
    const seed = BigInt(cls * 17);
    for (let px = 0; px < nFeatures; px++) {
      const r =
        (seed * 6364136223846793005n + BigInt(px) * 2862933555777941757n) &
        0xffffffffffffn;
      proto[px] = Number(r % 17n); // 0-16
    }

    for (let s = 0; s < samplesPerClass; s++) {
      const row = cls * samplesPerClass + s;
      target[row] = cls;
      for (let px = 0; px < nFeatures; px++) {
        // Add small noise
        const noise = (rng() - 0.5) * 2;
        const val = Math.max(0, Math.min(16, proto[px]! + noise));
        data[row * nFeatures + px] = Math.round(val);
      }
    }
  }

  const featureNames: string[] = [];
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) featureNames.push(`pixel_${r}_${c}`);
  const targetNames = Array.from({ length: nClass }, (_, i) => String(i));

  return {
    data,
    target,
    nSamples,
    featureNames,
    targetNames,
    DESCR: "Optical recognition of handwritten digits (synthetic).",
  };
}

/** Returns the Linnerud dataset (20 samples, 3 exercise features, 3 physiological targets). */
export function loadLinnerud(): LinnerudDataset {
  // Transcribed from sklearn reference data
  const exerciseRaw = [
    5, 162, 60, 2, 110, 60, 12, 101, 101, 12, 105, 37, 13, 155, 58, 4, 101, 42,
    8, 101, 38, 6, 125, 40, 15, 200, 40, 17, 251, 250, 17, 120, 38, 13, 210,
    115, 14, 215, 105, 1, 50, 50, 6, 70, 31, 12, 210, 120, 4, 60, 25, 11, 230,
    80, 15, 225, 73, 2, 110, 43, 10, 150, 75,
  ];
  const physiologicalRaw = [
    191, 36, 50, 189, 37, 52, 193, 38, 58, 162, 35, 62, 189, 35, 46, 182, 36,
    56, 211, 38, 56, 167, 34, 60, 176, 31, 74, 154, 33, 56, 169, 34, 50, 166,
    33, 52, 154, 34, 64, 247, 46, 50, 193, 36, 46, 202, 37, 62, 176, 37, 54,
    157, 32, 52, 156, 33, 54, 138, 33, 68,
  ];

  const nSamples = 20;
  const data = new Float64Array(nSamples * 3);
  const target = new Float64Array(nSamples * 3);
  for (let i = 0; i < nSamples * 3; i++) {
    data[i] = exerciseRaw[i] ?? 0;
    target[i] = physiologicalRaw[i] ?? 0;
  }

  return {
    data,
    target,
    nSamples,
    featureNames: ["Chins", "Situps", "Jumps"],
    targetNames: ["Weight", "Waist", "Pulse"],
    DESCR:
      "Linnerud physical exercise dataset (20 middle-aged men, 3 exercise × 3 physiological).",
  };
}

// --- helpers ---

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let z = Math.imul(s ^ (s >>> 15), 1 | s);
    z ^= z + Math.imul(z ^ (z >>> 7), 61 | z);
    return ((z ^ (z >>> 14)) >>> 0) / 0x100000000;
  };
}
