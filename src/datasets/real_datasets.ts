/**
 * Real-world dataset generators and synthetic alternatives.
 * Mirrors sklearn.datasets (california_housing, covtype, kddcup99, etc.)
 */

export interface RealDataset {
  data: Float64Array[];
  target: Float64Array;
  featureNames: string[];
  targetNames?: string[];
  description: string;
}

export interface RealClassificationDataset extends RealDataset {
  target: Float64Array; // integer class labels as floats
  classes: Int32Array;
}

/**
 * Generate a synthetic version of the California Housing dataset.
 * The real dataset has 20,640 instances and 8 features.
 * This generator produces a statistically similar synthetic dataset.
 *
 * Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
 * Target: median house value (in $100k)
 */
export function makeCaliforniaHousing(
  options: {
    nSamples?: number;
    noise?: number;
    seed?: number;
  } = {},
): RealDataset {
  const { nSamples = 1000, noise = 0.1, seed = 42 } = options;
  let rng = seed;
  const rand = () => {
    rng = (rng * 1664525 + 1013904223) & 0xffffffff;
    return (rng >>> 0) / 0xffffffff;
  };
  const randn = () => {
    const u = rand() || 1e-10;
    const v = rand() || 1e-10;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  const featureNames = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
  ];

  const data: Float64Array[] = [];
  const target = new Float64Array(nSamples);

  for (let i = 0; i < nSamples; i++) {
    const medInc = Math.max(0.5, 3.0 + randn() * 2.0);
    const houseAge = Math.max(1, Math.min(52, 28 + randn() * 12));
    const aveRooms = Math.max(1, 5.4 + randn() * 2.0);
    const aveBedrms = Math.max(0.5, 1.1 + randn() * 0.4);
    const population = Math.max(10, 1400 + randn() * 1100);
    const aveOccup = Math.max(1, 3.0 + randn() * 1.5);
    const latitude = 35.6 + randn() * 2.1;
    const longitude = -119.6 + randn() * 2.0;

    const row = new Float64Array([
      medInc,
      houseAge,
      aveRooms,
      aveBedrms,
      population,
      aveOccup,
      latitude,
      longitude,
    ]);
    data.push(row);

    // Approximate the California housing formula
    target[i] = Math.max(
      0.15,
      Math.min(
        5.0,
        0.4524 * medInc -
          0.0104 * houseAge +
          0.0 * aveRooms -
          0.0 * aveBedrms -
          (0.0 * population) / 1000 -
          0.0 * aveOccup -
          0.042 * latitude +
          0.0 * longitude +
          2.1 +
          randn() * noise,
      ),
    );
  }

  return {
    data,
    target,
    featureNames,
    description: "Synthetic California Housing dataset (sklearn-compatible)",
  };
}

/**
 * Generate a synthetic version of the Forest Covertype dataset.
 * The real dataset has 581,012 instances and 54 features with 7 cover types.
 *
 * Returns integer class labels 1-7 for cover type.
 */
export function makeCovtype(
  options: {
    nSamples?: number;
    seed?: number;
  } = {},
): RealClassificationDataset {
  const { nSamples = 500, seed = 42 } = options;
  let rng = seed;
  const rand = () => {
    rng = (rng * 1664525 + 1013904223) & 0xffffffff;
    return (rng >>> 0) / 0xffffffff;
  };
  const randn = () => {
    const u = rand() || 1e-10;
    const v = rand() || 1e-10;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  // 54 features: 10 continuous, 4 binary wilderness areas, 40 binary soil types
  const continuousFeatureNames = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
  ];
  const wildernessNames = [
    "Wilderness_Area1",
    "Wilderness_Area2",
    "Wilderness_Area3",
    "Wilderness_Area4",
  ];
  const soilNames = Array.from({ length: 40 }, (_, i) => `Soil_Type${i + 1}`);
  const featureNames = [
    ...continuousFeatureNames,
    ...wildernessNames,
    ...soilNames,
  ];

  const data: Float64Array[] = [];
  const target = new Float64Array(nSamples);
  const classes = new Int32Array([1, 2, 3, 4, 5, 6, 7]);

  // Cover type priors (approximate): 1=36.5%, 2=48.7%, 3=6.2%, 4=0.5%, 5=1.6%, 6=2.9%, 7=3.5%
  const priors = [0.365, 0.487, 0.062, 0.005, 0.016, 0.029, 0.035];
  const cdf = priors.reduce<number[]>((acc, p, i) => {
    acc.push((acc[i - 1] ?? 0) + p);
    return acc;
  }, []);

  for (let i = 0; i < nSamples; i++) {
    // Sample class label
    const u = rand();
    let cls = 1;
    for (let c = 0; c < cdf.length; c++) {
      if (u <= (cdf[c] ?? 1)) {
        cls = c + 1;
        break;
      }
    }
    target[i] = cls;

    // Continuous features (mean/std approximate per class)
    const elevation = 2800 + cls * 50 + randn() * 200;
    const aspect = 180 + randn() * 90;
    const slope = 12 + randn() * 8;
    const horizHydro = 300 + randn() * 250;
    const vertHydro = 20 + randn() * 50;
    const horizRoad = 2000 + randn() * 1500;
    const hillshade9am = Math.max(0, Math.min(255, 200 + randn() * 40));
    const hillshadeNoon = Math.max(0, Math.min(255, 220 + randn() * 30));
    const hillshade3pm = Math.max(0, Math.min(255, 135 + randn() * 60));
    const horizFire = 1500 + randn() * 1200;

    // Binary wilderness area (one-hot)
    const wArea = Math.floor(rand() * 4);
    const w = new Float64Array(4);
    w[wArea] = 1;

    // Binary soil type (one-hot among 40)
    const sType = Math.floor(rand() * 40);
    const s = new Float64Array(40);
    s[sType] = 1;

    const row = new Float64Array([
      elevation,
      aspect,
      slope,
      horizHydro,
      vertHydro,
      horizRoad,
      hillshade9am,
      hillshadeNoon,
      hillshade3pm,
      horizFire,
      ...w,
      ...s,
    ]);
    data.push(row);
  }

  return {
    data,
    target,
    featureNames,
    targetNames: [
      "Spruce/Fir",
      "Lodgepole Pine",
      "Ponderosa Pine",
      "Cottonwood/Willow",
      "Aspen",
      "Douglas-fir",
      "Krummholz",
    ],
    classes,
    description:
      "Synthetic Covertype dataset (sklearn-compatible, 7 classes, 54 features)",
  };
}

/**
 * Generate a synthetic version of the KDD Cup 1999 dataset.
 * Returns a simplified intrusion detection dataset.
 *
 * @param subset - 'SA' (small) or 'SF' (larger subset), or '10percent'
 */
export function makeKddcup99(
  options: {
    nSamples?: number;
    subset?: "SA" | "SF" | "10percent";
    percentAnomalies?: number;
    seed?: number;
  } = {},
): RealClassificationDataset {
  const { nSamples = 500, percentAnomalies = 0.2, seed = 42 } = options;

  let rng = seed;
  const rand = () => {
    rng = (rng * 1664525 + 1013904223) & 0xffffffff;
    return (rng >>> 0) / 0xffffffff;
  };
  const randn = () => {
    const u = rand() || 1e-10;
    const v = rand() || 1e-10;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  const featureNames = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
  ];

  const nAnomalies = Math.floor(nSamples * percentAnomalies);
  const nNormal = nSamples - nAnomalies;

  const data: Float64Array[] = [];
  const target = new Float64Array(nSamples);
  // 0 = normal, 1 = anomaly
  const classes = new Int32Array([0, 1]);

  for (let i = 0; i < nSamples; i++) {
    const isAnomaly = i < nAnomalies;
    target[i] = isAnomaly ? 1 : 0;

    const row = new Float64Array(featureNames.length);
    if (isAnomaly) {
      // Anomaly pattern: high src_bytes, high error rates
      row[0] = Math.max(0, randn() * 2);
      row[4] = Math.max(0, 100000 + randn() * 50000);
      row[5] = Math.max(0, randn() * 100);
      row[24] = Math.max(0, Math.min(1, 0.8 + randn() * 0.2));
      row[26] = Math.max(0, Math.min(1, 0.7 + randn() * 0.2));
    } else {
      // Normal: small transfers, low error
      row[0] = Math.max(0, randn() * 5);
      row[4] = Math.max(0, 500 + randn() * 1000);
      row[5] = Math.max(0, 2000 + randn() * 3000);
      row[24] = Math.max(0, Math.min(1, 0.02 + randn() * 0.05));
      row[26] = Math.max(0, Math.min(1, 0.01 + randn() * 0.03));
    }
    row[22] = Math.max(0, Math.min(511, Math.abs(randn() * 50 + 10)));
    row[31] = Math.max(0, Math.min(255, Math.abs(randn() * 50 + 100)));
    data.push(row);
  }

  // Shuffle
  for (let i = nSamples - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = data[i]!;
    data[i] = data[j]!;
    data[j] = tmp;
    const ttmp = target[i]!;
    target[i] = target[j]!;
    target[j] = ttmp;
  }

  _ = nNormal; // suppress unused var

  return {
    data,
    target,
    featureNames,
    targetNames: ["normal", "anomaly"],
    classes,
    description: "Synthetic KDD Cup 1999 network intrusion detection dataset",
  };
}

// Suppress TS unused variable error
let _: number;

/**
 * Load a synthetic version of the Olivetti faces dataset.
 * 400 samples, 64x64 pixel face images (4096 features), 40 subjects.
 */
export function makeOlivettiFaces(
  options: {
    nSamples?: number;
    nSubjects?: number;
    seed?: number;
  } = {},
): RealDataset {
  const { nSamples = 400, nSubjects = 40, seed = 42 } = options;
  let rng = seed;
  const rand = () => {
    rng = (rng * 1664525 + 1013904223) & 0xffffffff;
    return (rng >>> 0) / 0xffffffff;
  };
  const randn = () => {
    const u = rand() || 1e-10;
    const v = rand() || 1e-10;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  const nFeatures = 4096; // 64x64
  const data: Float64Array[] = [];
  const target = new Float64Array(nSamples);
  const featureNames = Array.from(
    { length: nFeatures },
    (_, i) => `pixel_${i}`,
  );

  // Each subject has a "prototype" face
  const prototypes: Float64Array[] = Array.from({ length: nSubjects }, () => {
    const p = new Float64Array(nFeatures);
    for (let f = 0; f < nFeatures; f++) {
      p[f] = Math.max(0, Math.min(1, 0.5 + randn() * 0.2));
    }
    return p;
  });

  for (let i = 0; i < nSamples; i++) {
    const subject = i % nSubjects;
    target[i] = subject;
    const proto = prototypes[subject]!;
    const row = new Float64Array(nFeatures);
    for (let f = 0; f < nFeatures; f++) {
      row[f] = Math.max(0, Math.min(1, proto[f]! + randn() * 0.05));
    }
    data.push(row);
  }

  return {
    data,
    target,
    featureNames,
    targetNames: Array.from({ length: nSubjects }, (_, i) => `subject_${i}`),
    description: `Synthetic Olivetti faces dataset (${nSubjects} subjects, ${nSamples} samples)`,
  };
}
