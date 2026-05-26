/**
 * KDD Cup datasets: synthetic versions of network intrusion data.
 */

export interface KDDCupDataset {
  data: Float64Array[];
  target: Int32Array;
  featureNames: string[];
  targetNames: string[];
  nSamples: number;
  nFeatures: number;
  description: string;
}

export const KDD_FEATURE_NAMES = [
  "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
  "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
  "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
  "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
  "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
  "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
  "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
  "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
  "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
] as const;

export const KDD_TARGET_NAMES = ["normal", "dos", "probe", "r2l", "u2r"] as const;

export function makeKDDCupSynthetic(nSamples = 500, seed = 42): KDDCupDataset {
  const rng = seededRng(seed);
  const nFeatures = KDD_FEATURE_NAMES.length;
  const nClasses = KDD_TARGET_NAMES.length;
  const data: Float64Array[] = [];
  const target: number[] = [];

  for (let i = 0; i < nSamples; i++) {
    const cls = Math.floor(rng() * nClasses);
    const x = new Float64Array(nFeatures);
    // Generate class-specific features
    for (let f = 0; f < nFeatures; f++) {
      x[f] = rng() * 100 + cls * 5;
    }
    // Specific feature patterns per class
    switch (cls) {
      case 0: // normal
        x[0] = rng() * 10; // short duration
        x[5] = rng() * 1000; // some dst_bytes
        break;
      case 1: // dos
        x[4] = rng() * 10000 + 5000; // high src_bytes
        x[22] = rng() * 200 + 100; // high count
        break;
      case 2: // probe
        x[22] = rng() * 100; // count
        x[24] = rng(); // serror_rate
        break;
      case 3: // r2l
        x[11] = 0; // not logged in
        x[9] = rng() * 5; // low hot
        break;
      case 4: // u2r
        x[14] = 1; // su_attempted
        x[13] = 1; // root_shell
        break;
    }
    data.push(x);
    target.push(cls);
  }

  return {
    data,
    target: new Int32Array(target),
    featureNames: [...KDD_FEATURE_NAMES],
    targetNames: [...KDD_TARGET_NAMES],
    nSamples,
    nFeatures,
    description: "Synthetic KDD Cup 1999 network intrusion dataset. Each row is a network connection with class labels: normal, dos, probe, r2l, u2r.",
  };
}

function seededRng(seed: number): () => number {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
}

export function loadKDDCup99(nSamples = 494021, seed = 42): KDDCupDataset {
  return makeKDDCupSynthetic(Math.min(nSamples, 10000), seed);
}
