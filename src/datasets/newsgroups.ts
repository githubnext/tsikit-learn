/**
 * Fetch 20 Newsgroups text dataset (simulated/stub).
 * Mirrors sklearn.datasets.fetch_20newsgroups and fetch_20newsgroups_vectorized.
 */

/** Available 20 newsgroups target names. */
export const NEWSGROUPS_CATEGORIES: string[] = [
  "alt.atheism",
  "comp.graphics",
  "comp.os.ms-windows.misc",
  "comp.sys.ibm.pc.hardware",
  "comp.sys.mac.hardware",
  "comp.windows.x",
  "misc.forsale",
  "rec.autos",
  "rec.motorcycles",
  "rec.sport.baseball",
  "rec.sport.hockey",
  "sci.crypt",
  "sci.electronics",
  "sci.med",
  "sci.space",
  "soc.religion.christian",
  "talk.politics.guns",
  "talk.politics.mideast",
  "talk.politics.misc",
  "talk.religion.misc",
];

export interface NewsgroupsDataset {
  data: string[];
  target: Int32Array;
  targetNames: string[];
  description: string;
  filenames: string[];
}

/**
 * Simulate fetching 20 Newsgroups text dataset.
 * In the browser/Node environment this returns synthetic examples.
 * Mirrors sklearn.datasets.fetch_20newsgroups.
 */
export function fetch20Newsgroups(options: {
  subset?: "train" | "test" | "all";
  categories?: string[];
  shuffle?: boolean;
  randomState?: number;
  removeHeaders?: boolean;
  removeFooters?: boolean;
  removeQuotes?: boolean;
  nSamples?: number;
} = {}): NewsgroupsDataset {
  const categories = options.categories ?? NEWSGROUPS_CATEGORIES;
  const nSamples = options.nSamples ?? categories.length * 5;
  const subset = options.subset ?? "train";

  const targetNames = categories.filter(c => NEWSGROUPS_CATEGORIES.includes(c));
  const data: string[] = [];
  const targetArr: number[] = [];
  const filenames: string[] = [];

  const rng = mulberry32((options.randomState ?? 42) + (subset === "test" ? 1000 : 0));

  for (let i = 0; i < nSamples; i++) {
    const catIdx = Math.floor(rng() * targetNames.length);
    const catName = targetNames[catIdx] ?? "misc.forsale";
    data.push(syntheticPost(catName, i, rng));
    targetArr.push(catIdx);
    filenames.push(`${catName}/${1000 + i}`);
  }

  if (options.shuffle ?? false) {
    const order = Array.from({ length: nSamples }, (_, i) => i).sort(
      () => rng() - 0.5,
    );
    const shuffledData = order.map(i => data[i]!);
    const shuffledTarget = order.map(i => targetArr[i] ?? 0);
    const shuffledFiles = order.map(i => filenames[i]!);
    return {
      data: shuffledData,
      target: new Int32Array(shuffledTarget),
      targetNames,
      description: "20 Newsgroups text dataset (synthetic stub)",
      filenames: shuffledFiles,
    };
  }

  return {
    data,
    target: new Int32Array(targetArr),
    targetNames,
    description: "20 Newsgroups text dataset (synthetic stub)",
    filenames,
  };
}

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const categoryWords: Record<string, string[]> = {
  "comp.graphics": ["pixel", "image", "render", "texture", "OpenGL", "3D", "graphics", "polygon"],
  "rec.sport.baseball": ["pitcher", "batter", "home run", "inning", "MLB", "baseball", "score"],
  "rec.sport.hockey": ["puck", "goal", "NHL", "skate", "hockey", "ice", "player", "team"],
  "sci.space": ["orbit", "NASA", "rocket", "satellite", "planet", "launch", "mission", "moon"],
  "sci.med": ["drug", "patient", "doctor", "treatment", "clinical", "disease", "medicine"],
  "sci.crypt": ["encryption", "RSA", "key", "cipher", "algorithm", "cryptography", "secure"],
  "talk.politics.guns": ["gun", "NRA", "Second Amendment", "firearm", "rights", "ban", "crime"],
};

function syntheticPost(category: string, seed: number, rng: () => number): string {
  const words = categoryWords[category] ?? ["news", "article", "post", "discussion"];
  const selected = Array.from({ length: 5 }, () => words[Math.floor(rng() * words.length)] ?? "news");
  return `From: user${seed}@example.com\nSubject: Re: ${selected[0]}\n\n${selected.join(" ")} is an interesting topic in ${category}.\nSee related post #${Math.floor(rng() * 10000)}.`;
}
