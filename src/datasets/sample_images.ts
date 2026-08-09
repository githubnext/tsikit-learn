/**
 * Sample image datasets.
 * Mirrors scikit-learn's datasets.load_sample_image and load_sample_images.
 */

export interface SampleImage {
  name: string;
  data: Uint8Array;
  height: number;
  width: number;
  channels: number;
}

/** Available sample image names */
export const SAMPLE_IMAGE_NAMES = ["china", "flower"] as const;
export type SampleImageName = (typeof SAMPLE_IMAGE_NAMES)[number];

/** Generate a synthetic sample image for testing/demos. */
function generateSyntheticImage(
  name: SampleImageName,
  height: number,
  width: number,
): Uint8Array {
  const data = new Uint8Array(height * width * 3);
  let seed = name === "china" ? 1337 : 7331;
  const rng = (): number => {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  };

  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      const base = (i * width + j) * 3;
      if (name === "china") {
        // Sky gradient + random texture
        const t = i / height;
        data[base] = Math.floor(135 + 120 * (1 - t) + rng() * 20);
        data[base + 1] = Math.floor(206 * (1 - t * 0.5) + rng() * 20);
        data[base + 2] = Math.floor(235 * (1 - t * 0.3) + rng() * 20);
      } else {
        // Flower: radial gradient
        const cx = 0.5;
        const cy = 0.5;
        const r = Math.sqrt((j / width - cx) ** 2 + (i / height - cy) ** 2);
        const angle = Math.atan2(i / height - cy, j / width - cx);
        const petal = Math.sin(angle * 6) > 0 ? 1 : 0;
        const inFlower = r < 0.4 ? 1 : 0;
        data[base] = Math.floor(255 * petal * inFlower + rng() * 30);
        data[base + 1] = Math.floor(200 * (1 - r) * inFlower + rng() * 30);
        data[base + 2] = Math.floor(50 * inFlower + rng() * 30);
      }
    }
  }
  return data;
}

/**
 * Load a single sample image by name.
 */
export function loadSampleImage(imageName: SampleImageName): SampleImage {
  const height = 427;
  const width = imageName === "china" ? 640 : 483;
  return {
    name: imageName,
    data: generateSyntheticImage(imageName, height, width),
    height,
    width,
    channels: 3,
  };
}

/**
 * Load all sample images.
 */
export function loadSampleImages(): SampleImage[] {
  return SAMPLE_IMAGE_NAMES.map((name) => loadSampleImage(name));
}
