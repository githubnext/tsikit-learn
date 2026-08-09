/**
 * Image feature extraction utilities.
 * Images are represented as Float64Array[] (array of rows, each row is a Float64Array of pixel values).
 */

import { NotFittedError } from "../exceptions.js";

/**
 * Extract 2D patches from an image.
 * @param image - 2D image as Float64Array[] (rows), each row has `width` elements.
 * @param patchSize - [patchHeight, patchWidth]
 * @param maxPatches - optional maximum number of patches to extract
 */
export function extractPatches2d(
  image: Float64Array[],
  patchSize: [number, number],
  maxPatches?: number,
): Float64Array[] {
  const imgH = image.length;
  const imgW = (image[0] ?? new Float64Array(0)).length;
  const [pH, pW] = patchSize;
  const patches: Float64Array[] = [];

  for (let r = 0; r <= imgH - pH; r++) {
    for (let c = 0; c <= imgW - pW; c++) {
      const patch = new Float64Array(pH * pW);
      for (let pr = 0; pr < pH; pr++) {
        const row = image[r + pr] ?? new Float64Array(0);
        for (let pc = 0; pc < pW; pc++) {
          patch[pr * pW + pc] = row[c + pc] ?? 0;
        }
      }
      patches.push(patch);
      if (maxPatches !== undefined && patches.length >= maxPatches)
        return patches;
    }
  }
  return patches;
}

/**
 * Reconstruct a 2D image (as Float64Array[]) from overlapping patches by averaging.
 */
export function reconstructFromPatches2d(
  patches: Float64Array[],
  imageSize: [number, number],
  patchSize: [number, number],
): Float64Array[] {
  const [imgH, imgW] = imageSize;
  const [pH, pW] = patchSize;
  const image: Float64Array[] = Array.from(
    { length: imgH },
    () => new Float64Array(imgW),
  );
  const counts: Float64Array[] = Array.from(
    { length: imgH },
    () => new Float64Array(imgW),
  );

  let patchIdx = 0;
  for (let r = 0; r <= imgH - pH; r++) {
    for (let c = 0; c <= imgW - pW; c++) {
      if (patchIdx >= patches.length) break;
      const patch = patches[patchIdx++] ?? new Float64Array(pH * pW);
      for (let pr = 0; pr < pH; pr++) {
        const imgRow = image[r + pr] ?? new Float64Array(imgW);
        const cntRow = counts[r + pr] ?? new Float64Array(imgW);
        for (let pc = 0; pc < pW; pc++) {
          imgRow[c + pc]! = (imgRow[c + pc] ?? 0) + (patch[pr * pW + pc] ?? 0);
          cntRow[c + pc]! = (cntRow[c + pc] ?? 0) + 1;
        }
      }
    }
  }

  for (let r = 0; r < imgH; r++) {
    const imgRow = image[r] ?? new Float64Array(imgW);
    const cntRow = counts[r] ?? new Float64Array(imgW);
    for (let c = 0; c < imgW; c++) {
      imgRow[c]! = (imgRow[c] ?? 0) / ((cntRow[c] ?? 1) || 1);
    }
  }
  return image;
}

export interface PatchExtractorOptions {
  patchSize?: [number, number];
  maxPatches?: number;
}

/** Extracts patches from a collection of images. */
export class PatchExtractor {
  private patchSize: [number, number];
  private maxPatches: number | undefined;
  private fitted = false;

  constructor(options: PatchExtractorOptions = {}) {
    this.patchSize = options.patchSize ?? [8, 8];
    this.maxPatches = options.maxPatches;
  }

  fit(_images: Float64Array[][]): this {
    this.fitted = true;
    return this;
  }

  transform(images: Float64Array[][]): Float64Array[] {
    if (!this.fitted) throw new NotFittedError("PatchExtractor");
    const all: Float64Array[] = [];
    for (const img of images) {
      const patches = extractPatches2d(img, this.patchSize, this.maxPatches);
      for (const p of patches) all.push(p);
    }
    return all;
  }

  fitTransform(images: Float64Array[][]): Float64Array[] {
    return this.fit(images).transform(images);
  }
}
