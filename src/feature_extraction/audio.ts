/**
 * Audio feature extraction utilities.
 * Mirrors sklearn.feature_extraction (audio-adjacent features) and common
 * signal-processing routines used in audio ML pipelines:
 *   - Short-Time Fourier Transform (STFT) magnitude
 *   - Mel-filterbank energies
 *   - Mel-Frequency Cepstral Coefficients (MFCCs)
 *   - Root-mean-square (RMS) energy per frame
 *   - Zero-crossing rate (ZCR) per frame
 */

// ─── DFT helpers ─────────────────────────────────────────────────────────────

/**
 * Compute the magnitude spectrum of a real-valued signal via a naive DFT.
 * For short frames (typical: 256–2048 samples) this is adequate.
 *
 * @param frame - Real signal frame.
 * @returns Magnitude spectrum of length floor(frame.length/2)+1.
 */
export function magnitudeSpectrum(frame: Float64Array): Float64Array {
  const n = frame.length;
  const out = new Float64Array(Math.floor(n / 2) + 1);
  for (let k = 0; k < out.length; k++) {
    let re = 0;
    let im = 0;
    for (let t = 0; t < n; t++) {
      const phi = (2 * Math.PI * k * t) / n;
      re += (frame[t]! ?? 0) * Math.cos(phi);
      im -= (frame[t]! ?? 0) * Math.sin(phi);
    }
    out[k]! = Math.sqrt(re * re + im * im);
  }
  return out;
}

// ─── Windowing ───────────────────────────────────────────────────────────────

/**
 * Apply a Hann window to `frame` in-place and return it.
 */
export function hannWindow(frame: Float64Array): Float64Array {
  const n = frame.length;
  for (let i = 0; i < n; i++) {
    frame[i]! *= 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  }
  return frame;
}

// ─── STFT ────────────────────────────────────────────────────────────────────

export interface STFTOptions {
  nFft?: number;
  hopLength?: number;
  window?: "hann" | "none";
}

/**
 * Compute the STFT magnitude spectrogram of `signal`.
 *
 * @param signal - 1-D audio signal.
 * @param options - FFT size, hop length, window type.
 * @returns Spectrogram as [nFrames x (nFft/2 + 1)] matrix.
 */
export function stftMagnitude(
  signal: Float64Array,
  options: STFTOptions = {},
): Float64Array[] {
  const nFft = options.nFft ?? 512;
  const hopLength = options.hopLength ?? Math.floor(nFft / 4);
  const useHann = (options.window ?? "hann") === "hann";

  const nFrames = Math.max(0, Math.floor((signal.length - nFft) / hopLength) + 1);
  const frames: Float64Array[] = [];
  for (let f = 0; f < nFrames; f++) {
    const start = f * hopLength;
    const frame = signal.slice(start, start + nFft);
    if (useHann) hannWindow(frame);
    frames.push(magnitudeSpectrum(frame));
  }
  return frames;
}

// ─── Mel filterbank ──────────────────────────────────────────────────────────

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

export interface MelFilterbankOptions {
  nMels?: number;
  sampleRate?: number;
  nFft?: number;
  fMin?: number;
  fMax?: number;
}

/**
 * Build a triangular Mel filterbank matrix.
 *
 * @returns Matrix [nMels x (nFft/2 + 1)] of filter weights.
 */
export function melFilterbank(options: MelFilterbankOptions = {}): Float64Array[] {
  const nMels = options.nMels ?? 40;
  const sampleRate = options.sampleRate ?? 22050;
  const nFft = options.nFft ?? 512;
  const fMin = options.fMin ?? 0;
  const fMax = options.fMax ?? sampleRate / 2;

  const nBins = Math.floor(nFft / 2) + 1;
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);

  // Centre frequencies of each mel filter + edges
  const melPoints = Float64Array.from(
    { length: nMels + 2 },
    (_, i) => melMin + (i / (nMels + 1)) * (melMax - melMin),
  );
  const hzPoints = Float64Array.from(melPoints, (m) => melToHz(m));

  // Map Hz centres to FFT bin indices
  const binFreqs = Float64Array.from({ length: nBins }, (_, k) => (k * sampleRate) / nFft);

  return Array.from({ length: nMels }, (_, m) => {
    const lo = hzPoints[m]! ?? fMin;
    const mid = hzPoints[m + 1]! ?? lo;
    const hi = hzPoints[m + 2]! ?? mid;
    const filter = new Float64Array(nBins);
    for (let k = 0; k < nBins; k++) {
      const f = binFreqs[k]! ?? 0;
      if (f >= lo && f <= mid) {
        filter[k]! = (f - lo) / Math.max(mid - lo, 1e-10);
      } else if (f > mid && f <= hi) {
        filter[k]! = (hi - f) / Math.max(hi - mid, 1e-10);
      }
    }
    return filter;
  });
}

// ─── Mel spectrogram ─────────────────────────────────────────────────────────

/**
 * Compute a mel-spectrogram from a raw audio signal.
 *
 * @returns [nFrames x nMels] matrix of mel energies (linear scale).
 */
export function melSpectrogram(
  signal: Float64Array,
  options: STFTOptions & MelFilterbankOptions = {},
): Float64Array[] {
  const specFrames = stftMagnitude(signal, options);
  const fbOptions: MelFilterbankOptions = { nFft: options.nFft ?? 512 };
  if (options.nMels !== undefined) fbOptions.nMels = options.nMels;
  if (options.sampleRate !== undefined) fbOptions.sampleRate = options.sampleRate;
  if (options.fMin !== undefined) fbOptions.fMin = options.fMin;
  if (options.fMax !== undefined) fbOptions.fMax = options.fMax;
  const filterbank = melFilterbank(fbOptions);
  const nMels = filterbank.length;

  return specFrames.map((frame) => {
    const melEnergies = new Float64Array(nMels);
    for (let m = 0; m < nMels; m++) {
      let energy = 0;
      for (let k = 0; k < frame.length; k++) {
        energy += (frame[k]! ?? 0) * (filterbank[m]![k]! ?? 0);
      }
      melEnergies[m]! = energy;
    }
    return melEnergies;
  });
}

// ─── MFCCs ───────────────────────────────────────────────────────────────────

/**
 * Discrete Cosine Transform (DCT-II) of a sequence.
 * Used to decorrelate mel filterbank energies into MFCCs.
 */
function dct2(x: Float64Array): Float64Array {
  const n = x.length;
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      s += (x[i]! ?? 0) * Math.cos((Math.PI / n) * (i + 0.5) * k);
    }
    out[k]! = s;
  }
  return out;
}

export interface MFCCOptions extends STFTOptions, MelFilterbankOptions {
  nMfcc?: number;
}

/**
 * Compute Mel-Frequency Cepstral Coefficients (MFCCs) from a signal.
 *
 * @param signal - 1-D audio signal.
 * @param options - Configuration for FFT, mel filterbank, and nMfcc.
 * @returns [nFrames x nMfcc] matrix of MFCC coefficients.
 */
export function mfcc(
  signal: Float64Array,
  options: MFCCOptions = {},
): Float64Array[] {
  const nMfcc = options.nMfcc ?? 13;
  const melFrames = melSpectrogram(signal, options);

  return melFrames.map((frame) => {
    // Log compression
    const logMel = Float64Array.from(frame, (v) => Math.log(Math.max(v, 1e-10)));
    // DCT-II to decorrelate
    const ceps = dct2(logMel);
    return ceps.slice(0, nMfcc);
  });
}

// ─── Frame-level features ────────────────────────────────────────────────────

/**
 * Root-mean-square (RMS) energy per frame.
 *
 * @param signal - 1-D audio signal.
 * @param frameLength - Frame length in samples.
 * @param hopLength - Hop length in samples.
 * @returns RMS energy value per frame.
 */
export function rmsEnergy(
  signal: Float64Array,
  frameLength = 512,
  hopLength = 128,
): Float64Array {
  const nFrames = Math.max(0, Math.floor((signal.length - frameLength) / hopLength) + 1);
  return Float64Array.from({ length: nFrames }, (_, f) => {
    let sumSq = 0;
    const start = f * hopLength;
    for (let i = start; i < start + frameLength && i < signal.length; i++) {
      sumSq += (signal[i]! ?? 0) ** 2;
    }
    return Math.sqrt(sumSq / frameLength);
  });
}

/**
 * Zero-crossing rate per frame.
 *
 * @param signal - 1-D audio signal.
 * @param frameLength - Frame length in samples.
 * @param hopLength - Hop length in samples.
 * @returns Zero-crossing rate (count/frame) per frame.
 */
export function zeroCrossingRate(
  signal: Float64Array,
  frameLength = 512,
  hopLength = 128,
): Float64Array {
  const nFrames = Math.max(0, Math.floor((signal.length - frameLength) / hopLength) + 1);
  return Float64Array.from({ length: nFrames }, (_, f) => {
    let crossings = 0;
    const start = f * hopLength;
    for (let i = start + 1; i < start + frameLength && i < signal.length; i++) {
      if (((signal[i]! ?? 0) >= 0) !== ((signal[i - 1]! ?? 0) >= 0)) crossings++;
    }
    return crossings / frameLength;
  });
}
