/**
 * Audio feature extraction extensions.
 * Port of sklearn.feature_extraction._audio
 */

/**
 * Compute MFCC (Mel-frequency cepstral coefficients) from a signal.
 * Simplified implementation for feature extraction.
 */
export function mfcc(
	signal: Float64Array,
	sampleRate: number,
	nMfcc = 13,
	nFft = 512,
	hopLength = 256,
	nMels = 40,
	fMin = 0,
	fMax: number | null = null,
): Float64Array[] {
	const fMaxHz = fMax ?? sampleRate / 2;
	const frames = framingSignal(signal, nFft, hopLength);
	const melFilters = melFilterbank(nFft, nMels, sampleRate, fMin, fMaxHz);

	return frames.map((frame) => {
		// FFT power spectrum
		const spectrum = powerSpectrum(applyWindow(frame, "hann"), nFft);

		// Mel spectrum
		const melSpectrum = new Float64Array(nMels);
		for (let m = 0; m < nMels; m++) {
			for (let k = 0; k < spectrum.length; k++) {
				melSpectrum[m]! += (melFilters[m]?.[k] ?? 0) * (spectrum[k] ?? 0);
			}
			melSpectrum[m] = Math.log(melSpectrum[m]! + 1e-10);
		}

		// DCT to get MFCCs
		const coeffs = new Float64Array(nMfcc);
		for (let n = 0; n < nMfcc; n++) {
			for (let m = 0; m < nMels; m++) {
				coeffs[n]! += melSpectrum[m]! * Math.cos(Math.PI * n * (m + 0.5) / nMels);
			}
			coeffs[n]! *= Math.sqrt(2 / nMels);
		}
		return coeffs;
	});
}

/** Frame a signal into overlapping segments */
export function framingSignal(
	signal: Float64Array,
	frameLength: number,
	hopLength: number,
): Float64Array[] {
	const frames: Float64Array[] = [];
	for (let start = 0; start + frameLength <= signal.length; start += hopLength) {
		const frame = new Float64Array(frameLength);
		for (let i = 0; i < frameLength; i++) frame[i] = signal[start + i] ?? 0;
		frames.push(frame);
	}
	if (frames.length === 0) {
		const frame = new Float64Array(frameLength);
		for (let i = 0; i < Math.min(frameLength, signal.length); i++) frame[i] = signal[i] ?? 0;
		frames.push(frame);
	}
	return frames;
}

/** Apply a window function to a frame */
export function applyWindow(frame: Float64Array, type: "hann" | "hamming" | "blackman" = "hann"): Float64Array {
	const n = frame.length;
	const result = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		let w: number;
		if (type === "hann") w = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
		else if (type === "hamming") w = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1));
		else w = 0.42 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)) + 0.08 * Math.cos(4 * Math.PI * i / (n - 1));
		result[i] = (frame[i] ?? 0) * w;
	}
	return result;
}

/** Simple DFT-based power spectrum (only positive frequencies) */
export function powerSpectrum(frame: Float64Array, nFft: number): Float64Array {
	const n = Math.min(frame.length, nFft);
	const halfN = Math.floor(nFft / 2) + 1;
	const power = new Float64Array(halfN);
	for (let k = 0; k < halfN; k++) {
		let re = 0;
		let im = 0;
		for (let i = 0; i < n; i++) {
			const angle = 2 * Math.PI * k * i / nFft;
			re += (frame[i] ?? 0) * Math.cos(angle);
			im -= (frame[i] ?? 0) * Math.sin(angle);
		}
		power[k] = (re * re + im * im) / nFft;
	}
	return power;
}

/** Build Mel filterbank */
export function melFilterbank(
	nFft: number,
	nMels: number,
	sampleRate: number,
	fMin: number,
	fMax: number,
): Float64Array[] {
	const freqToMel = (f: number) => 2595 * Math.log10(1 + f / 700);
	const melToFreq = (m: number) => 700 * (10 ** (m / 2595) - 1);

	const melMin = freqToMel(fMin);
	const melMax = freqToMel(fMax);
	const melPoints = Array.from({ length: nMels + 2 }, (_, i) => melMin + i * (melMax - melMin) / (nMels + 1));
	const freqPoints = melPoints.map(melToFreq);
	const binPoints = freqPoints.map((f) => Math.floor((nFft + 1) * f / sampleRate));

	const halfN = Math.floor(nFft / 2) + 1;
	const filters: Float64Array[] = Array.from({ length: nMels }, () => new Float64Array(halfN));

	for (let m = 1; m <= nMels; m++) {
		for (let k = 0; k < halfN; k++) {
			const left = binPoints[m - 1]!;
			const center = binPoints[m]!;
			const right = binPoints[m + 1]!;
			if (k >= left && k <= center) {
				filters[m - 1]![k] = (k - left) / (center - left + 1);
			} else if (k > center && k <= right) {
				filters[m - 1]![k] = (right - k) / (right - center + 1);
			}
		}
	}
	return filters;
}

/**
 * Zero-crossing rate of a signal.
 */
export function zeroCrossingRate(signal: Float64Array, frameLength: number, hopLength: number): Float64Array {
	const frames = framingSignal(signal, frameLength, hopLength);
	return new Float64Array(frames.map((frame) => {
		let zcr = 0;
		for (let i = 1; i < frame.length; i++) {
			if ((frame[i - 1]! >= 0) !== (frame[i]! >= 0)) zcr++;
		}
		return zcr / (frame.length - 1);
	}));
}

/**
 * Root Mean Square energy.
 */
export function rmsEnergy(signal: Float64Array, frameLength: number, hopLength: number): Float64Array {
	const frames = framingSignal(signal, frameLength, hopLength);
	return new Float64Array(frames.map((frame) => {
		const rms = Math.sqrt(frame.reduce((s, v) => s + v * v, 0) / frame.length);
		return rms;
	}));
}

/**
 * Spectral centroid.
 */
export function spectralCentroid(signal: Float64Array, sampleRate: number, nFft = 512, hopLength = 256): Float64Array {
	const frames = framingSignal(signal, nFft, hopLength);
	const halfN = Math.floor(nFft / 2) + 1;
	const freqs = Array.from({ length: halfN }, (_, k) => k * sampleRate / nFft);
	return new Float64Array(frames.map((frame) => {
		const windowed = applyWindow(frame, "hann");
		const spectrum = powerSpectrum(windowed, nFft);
		let weightedFreq = 0;
		let totalPower = 0;
		for (let k = 0; k < halfN; k++) {
			weightedFreq += freqs[k]! * (spectrum[k] ?? 0);
			totalPower += spectrum[k] ?? 0;
		}
		return weightedFreq / (totalPower + 1e-10);
	}));
}
