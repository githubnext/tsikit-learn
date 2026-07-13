/**
 * ConvolutionalMLP and Conv1DExtractor — 1D convolutional neural network port.
 */

export class Conv1DLayer {
  filters: number;
  kernelSize: number;
  stride: number;
  padding: "valid" | "same";
  activation: "relu" | "tanh" | "linear";
  kernels: Float64Array[] | null = null;
  biases: Float64Array | null = null;

  constructor(
    filters = 32,
    kernelSize = 3,
    stride = 1,
    padding: "valid" | "same" = "valid",
    activation: "relu" | "tanh" | "linear" = "relu",
  ) {
    this.filters = filters;
    this.kernelSize = kernelSize;
    this.stride = stride;
    this.padding = padding;
    this.activation = activation;
    this._initWeights(1);
  }

  private _initWeights(inChannels: number): void {
    const scale = Math.sqrt(2 / (this.kernelSize * inChannels));
    this.kernels = Array.from({ length: this.filters }, () =>
      Float64Array.from({ length: this.kernelSize * inChannels }, () => (Math.random() - 0.5) * 2 * scale)
    );
    this.biases = new Float64Array(this.filters);
  }

  initForInput(inChannels: number): void {
    this._initWeights(inChannels);
  }

  forward(input: Float64Array[]): Float64Array[] {
    const seqLen = input.length;
    const inChannels = input[0]?.length ?? 0;
    if (!this.kernels || inChannels !== this.kernels[0]!.length / this.kernelSize) {
      this._initWeights(inChannels);
    }
    const outLen = this.padding === "valid"
      ? Math.floor((seqLen - this.kernelSize) / this.stride) + 1
      : Math.ceil(seqLen / this.stride);

    const output: Float64Array[] = [];
    for (let t = 0; t < outLen; t++) {
      const tStart = t * this.stride - (this.padding === "same" ? Math.floor((this.kernelSize - 1) / 2) : 0);
      const out = new Float64Array(this.filters);
      for (let f = 0; f < this.filters; f++) {
        let s = this.biases?.[f] ?? 0;
        for (let k = 0; k < this.kernelSize; k++) {
          const tIdx = tStart + k;
          if (tIdx < 0 || tIdx >= seqLen) continue;
          for (let c = 0; c < inChannels; c++) {
            s += (input[tIdx]?.[c] ?? 0) * (this.kernels?.[f]?.[k * inChannels + c] ?? 0);
          }
        }
        out[f] = this._activate(s);
      }
      output.push(out);
    }
    return output;
  }

  private _activate(x: number): number {
    if (this.activation === "relu") return Math.max(0, x);
    if (this.activation === "tanh") return Math.tanh(x);
    return x;
  }
}

export class GlobalAveragePooling1D {
  forward(input: Float64Array[]): Float64Array {
    if (input.length === 0) return new Float64Array(0);
    const features = input[0]?.length ?? 0;
    const out = new Float64Array(features);
    for (const vec of input) {
      for (let j = 0; j < features; j++) out[j] = (out[j] ?? 0) + (vec[j] ?? 0) / input.length;
    }
    return out;
  }
}

export class GlobalMaxPooling1D {
  forward(input: Float64Array[]): Float64Array {
    if (input.length === 0) return new Float64Array(0);
    const features = input[0]?.length ?? 0;
    const out = Float64Array.from({ length: features }, () => Number.NEGATIVE_INFINITY);
    for (const vec of input) {
      for (let j = 0; j < features; j++) {
        if ((vec[j] ?? Number.NEGATIVE_INFINITY) > (out[j] ?? Number.NEGATIVE_INFINITY)) out[j] = vec[j] ?? 0;
      }
    }
    return out;
  }
}

export class ConvolutionalMLP {
  private convLayers: Conv1DLayer[];
  private denseWeights: Float64Array[] | null = null;
  private denseBias: Float64Array | null = null;
  nOutputs: number;
  learningRate: number;
  nEpochs: number;

  constructor(
    convConfigs: Array<{ filters: number; kernelSize: number; activation?: "relu" | "tanh" | "linear" }> = [{ filters: 32, kernelSize: 3 }],
    nOutputs = 1,
    learningRate = 0.01,
    nEpochs = 10,
  ) {
    this.convLayers = convConfigs.map((c) => new Conv1DLayer(c.filters, c.kernelSize, 1, "valid", c.activation ?? "relu"));
    this.nOutputs = nOutputs;
    this.learningRate = learningRate;
    this.nEpochs = nEpochs;
  }

  private _forward(x: Float64Array[]): Float64Array {
    let h: Float64Array[] = x;
    for (const layer of this.convLayers) h = layer.forward(h);
    const pooled = new GlobalAveragePooling1D().forward(h);

    if (!this.denseWeights) {
      const inDim = pooled.length;
      const scale = Math.sqrt(2 / inDim);
      this.denseWeights = Array.from({ length: inDim }, () =>
        Float64Array.from({ length: this.nOutputs }, () => (Math.random() - 0.5) * 2 * scale)
      );
      this.denseBias = new Float64Array(this.nOutputs);
    }

    const out = new Float64Array(this.nOutputs);
    for (let j = 0; j < this.nOutputs; j++) out[j] = this.denseBias?.[j] ?? 0;
    for (let i = 0; i < pooled.length; i++) {
      for (let j = 0; j < this.nOutputs; j++) {
        out[j] = (out[j] ?? 0) + (pooled[i] ?? 0) * (this.denseWeights?.[i]?.[j] ?? 0);
      }
    }
    return out;
  }

  fit(sequences: Float64Array[][], y: Float64Array): this {
    // Initialize by running one forward pass
    if (sequences[0]) this._forward(sequences[0] as Float64Array[]);

    for (let epoch = 0; epoch < this.nEpochs; epoch++) {
      for (let i = 0; i < sequences.length; i++) {
        const out = this._forward(sequences[i] as Float64Array[]);
        // Simple MSE gradient update for dense layer only (simplified backprop)
        const yTrue = y[i] ?? 0;
        const pred = out[0] ?? 0;
        const grad = 2 * (pred - yTrue) / sequences.length;
        // Update dense bias
        if (this.denseBias) this.denseBias[0] = (this.denseBias[0] ?? 0) - this.learningRate * grad;
      }
    }
    return this;
  }

  predict(sequences: Float64Array[][]): Float64Array {
    return new Float64Array(sequences.map((seq) => this._forward(seq as Float64Array[])[0] ?? 0));
  }
}

export class Conv1DFeatureExtractor {
  private layer: Conv1DLayer;
  private pooling: GlobalAveragePooling1D;
  features_: number = 0;

  constructor(filters = 64, kernelSize = 3, activation: "relu" | "tanh" = "relu") {
    this.layer = new Conv1DLayer(filters, kernelSize, 1, "valid", activation);
    this.pooling = new GlobalAveragePooling1D();
  }

  fitTransform(sequences: Float64Array[][]): Float64Array[] {
    this.features_ = this.layer.filters;
    return sequences.map((seq) => this.pooling.forward(this.layer.forward(seq as Float64Array[])));
  }

  transform(sequences: Float64Array[][]): Float64Array[] {
    return sequences.map((seq) => this.pooling.forward(this.layer.forward(seq as Float64Array[])));
  }
}
