/**
 * Neural network activation functions.
 * Port of sklearn.neural_network._base (activation functions)
 */

/** Sigmoid (logistic) activation */
export function sigmoid(z: Float64Array): Float64Array {
	const result = new Float64Array(z.length);
	for (let i = 0; i < z.length; i++) result[i] = 1 / (1 + Math.exp(-(z[i] ?? 0)));
	return result;
}

/** ReLU activation */
export function relu(z: Float64Array): Float64Array {
	const result = new Float64Array(z.length);
	for (let i = 0; i < z.length; i++) result[i] = Math.max(0, z[i] ?? 0);
	return result;
}

/** Leaky ReLU activation */
export function leakyRelu(z: Float64Array, negativeSlope = 0.01): Float64Array {
	const result = new Float64Array(z.length);
	for (let i = 0; i < z.length; i++) {
		const zi = z[i] ?? 0;
		result[i] = zi >= 0 ? zi : negativeSlope * zi;
	}
	return result;
}

/** ELU activation */
export function elu(z: Float64Array, alpha = 1.0): Float64Array {
	const result = new Float64Array(z.length);
	for (let i = 0; i < z.length; i++) {
		const zi = z[i] ?? 0;
		result[i] = zi >= 0 ? zi : alpha * (Math.exp(zi) - 1);
	}
	return result;
}

/** SELU activation */
export function selu(z: Float64Array): Float64Array {
	const scale = 1.0507009873554804934193349852946;
	const alpha = 1.6732632423543772848170429916717;
	const result = new Float64Array(z.length);
	for (let i = 0; i < z.length; i++) {
		const zi = z[i] ?? 0;
		result[i] = scale * (zi >= 0 ? zi : alpha * (Math.exp(zi) - 1));
	}
	return result;
}

/** Tanh activation */
export function tanh(z: Float64Array): Float64Array {
	const result = new Float64Array(z.length);
	for (let i = 0; i < z.length; i++) result[i] = Math.tanh(z[i] ?? 0);
	return result;
}

/** Softmax activation */
export function softmax(z: Float64Array): Float64Array {
	const result = new Float64Array(z.length);
	const maxZ = Math.max(...z);
	let sum = 0;
	for (let i = 0; i < z.length; i++) { result[i] = Math.exp((z[i] ?? 0) - maxZ); sum += result[i]!; }
	for (let i = 0; i < z.length; i++) result[i]! /= sum;
	return result;
}

/** Softplus activation */
export function softplus(z: Float64Array): Float64Array {
	const result = new Float64Array(z.length);
	for (let i = 0; i < z.length; i++) result[i] = Math.log(1 + Math.exp(z[i] ?? 0));
	return result;
}

/** Swish activation (x * sigmoid(x)) */
export function swish(z: Float64Array): Float64Array {
	const result = new Float64Array(z.length);
	for (let i = 0; i < z.length; i++) {
		const zi = z[i] ?? 0;
		result[i] = zi / (1 + Math.exp(-zi));
	}
	return result;
}

/** GELU activation (Gaussian Error Linear Unit) */
export function gelu(z: Float64Array): Float64Array {
	const result = new Float64Array(z.length);
	const sqrt2 = Math.sqrt(2);
	for (let i = 0; i < z.length; i++) {
		const zi = z[i] ?? 0;
		// Approximation: 0.5 * x * (1 + erf(x/sqrt(2)))
		const erf = 2 / (1 + Math.exp(-2 * zi * sqrt2)) - 1;
		result[i] = 0.5 * zi * (1 + erf);
	}
	return result;
}

/** Derivative of sigmoid */
export function sigmoidDerivative(activated: Float64Array): Float64Array {
	const result = new Float64Array(activated.length);
	for (let i = 0; i < activated.length; i++) {
		const a = activated[i] ?? 0;
		result[i] = a * (1 - a);
	}
	return result;
}

/** Derivative of tanh */
export function tanhDerivative(activated: Float64Array): Float64Array {
	const result = new Float64Array(activated.length);
	for (let i = 0; i < activated.length; i++) {
		const a = activated[i] ?? 0;
		result[i] = 1 - a * a;
	}
	return result;
}

/** Derivative of ReLU */
export function reluDerivative(z: Float64Array): Float64Array {
	const result = new Float64Array(z.length);
	for (let i = 0; i < z.length; i++) result[i] = (z[i] ?? 0) > 0 ? 1 : 0;
	return result;
}

/** Identity activation */
export function identity(z: Float64Array): Float64Array {
	return z.slice();
}

/** Activation function registry */
export type ActivationName = "sigmoid" | "tanh" | "relu" | "leaky_relu" | "elu" | "selu" | "softmax" | "softplus" | "swish" | "gelu" | "identity";

export function getActivation(name: ActivationName): (z: Float64Array) => Float64Array {
	switch (name) {
		case "sigmoid": return sigmoid;
		case "tanh": return tanh;
		case "relu": return relu;
		case "leaky_relu": return leakyRelu;
		case "elu": return elu;
		case "selu": return selu;
		case "softmax": return softmax;
		case "softplus": return softplus;
		case "swish": return swish;
		case "gelu": return gelu;
		case "identity": return identity;
	}
}

export function getActivationDerivative(name: ActivationName): ((z: Float64Array) => Float64Array) | null {
	switch (name) {
		case "sigmoid": return sigmoidDerivative;
		case "tanh": return tanhDerivative;
		case "relu": return reluDerivative;
		default: return null;
	}
}
