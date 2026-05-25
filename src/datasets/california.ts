/**
 * California Housing dataset utilities.
 * Port of sklearn.datasets._california_housing
 */

export interface CaliforniaHousingData {
	data: Float64Array[];
	target: Float64Array;
	featureNames: string[];
	targetNames: string[];
	description: string;
}

/**
 * Generate synthetic California housing-like data.
 * Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
 */
export function makeCaliforniaHousing(
	nSamples = 100,
	randomState = 42,
): CaliforniaHousingData {
	// Simple LCG random
	let seed = randomState;
	const rand = (): number => {
		seed = (seed * 1664525 + 1013904223) & 0xffffffff;
		return ((seed >>> 0) / 0x100000000);
	};
	const featureNames = [
		"MedInc", "HouseAge", "AveRooms", "AveBedrms",
		"Population", "AveOccup", "Latitude", "Longitude",
	];
	const data: Float64Array[] = [];
	const target = new Float64Array(nSamples);
	for (let i = 0; i < nSamples; i++) {
		const medInc = 0.5 + rand() * 10;
		const houseAge = 1 + rand() * 52;
		const aveRooms = 2 + rand() * 8;
		const aveBedrms = 0.5 + rand() * 2;
		const population = 100 + rand() * 3000;
		const aveOccup = 1 + rand() * 5;
		const latitude = 32 + rand() * 10;
		const longitude = -124 + rand() * 10;
		data.push(new Float64Array([medInc, houseAge, aveRooms, aveBedrms, population, aveOccup, latitude, longitude]));
		// Simplified price model
		target[i] = 0.5 + 0.4 * medInc - 0.001 * population + rand() * 0.5;
	}
	return {
		data,
		target,
		featureNames,
		targetNames: ["MedHouseVal"],
		description: "Synthetic California Housing dataset (generated). " +
			"Original from StatLib repository. 8 features, regression target is median house value.",
	};
}

export interface FetchCaliforniaHousingOptions {
	dataHome?: string;
	download?: boolean;
	returnXy?: boolean;
	asFrame?: boolean;
}

/**
 * Fetch (or generate) the California Housing dataset.
 * In browser/Bun environments, returns generated data.
 */
export function fetchCaliforniaHousing(
	opts: FetchCaliforniaHousingOptions = {},
): CaliforniaHousingData {
	void opts;
	return makeCaliforniaHousing(20640);
}
