export * from "./make_datasets.js";
export * from "./load_datasets.js";
export * from "./svmlight.js";
export * from "./openml.js";
export * from "./samples_generator.js";
export * from "./rcv1.js";
export * from "./real_datasets.js";
export * from "./digits.js";
export * from "./newsgroups.js";
export {
  type LowRankMatrixOptions,
  type LowRankMatrixResult,
  makeLowRankMatrix,
  type SparseCodingOptions,
  type SparseCodingResult,
  makeSparseCodedSignal,
  type BiclustersOptions,
  type BiclustersResult,
  makeBiclusters,
  type CheckerboardOptions,
  type CheckerboardResult,
} from "./generator_ext.js";
export {
  type FetchedDataset,
  fetchCaliforniaHousing,
  fetchCovtype,
  fetchKddcup99,
  fetchLfw,
  fetchOlivettiFaces,
} from "./fetch_datasets.js";
