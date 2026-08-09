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
  makeBiclusters,
  makeLowRankMatrix,
  makeSparseCodedSignal,
} from "./generator_ext.js";
export type {
  BiclustersOptions,
  BiclustersResult,
  CheckerboardOptions,
  CheckerboardResult,
  LowRankMatrixOptions,
  LowRankMatrixResult,
  SparseCodingOptions,
  SparseCodingResult,
} from "./generator_ext.js";
export {
  fetchCaliforniaHousing,
  fetchCovtype,
  fetchKddcup99,
  fetchLfw,
  fetchOlivettiFaces,
} from "./fetch_datasets.js";
export type { FetchedDataset } from "./fetch_datasets.js";
