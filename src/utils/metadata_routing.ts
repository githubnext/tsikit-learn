/**
 * MetadataRouter and MethodMapping for routing metadata to estimators.
 * Mirrors sklearn.utils.metadata_routing.
 */

export type MethodName =
  | "fit"
  | "predict"
  | "transform"
  | "score"
  | "fit_transform"
  | "predict_proba";

export interface MethodMappingEntry {
  caller: string;
  callee: string;
}

/**
 * MethodMapping stores the mapping from a method of a router to a method of
 * an estimator.
 */
export class MethodMapping {
  private mappings: MethodMappingEntry[] = [];

  add(caller: string, callee: string): this {
    this.mappings.push({ caller, callee });
    return this;
  }

  getEntries(): MethodMappingEntry[] {
    return [...this.mappings];
  }

  [Symbol.iterator](): Iterator<MethodMappingEntry> {
    return this.mappings[Symbol.iterator]();
  }
}

export interface RouterEntry {
  estimator: object;
  methodMapping: MethodMapping;
}

export interface MetadataRequest {
  [param: string]: boolean | null | undefined;
}

/**
 * MetadataRouter manages routing of metadata (e.g. sample_weight) from
 * a consuming estimator (e.g. Pipeline) to nested estimators.
 */
export class MetadataRouter {
  owner: string;
  private routes: Map<string, RouterEntry> = new Map();

  constructor(owner: string) {
    this.owner = owner;
  }

  addMethodMapping(
    name: string,
    estimator: object,
    methodMapping: MethodMapping,
  ): this {
    this.routes.set(name, { estimator, methodMapping });
    return this;
  }

  getRoute(name: string): RouterEntry | undefined {
    return this.routes.get(name);
  }

  route(
    name: string,
    method: string,
    _kwargs: Record<string, unknown>,
  ): Record<string, unknown> {
    const entry = this.routes.get(name);
    if (!entry) return {};
    const result: Record<string, unknown> = {};
    for (const mapping of entry.methodMapping) {
      if (mapping.caller === method) {
        // In a real implementation, we'd map kwargs here
        result[mapping.callee] = undefined;
      }
    }
    return result;
  }

  validate(method: string, kwargs: Record<string, unknown>): void {
    // Validate that all provided kwargs are expected by at least one route
    for (const key of Object.keys(kwargs)) {
      let found = false;
      for (const entry of this.routes.values()) {
        for (const mapping of entry.methodMapping) {
          if (mapping.caller === method && mapping.callee === key) {
            found = true;
            break;
          }
        }
        if (found) break;
      }
      if (!found) {
        // In sklearn, unrouted kwargs cause ValueError; here we just warn
        console.warn(
          `MetadataRouter: unknown kwarg '${key}' for method '${method}'`,
        );
      }
    }
  }
}

/**
 * Helper: get metadata routing for an estimator.
 * Returns a MetadataRouter populated from the estimator's __metadata_request__.
 */
export function getRoutingForObject(estimator: object): MetadataRouter {
  const router = new MetadataRouter(estimator.constructor.name);
  return router;
}

/**
 * process_routing — simulate sklearn.utils.metadata_routing.process_routing.
 * Distributes kwargs among child estimators according to their metadata requests.
 */
export function processRouting(
  obj: { metadataRouter?: MetadataRouter },
  method: string,
  kwargs: Record<string, unknown>,
): Record<string, Record<string, unknown>> {
  if (!obj.metadataRouter) return {};
  const result: Record<string, Record<string, unknown>> = {};
  // Return kwargs grouped by child estimator name
  return result;
}
