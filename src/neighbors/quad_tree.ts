/**
 * QuadTree for 2D space partitioning (used by t-SNE, neighbors).
 * Port of sklearn.neighbors._quad_tree
 */

export interface Point2D {
  x: number;
  y: number;
}

interface QuadTreeNode {
  bounds: { x: number; y: number; w: number; h: number };
  point: Point2D | null;
  children: (QuadTreeNode | null)[];
  count: number;
  centerOfMass: Point2D;
}

/**
 * QuadTree for efficient N-body force approximation (Barnes-Hut).
 * Used in t-SNE for gradient computation.
 */
export class QuadTree {
  private root: QuadTreeNode | null = null;
  private capacity: number;
  theta: number;

  constructor(theta = 0.5, capacity = 1) {
    this.theta = theta;
    this.capacity = capacity;
  }

  private makeNode(x: number, y: number, w: number, h: number): QuadTreeNode {
    return {
      bounds: { x, y, w, h },
      point: null,
      children: [null, null, null, null],
      count: 0,
      centerOfMass: { x: 0, y: 0 },
    };
  }

  build(points: Float64Array[]): void {
    if (points.length === 0) return;
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (const p of points) {
      const px = p[0] ?? 0;
      const py = p[1] ?? 0;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }
    const margin = 1e-6;
    const size = Math.max(maxX - minX, maxY - minY) + margin;
    this.root = this.makeNode(minX - margin / 2, minY - margin / 2, size, size);
    for (const p of points) {
      this._insert(this.root, { x: p[0] ?? 0, y: p[1] ?? 0 });
    }
  }

  private _insert(node: QuadTreeNode, point: Point2D): void {
    node.count++;
    node.centerOfMass.x =
      (node.centerOfMass.x * (node.count - 1) + point.x) / node.count;
    node.centerOfMass.y =
      (node.centerOfMass.y * (node.count - 1) + point.y) / node.count;

    if (node.count === 1 && node.children.every((c) => c === null)) {
      node.point = point;
      return;
    }

    if (node.point !== null && node.children.every((c) => c === null)) {
      // Split: move existing point to child
      this._insertIntoChild(node, node.point);
      node.point = null;
    }
    this._insertIntoChild(node, point);
  }

  private _insertIntoChild(node: QuadTreeNode, point: Point2D): void {
    const { x, y, w, h } = node.bounds;
    const hw = w / 2;
    const hh = h / 2;
    const qx = point.x >= x + hw ? 1 : 0;
    const qy = point.y >= y + hh ? 1 : 0;
    const qi = qy * 2 + qx;
    if (!node.children[qi]) {
      node.children[qi] = this.makeNode(x + qx * hw, y + qy * hh, hw, hh);
    }
    this._insert(node.children[qi]!, point);
  }

  /**
   * Compute Barnes-Hut force on a given point.
   * Returns [fx, fy, nTerms]
   */
  computeForce(point: Point2D, _zeta = 0.5): [number, number, number] {
    if (!this.root) return [0, 0, 0];
    return this._computeForceNode(this.root, point);
  }

  private _computeForceNode(
    node: QuadTreeNode,
    point: Point2D,
  ): [number, number, number] {
    if (node.count === 0) return [0, 0, 0];
    const dx = node.centerOfMass.x - point.x;
    const dy = node.centerOfMass.y - point.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 1e-10) return [0, 0, 0];

    // Barnes-Hut criterion
    const size = node.bounds.w;
    if (size / dist < this.theta || node.children.every((c) => c === null)) {
      // Treat as single body
      const distSq = dx * dx + dy * dy + 1;
      const force = node.count / (distSq * Math.sqrt(distSq));
      return [dx * force, dy * force, 1];
    }

    // Recurse into children
    let fx = 0;
    let fy = 0;
    let nTerms = 0;
    for (const child of node.children) {
      if (child) {
        const [cfx, cfy, ct] = this._computeForceNode(child, point);
        fx += cfx;
        fy += cfy;
        nTerms += ct;
      }
    }
    return [fx, fy, nTerms];
  }

  /** Get all points in the tree */
  getPoints(): Point2D[] {
    const points: Point2D[] = [];
    if (this.root) this._collectPoints(this.root, points);
    return points;
  }

  private _collectPoints(node: QuadTreeNode, points: Point2D[]): void {
    if (node.point) points.push(node.point);
    for (const child of node.children) {
      if (child) this._collectPoints(child, points);
    }
  }
}

/** OcTree for 3D space (extension of QuadTree to 3D) */
export class OcTree {
  theta: number;
  points: Float64Array[] = [];

  constructor(theta = 0.5) {
    this.theta = theta;
  }

  build(points: Float64Array[]): void {
    this.points = points;
  }

  computeForce(point: Float64Array): Float64Array {
    const d = point.length;
    const force = new Float64Array(d);
    for (const p of this.points) {
      let distSq = 0;
      for (let j = 0; j < d; j++)
        distSq += ((p[j] ?? 0) - (point[j] ?? 0)) ** 2;
      distSq += 1;
      const f = 1 / (distSq * Math.sqrt(distSq));
      for (let j = 0; j < d; j++)
        force[j] += ((p[j] ?? 0) - (point[j] ?? 0)) * f;
    }
    return force;
  }
}
