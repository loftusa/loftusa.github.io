export declare const SPRING_DEFAULT: number;
export declare const HOME_DEFAULT: number;
export type SimNode = { x: number; y: number; x0: number; y0: number; pinned?: boolean };
export function relax(
  nodes: SimNode[],
  links: Array<[number, number]>,
  opts?: { iterations?: number; spring?: number; home?: number },
): void;
export function settled(nodes: SimNode[], eps?: number): boolean;
