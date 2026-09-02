export declare const Filter: Readonly<{ Box: 0; Lanczos3: 1; Mitchell: 2 }>;
export type FilterId = 0 | 1 | 2;
export declare function resample(src: Float32Array, sw: number, sh: number, ch: number, dw: number, dh: number, filter?: FilterId): Promise<Float32Array>;
export declare function srgbToLinear(u8: Uint8Array): Promise<Float32Array>;
export declare function linearToSrgb(f32: Float32Array): Promise<Uint8Array>;
export declare function resize(u8: Uint8Array, sw: number, sh: number, ch: number, dw: number, dh: number, filter?: FilterId): Promise<Uint8Array>;
