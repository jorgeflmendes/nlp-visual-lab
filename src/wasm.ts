interface CoreExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  alloc_u32(length: number): number;
  dealloc_u32(pointer: number, length: number): void;
  edit_distance(leftPointer: number, leftLength: number, rightPointer: number, rightLength: number): number;
}

let exportsPromise: Promise<CoreExports> | undefined;

async function loadCore(): Promise<CoreExports> {
  exportsPromise ??= fetch(`${import.meta.env.BASE_URL}wasm/nlp_core.wasm`)
    .then((response) => {
      if (!response.ok) throw new Error(`WASM ${response.status}`);
      return WebAssembly.instantiateStreaming(response, {});
    })
    .then(({ instance }) => instance.exports as CoreExports);
  return exportsPromise;
}

function writeU32(core: CoreExports, values: number[]): number {
  const pointer = core.alloc_u32(values.length);
  new Uint32Array(core.memory.buffer, pointer, values.length).set(values);
  return pointer;
}

export async function wasmEditDistance(left: string, right: string): Promise<number> {
  const core = await loadCore();
  const a = Array.from(left, (character) => character.codePointAt(0) ?? 0);
  const b = Array.from(right, (character) => character.codePointAt(0) ?? 0);
  const leftPointer = writeU32(core, a);
  const rightPointer = writeU32(core, b);
  try {
    return core.edit_distance(leftPointer, a.length, rightPointer, b.length);
  } finally {
    core.dealloc_u32(leftPointer, a.length);
    core.dealloc_u32(rightPointer, b.length);
  }
}
