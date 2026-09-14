import type { Candidate, TraceTensor } from "./modelTrace.ts";

export function assertEquivalent(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance = 1e-5): void {
  if (actual.length !== expected.length) throw new Error("The recorded tensor has an unexpected shape.");
  for (let index = 0; index < actual.length; index += 1) {
    if (!Number.isFinite(actual[index]) || !Number.isFinite(expected[index]) || Math.abs(actual[index] - expected[index]) > tolerance) {
      throw new Error("The recorded computation did not match the model output.");
    }
  }
}

export function validateTensor(tensor: TraceTensor): void {
  if (tensor.shape.reduce((size, dimension) => size * dimension, 1) !== tensor.values.length) {
    throw new Error(`The recorded shape of ${tensor.name} does not match its values.`);
  }
  for (const value of tensor.values) if (!Number.isFinite(value)) throw new Error(`The model returned a non-finite value in ${tensor.name}.`);
}

export function candidates(values: ArrayLike<number>, decode: (id: number) => string, logits = false, count = 8): Candidate[] {
  let maximum = -Infinity;
  for (let index = 0; index < values.length; index += 1) maximum = Math.max(maximum, values[index]);
  let total = 0;
  for (let index = 0; index < values.length; index += 1) total += logits ? Math.exp(values[index] - maximum) : values[index];
  if (!Number.isFinite(total) || total <= 0 || (!logits && Math.abs(total - 1) > 1e-4)) throw new Error("The model returned an invalid probability distribution.");
  const best: number[] = [];
  for (let index = 0; index < values.length; index += 1) {
    if (!logits && values[index] < 0) throw new Error("The model returned a negative probability.");
    const position = best.findIndex(other => values[index] > values[other]);
    if (position >= 0) best.splice(position, 0, index);
    else if (best.length < count) best.push(index);
    if (best.length > count) best.pop();
  }
  return best.map((id, rank) => ({ token: decode(id), id, probability: logits ? Math.exp(values[id] - maximum) / total : values[id], ...(logits ? { logit: values[id] } : {}), selected: rank === 0 }));
}
