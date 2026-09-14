import assert from "node:assert/strict";
import test from "node:test";
import { assertEquivalent, candidates, validateTensor } from "../src/modelNumerics.ts";

test("logit candidates use the complete vocabulary denominator", () => {
  const result = candidates([1001, 1000, 999], String, true, 2);
  assert.equal(result.length, 2);
  assert.equal(result[0].id, 0);
  assert.equal(result[0].logit, 1001);
  assert.ok(Math.abs(result[0].probability - 1 / (1 + Math.exp(-1) + Math.exp(-2))) < 1e-12);
  assert.ok(result.reduce((sum, row) => sum + row.probability, 0) < 1);
});

test("probability candidates preserve stored values and greedy ties", () => {
  const result = candidates([0.4, 0.4, 0.2], String);
  assert.deepEqual(result.map(row => row.probability), [0.4, 0.4, 0.2]);
  assert.equal(result[0].selected, true);
  assert.equal(result[1].selected, false);
  assert.equal(result[0].id, 0);
});

test("invalid distributions never become plausible candidates", () => {
  for (const values of [[0.8, 0.8], [-0.1, 1.1], [NaN, 1], [Infinity, 0], []]) {
    assert.throws(() => candidates(values, String));
  }
});

test("trace verification rejects mismatched shapes and non-finite states", () => {
  validateTensor({ name: "state", shape: [1, 2], values: [0, 1] });
  assert.throws(() => validateTensor({ name: "state", shape: [2, 2], values: [0, 1] }));
  assert.throws(() => validateTensor({ name: "state", shape: [1], values: [NaN] }));
  assertEquivalent([0.1], [0.1 + 1e-7]);
  assert.throws(() => assertEquivalent([0.1], [0.2]));
  assert.throws(() => assertEquivalent([0.1], [NaN]));
  assert.throws(() => assertEquivalent([0.1], []));
});
