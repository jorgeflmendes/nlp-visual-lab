import test from "node:test";
import assert from "node:assert/strict";
import { groupExecutionSteps } from "../src/modelTrace.ts";

test("groupExecutionSteps groups contiguous steps by stage", () => {
  const steps = [
    { name: "Input 1", stage: "Input", operation: "lookup", description: "", tensors: [] },
    { name: "Input 2", stage: "Input", operation: "lookup", description: "", tensors: [] },
    { name: "Embed 1", stage: "Embedding", operation: "embed", description: "", tensors: [] },
    { name: "Decode 1", stage: "Decoder", operation: "lstm", description: "", tensors: [] },
    { name: "Decode 2", stage: "Decoder", operation: "lstm", description: "", tensors: [] },
    { name: "Output 1", stage: "Output", operation: "dense", description: "", tensors: [] },
  ];

  const groups = groupExecutionSteps(steps);
  assert.equal(groups.length, 4);

  assert.equal(groups[0].stage, "Input");
  assert.equal(groups[0].startIndex, 0);
  assert.equal(groups[0].endIndex, 1);
  assert.equal(groups[0].steps.length, 2);
  assert.equal(groups[0].steps[0].index, 0);
  assert.equal(groups[0].steps[1].index, 1);

  assert.equal(groups[1].stage, "Embedding");
  assert.equal(groups[1].startIndex, 2);
  assert.equal(groups[1].endIndex, 2);

  assert.equal(groups[2].stage, "Decoder");
  assert.equal(groups[2].startIndex, 3);
  assert.equal(groups[2].endIndex, 4);

  assert.equal(groups[3].stage, "Output");
  assert.equal(groups[3].startIndex, 5);
  assert.equal(groups[3].endIndex, 5);
});

test("groupExecutionSteps handles empty steps gracefully", () => {
  assert.deepEqual(groupExecutionSteps([]), []);
});

test("groupExecutionSteps separates non-contiguous steps of the same stage", () => {
  const steps = [
    { name: "Step A", stage: "Decoder", operation: "lstm", description: "", tensors: [] },
    { name: "Step B", stage: "Attention", operation: "dot", description: "", tensors: [] },
    { name: "Step C", stage: "Decoder", operation: "lstm", description: "", tensors: [] },
  ];
  const groups = groupExecutionSteps(steps);
  assert.equal(groups.length, 3);
  assert.equal(groups[0].stage, "Decoder");
  assert.equal(groups[1].stage, "Attention");
  assert.equal(groups[2].stage, "Decoder");
});
