import test from "node:test";
import assert from "node:assert/strict";
import {
  meanPooling,
  l2Normalize,
  cosineSimilaritySentence,
  buildSentenceMatrix,
  rankBySimilarity,
  projectSentencePCA,
} from "../src/sentenceEmbeddingsAlgorithm.ts";

test("meanPooling computes correct average excluding padded tokens", () => {
  // Batch 1, Sequence length 3, Hidden dimension 2
  // Token 0: [2, 4], Token 1: [4, 6], Token 2 (pad): [100, 100]
  // Attention Mask: [1, 1, 0]
  // Expected pooled: [(2+4)/2, (4+6)/2] = [3, 5]
  const hiddenState = [[[2, 4], [4, 6], [100, 100]]];
  const mask = [[1, 1, 0]];
  const [pooled] = meanPooling(hiddenState, mask);

  assert.equal(pooled.length, 2);
  assert.equal(pooled[0], 3);
  assert.equal(pooled[1], 5);
});

test("l2Normalize produces unit vector with norm 1.0", () => {
  const v = new Float32Array([3, 4]); // length = 5
  const normV = l2Normalize(v);

  assert.ok(Math.abs(normV[0] - 0.6) < 1e-6);
  assert.ok(Math.abs(normV[1] - 0.8) < 1e-6);
  const norm = Math.hypot(...normV);
  assert.ok(Math.abs(norm - 1.0) < 1e-6);
});

test("cosineSimilaritySentence accurately measures angle between vectors", () => {
  const a = new Float32Array([1, 0]);
  const b = new Float32Array([0, 1]);
  const c = new Float32Array([0.7071068, 0.7071068]);

  assert.ok(Math.abs(cosineSimilaritySentence(a, b)) < 1e-6);
  assert.ok(Math.abs(cosineSimilaritySentence(a, a) - 1.0) < 1e-6);
  assert.ok(Math.abs(cosineSimilaritySentence(a, c) - 0.7071068) < 1e-5);
});

test("buildSentenceMatrix produces symmetric matrix with unit diagonals", () => {
  const embs = [
    new Float32Array([1, 0]),
    new Float32Array([0.7071068, 0.7071068]),
    new Float32Array([0, 1]),
  ];
  const mat = buildSentenceMatrix(embs);

  assert.equal(mat.length, 3);
  assert.ok(Math.abs(mat[0][0] - 1.0) < 1e-5);
  assert.ok(Math.abs(mat[1][1] - 1.0) < 1e-5);
  assert.ok(Math.abs(mat[2][2] - 1.0) < 1e-5);
  assert.ok(Math.abs(mat[0][1] - mat[1][0]) < 1e-5);
  assert.ok(Math.abs(mat[0][2] - mat[2][0]) < 1e-5);
});

test("rankBySimilarity ranks most aligned sentence first", () => {
  const query = new Float32Array([1, 0]);
  const docs = [
    new Float32Array([0, 1]),
    new Float32Array([0.9, 0.1]),
    new Float32Array([0.5, 0.5]),
  ];
  const ranking = rankBySimilarity(query, docs);

  assert.equal(ranking.length, 3);
  assert.equal(ranking[0].index, 1);
  assert.ok(ranking[0].score > ranking[1].score);
  assert.ok(ranking[1].score > ranking[2].score);
});

test("projectSentencePCA normalizes spatial coordinates into bounds", () => {
  const embs = [
    new Float32Array([1, 0, 0]),
    new Float32Array([0, 1, 0]),
    new Float32Array([0, 0, 1]),
    new Float32Array([0.5, 0.5, 0.5]),
  ];
  const pts = projectSentencePCA(embs);

  assert.equal(pts.length, 4);
  for (const pt of pts) {
    assert.ok(pt.x >= -1.05 && pt.x <= 1.05);
    assert.ok(pt.y >= -1.05 && pt.y <= 1.05);
  }
});
