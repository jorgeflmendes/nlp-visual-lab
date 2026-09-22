import test from "node:test";
import assert from "node:assert/strict";
import {
  createEmbeddingIndex,
  getWordVector,
  dotProduct,
  cosineSimilarity,
  findNearestNeighbors,
  solveAnalogy,
  parseExpression,
  evaluateExpression,
  computeSimilarityMatrix,
  project2DPCA,
} from "../src/wordEmbeddingsAlgorithm.ts";

test("createEmbeddingIndex indexes vocab and maps words correctly", () => {
  const vocab = ["king", "queen", "man", "woman"];
  const dim = 4;
  const buffer = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]).buffer;

  const index = createEmbeddingIndex(vocab, buffer, dim);
  assert.equal(index.vocab.length, 4);
  assert.equal(index.dim, 4);
  assert.equal(index.wordToId.get("queen"), 1);

  const manVec = getWordVector(index, "man");
  assert.ok(manVec);
  assert.equal(manVec[2], 1);

  const missing = getWordVector(index, "unknown_word");
  assert.equal(missing, null);
});

test("dotProduct and cosineSimilarity calculate mathematical distance", () => {
  const a = new Float32Array([1, 0, 0]);
  const b = new Float32Array([0, 1, 0]);
  const c = new Float32Array([2, 0, 0]);

  assert.equal(dotProduct(a, b), 0);
  assert.equal(dotProduct(a, c), 2);

  assert.equal(cosineSimilarity(a, b), 0);
  assert.ok(Math.abs(cosineSimilarity(a, c) - 1.0) < 1e-6);
});

test("findNearestNeighbors retrieves top matches excluding query words", () => {
  const vocab = ["king", "queen", "monarch", "apple"];
  const dim = 3;
  const buffer = new Float32Array([
    1, 0, 0,       // king
    0.95, 0.31, 0, // queen (close to king)
    0.98, 0.19, 0, // monarch (closest to king)
    0, 1, 0,       // apple
  ]).buffer;

  const index = createEmbeddingIndex(vocab, buffer, dim);
  const kingVec = getWordVector(index, "king");
  assert.ok(kingVec);

  const neighbors = findNearestNeighbors(index, kingVec, 2, new Set(["king"]));
  assert.equal(neighbors.length, 2);
  assert.equal(neighbors[0].word, "monarch");
  assert.equal(neighbors[1].word, "queen");
});

test("solveAnalogy calculates a - b + c and ranks candidates", () => {
  const vocab = ["king", "man", "woman", "queen", "apple"];
  const dim = 4;

  const data = new Float32Array(5 * 4);
  const setVec = (idx, arr) => {
    let s = 0;
    arr.forEach((x) => (s += x * x));
    const norm = Math.sqrt(s) || 1;
    arr.forEach((x, i) => (data[idx * 4 + i] = x / norm));
  };

  // Concept dimensions: [royalty, masculinity, femininity, food]
  setVec(0, [1, 0.9, 0, 0]);   // king
  setVec(1, [0, 0.9, 0, 0]);   // man
  setVec(2, [0, 0, 0.9, 0]);   // woman
  setVec(3, [1, 0, 0.9, 0]);   // queen
  setVec(4, [0, 0, 0, 1]);     // apple

  const index = createEmbeddingIndex(vocab, data.buffer, dim);
  const result = solveAnalogy(index, "king", "man", "woman", 2);

  assert.ok(result);
  assert.equal(result.candidates[0].word, "queen");
  assert.ok(result.candidates[0].similarity > 0.9);
});

test("computeSimilarityMatrix produces symmetric matrix with unit diagonals", () => {
  const vocab = ["cat", "dog", "car"];
  const dim = 2;
  const buffer = new Float32Array([
    1, 0,
    0.8, 0.6,
    0, 1,
  ]).buffer;

  const index = createEmbeddingIndex(vocab, buffer, dim);
  const { words, matrix } = computeSimilarityMatrix(index, ["cat", "dog", "car"]);

  assert.deepEqual(words, ["cat", "dog", "car"]);
  assert.equal(matrix.length, 3);
  assert.ok(Math.abs(matrix[0][0] - 1.0) < 1e-4);
  assert.ok(Math.abs(matrix[1][1] - 1.0) < 1e-4);
  assert.ok(Math.abs(matrix[2][2] - 1.0) < 1e-4);
  assert.ok(Math.abs(matrix[0][1] - matrix[1][0]) < 1e-4);
  assert.ok(Math.abs(matrix[0][2] - matrix[2][0]) < 1e-4);
});

test("project2DPCA projects words into normalized 2D coordinates", () => {
  const vocab = ["sun", "star", "water", "sea", "ocean"];
  const dim = 4;
  const data = new Float32Array(5 * 4);
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.sin(i * 1.5);
  }

  const index = createEmbeddingIndex(vocab, data.buffer, dim);
  const coords = project2DPCA(index, vocab);

  assert.equal(coords.length, 5);
  for (const pt of coords) {
    assert.ok(pt.x >= -1.05 && pt.x <= 1.05);
    assert.ok(pt.y >= -1.05 && pt.y <= 1.05);
  }
});

test("parseExpression parses arbitrary vector arithmetic", () => {
  const terms1 = parseExpression("king - man + woman");
  assert.equal(terms1.length, 3);
  assert.deepEqual(terms1[0], { sign: 1, weight: 1, word: "king" });
  assert.deepEqual(terms1[1], { sign: -1, weight: 1, word: "man" });
  assert.deepEqual(terms1[2], { sign: 1, weight: 1, word: "woman" });

  const terms2 = parseExpression("2 * paris - france + 0.5 * tokyo");
  assert.equal(terms2.length, 3);
  assert.deepEqual(terms2[0], { sign: 1, weight: 2, word: "paris" });
  assert.deepEqual(terms2[1], { sign: -1, weight: 1, word: "france" });
  assert.deepEqual(terms2[2], { sign: 1, weight: 0.5, word: "tokyo" });
});

test("evaluateExpression computes weighted vector linear combination", () => {
  const vocab = ["king", "man", "woman", "queen", "apple"];
  const dim = 4;
  const data = new Float32Array(5 * 4);
  const setVec = (idx, arr) => {
    let s = 0;
    arr.forEach((x) => (s += x * x));
    const norm = Math.sqrt(s) || 1;
    arr.forEach((x, i) => (data[idx * 4 + i] = x / norm));
  };

  setVec(0, [1, 0.9, 0, 0]);   // king
  setVec(1, [0, 0.9, 0, 0]);   // man
  setVec(2, [0, 0, 0.9, 0]);   // woman
  setVec(3, [1, 0, 0.9, 0]);   // queen
  setVec(4, [0, 0, 0, 1]);     // apple

  const index = createEmbeddingIndex(vocab, data.buffer, dim);
  const result = evaluateExpression(index, "king - man + woman", 2);

  assert.ok(result);
  assert.equal(result.candidates[0].word, "queen");
  assert.deepEqual(result.usedWords, ["king", "man", "woman"]);
});

