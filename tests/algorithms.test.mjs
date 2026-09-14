import assert from "node:assert/strict";
import test from "node:test";
import { bigramCount, bigramVocabulary, ckyChart, editMatrix, optimalEditPaths, parseCkyGrammar, sentenceBigramProbability, viterbi } from "../src/algorithms.ts";

test("edit distance handles substitutions and insertions", () => {
  const matrix = editMatrix("BIO", "BIFE");
  assert.equal(matrix.at(-1).at(-1).value, 2);
  assert.deepEqual(optimalEditPaths(matrix), [
    [[0, 0], [1, 1], [2, 2], [3, 2], [4, 3]],
    [[0, 0], [1, 1], [2, 2], [3, 3], [4, 3]],
  ]);
});

test("edit distance reconstructs paths when one string is empty", () => {
  const matrix = editMatrix("abc", "");
  assert.equal(matrix.at(-1).at(-1).value, 3);
  assert.deepEqual(optimalEditPaths(matrix), [[[0, 0], [0, 1], [0, 2], [0, 3]]]);
});

test("bigram counts include sentence boundaries", () => {
  const corpus = `in the summer it is hot
in the summer it is very hot
the butterflies are very hot
the butterflies fly a lot in the summer
in the summer the butterflies fly
the butterflies fly when it is hot`;
  assert.equal(bigramCount(corpus, "the", "butterflies"), 4);
  const result = sentenceBigramProbability(corpus, "the butterflies fly");
  assert.deepEqual(result.factors.map(({ bigramCount, previousCount }) => [bigramCount, previousCount]), [
    [3, 6],
    [4, 8],
    [3, 4],
    [1, 3],
  ]);
  assert.equal(result.probability, 1 / 16);
});

test("bigram vocabulary follows the edited corpus and query", () => {
  assert.deepEqual(bigramVocabulary("gatos dormem\ncães correm", "aves voam"), [
    "<s>", "gatos", "dormem", "cães", "correm", "aves", "voam", "</s>",
  ]);
});

test("CKY finds constituents across every split", () => {
  const grammar = parseCkyGrammar(`S -> NP VP
NP -> Det N | I
VP -> V NP
Det -> the
N -> dog
V -> saw`);
  assert.deepEqual(grammar.errors, []);
  const chart = ckyChart(["I", "saw", "the", "dog"], grammar.rules);
  assert.deepEqual(chart[2][4].symbols, ["NP"]);
  assert.deepEqual(chart[1][4].symbols, ["VP"]);
  assert.deepEqual(chart[0][4].symbols, ["S"]);
  assert.equal(chart[0][4].derivations[0].split, 1);
});

test("CKY reports grammar lines outside CNF", () => {
  const grammar = parseCkyGrammar("S -> NP V NP");
  assert.equal(grammar.rules.length, 0);
  assert.equal(grammar.errors.length, 1);
});

test("viterbi reconstructs the globally best path", () => {
  const result = viterbi(
    [[0.2, 0.25], [0.2, 0.1], [0.2, 0.25]],
    [[0.2, 0.4], [0.4, 0.55]],
    [0.7, 0.3],
  );
  assert.deepEqual(result.path, [0, 1, 1]);
});

test("viterbi accepts any number of states", () => {
  const result = viterbi(
    [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
    [[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
    [0.8, 0.1, 0.1],
  );
  assert.deepEqual(result.path, [0, 1]);
  assert.equal(result.scores[0].length, 3);
});
