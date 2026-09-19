import assert from "node:assert/strict";
import test from "node:test";
import {
  applyMerge,
  countAdjacentPairs,
  normalizeText,
  parseCorpus,
  tokenizeWithRules,
  trainBpe,
} from "../src/bpeAlgorithm.ts";

test("normalizeText handles NFC, lowercasing and punctuation removal", () => {
  const raw = "Olá, MUNDO! (teste)...";
  const normalized = normalizeText(raw, {
    normalizeNfc: true,
    lowercase: true,
    removePunctuation: true,
  });
  assert.equal(normalized.trim(), "olá mundo teste");
});

test("parseCorpus segments text according to splitMode", () => {
  const text = "low lower\nlowest";
  const ws = parseCorpus(text, { splitMode: "whitespace", normalizeNfc: true, lowercase: false, removePunctuation: false });
  assert.deepEqual(ws, ["low", "lower", "lowest"]);

  const lines = parseCorpus(text, { splitMode: "lines", normalizeNfc: true, lowercase: false, removePunctuation: false });
  assert.deepEqual(lines, ["low lower", "lowest"]);

  const comma = parseCorpus("a, b, c", { splitMode: "comma", normalizeNfc: true, lowercase: false, removePunctuation: false });
  assert.deepEqual(comma, ["a", "b", "c"]);
});

test("countAdjacentPairs counts frequencies and respects tie breaking", () => {
  const seqs = [
    ["l", "o", "w"],
    ["l", "o", "w", "e", "r"],
  ];
  const pairs = countAdjacentPairs(seqs, "first");
  const lo = pairs.find((p) => p.left === "l" && p.right === "o");
  const ow = pairs.find((p) => p.left === "o" && p.right === "w");
  assert.equal(lo?.count, 2);
  assert.equal(ow?.count, 2);
});

test("applyMerge combines adjacent pairs correctly across sequences", () => {
  const seqs = [
    ["l", "o", "w"],
    ["l", "o", "w", "e", "r"],
  ];
  const merged = applyMerge(seqs, { left: "l", right: "o" });
  assert.deepEqual(merged, [
    ["lo", "w"],
    ["lo", "w", "e", "r"],
  ]);
});

test("trainBpe trains correctly on simple corpus", () => {
  const corpus = "low lower lowest new newer newest wide wider widest";
  const model = trainBpe(corpus, {
    splitMode: "whitespace",
    tieBreak: "first",
    normalizeNfc: true,
    lowercase: false,
    removePunctuation: false,
    targetVocabSize: 30,
    maxMerges: 5,
    minPairFrequency: 2,
  });

  assert.ok(model.rules.length > 0);
  assert.equal(model.states.length, model.rules.length + 1);
  assert.ok(model.initialVocab.length > 0);
  assert.ok(model.states.at(-1).vocab.length > model.initialVocab.length);
});

test("tokenizeWithRules applies merges to unseen text and tracks trace", () => {
  const rules = [
    { left: "l", right: "o", merged: "lo", count: 2 },
    { left: "lo", right: "w", merged: "low", count: 2 },
  ];
  const { tokens, trace } = tokenizeWithRules(["l", "o", "w", "e", "r"], rules);
  assert.deepEqual(tokens, ["low", "e", "r"]);
  assert.equal(trace.length, 3);
  assert.deepEqual(trace[0], ["l", "o", "w", "e", "r"]);
  assert.deepEqual(trace[1], ["lo", "w", "e", "r"]);
  assert.deepEqual(trace[2], ["low", "e", "r"]);
});
