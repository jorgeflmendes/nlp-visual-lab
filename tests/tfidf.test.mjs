import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  tokenize,
  computeTf,
  computeIdf,
  computeTfidfState,
  computeCosine,
  performRetrieval,
  generateExercise,
} from "../src/tfidfAlgorithm.ts";

describe("TF-IDF Algorithm Suite", () => {
  const defaultOptions = {
    tfMode: "relative",
    idfMode: "classic",
    order: "alpha",
    precision: 3,
    lowercase: true,
    removePunctuation: true,
    removeStopwords: false,
  };

  it("tokenize handles lowercasing, punctuation, and stopwords", () => {
    const text = "The, quick brown fox! Jumps over a lazy dog.";
    const toksNoStop = tokenize(text, { lowercase: true, removePunctuation: true, removeStopwords: false });
    assert.deepEqual(toksNoStop, ["the", "quick", "brown", "fox", "jumps", "over", "a", "lazy", "dog"]);

    const toksWithStop = tokenize(text, { lowercase: true, removePunctuation: true, removeStopwords: true });
    assert.ok(!toksWithStop.includes("the"));
    assert.ok(!toksWithStop.includes("a"));
    assert.ok(toksWithStop.includes("quick"));
  });

  it("computeTf calculates relative, raw, binary and log TF correctly", () => {
    assert.equal(computeTf(0, 10, "relative"), 0);
    assert.equal(computeTf(2, 10, "relative"), 0.2);
    assert.equal(computeTf(3, 10, "raw"), 3);
    assert.equal(computeTf(4, 10, "binary"), 1);
    assert.equal(computeTf(0, 10, "binary"), 0);
    assert.equal(computeTf(1, 10, "log"), 1);
    assert.ok(Math.abs(computeTf(Math.E, 10, "log") - 2) < 1e-6);
  });

  it("computeIdf calculates classic, smooth, log10 and log2 IDF correctly", () => {
    assert.equal(computeIdf(10, 0, "classic"), 0);
    // Classic: ln(10 / 2) = ln(5)
    assert.ok(Math.abs(computeIdf(10, 2, "classic") - Math.log(5)) < 1e-6);
    // Smooth: ln((10+1)/(2+1)) + 1 = ln(11/3) + 1
    assert.ok(Math.abs(computeIdf(10, 2, "smooth") - (Math.log(11 / 3) + 1)) < 1e-6);
    // Log10: log10(10 / 1) = 1
    assert.ok(Math.abs(computeIdf(10, 1, "log10") - 1) < 1e-6);
    // Log2: log2(8 / 2) = 2
    assert.ok(Math.abs(computeIdf(8, 2, "log2") - 2) < 1e-6);
  });

  it("computeTfidfState extracts vocabulary and weights across documents", () => {
    const docs = [
      { name: "D1", text: "apple banana" },
      { name: "D2", text: "banana orange" },
    ];
    const state = computeTfidfState(docs, defaultOptions);
    assert.equal(state.N, 2);
    assert.deepEqual(state.vocab.sort(), ["apple", "banana", "orange"]);
    assert.equal(state.dfs.get("banana"), 2);
    assert.equal(state.dfs.get("apple"), 1);
    assert.equal(state.dfs.get("orange"), 1);
  });

  it("computeCosine measures vector similarity accurately", () => {
    const docs = [
      { name: "D1", text: "nlp deep learning" },
      { name: "D2", text: "nlp deep learning" },
      { name: "D3", text: "cooking recipes food" },
    ];
    const state = computeTfidfState(docs, defaultOptions);
    const identical = computeCosine(state, 0, 1);
    assert.ok(Math.abs(identical.cosine - 1.0) < 1e-5);

    const disjoint = computeCosine(state, 0, 2);
    assert.equal(disjoint.cosine, 0);
  });

  it("performRetrieval ranks documents against a query vector", () => {
    const docs = [
      { name: "D1", text: "machine learning models" },
      { name: "D2", text: "natural language processing" },
    ];
    const state = computeTfidfState(docs, defaultOptions);
    const res = performRetrieval(state, "machine learning", defaultOptions);

    assert.equal(res.ranking[0].name, "D1");
    assert.ok(res.ranking[0].score > 0);
    assert.equal(res.ranking[1].score, 0);
  });

  it("generateExercise creates self-consistent TF-IDF values", () => {
    const ex = generateExercise({ count: 2, length: 10, N: 100, df: 10 });
    assert.equal(ex.count, 2);
    assert.equal(ex.length, 10);
    assert.equal(ex.tf, 0.2);
    assert.ok(Math.abs(ex.idf - Math.log(10)) < 1e-6);
    assert.ok(Math.abs(ex.weight - 0.2 * Math.log(10)) < 1e-6);
  });
});
