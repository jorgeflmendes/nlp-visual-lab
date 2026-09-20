import test from "node:test";
import assert from "node:assert/strict";
import {
  foundationsTopics,
  structuredTopics,
  neuralTopics,
  allCategories,
} from "../src/data/topics.ts";

test("taxonomy contains 3 pillars totaling 10 laboratories", () => {
  assert.equal(foundationsTopics.length, 3);
  assert.equal(structuredTopics.length, 3);
  assert.equal(neuralTopics.length, 4);
  assert.equal(allCategories.length, 3);
});

test("foundations contains BPE, Edit Distance and TF-IDF", () => {
  const slugs = foundationsTopics.map((t) => t.slug);
  assert.deepEqual(slugs, ["bpe", "edit-distance", "tfidf"]);
});

test("structured contains N-grams, Viterbi and CKY", () => {
  const slugs = structuredTopics.map((t) => t.slug);
  assert.deepEqual(slugs, ["n-grams", "viterbi", "cky"]);
});

test("neural contains LSTM, Seq2Seq, Attention and Transformers", () => {
  const slugs = neuralTopics.map((t) => t.slug);
  assert.deepEqual(slugs, ["lstm", "seq2seq", "attention", "transformers"]);
});
