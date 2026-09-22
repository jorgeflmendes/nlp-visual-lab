import test from "node:test";
import assert from "node:assert/strict";
import { getTopicIcon } from "../src/data/icons.ts";

const allSlugs = [
  "bpe",
  "edit-distance",
  "tfidf",
  "word-embeddings",
  "n-grams",
  "viterbi",
  "cky",
  "lstm",
  "seq2seq",
  "attention",
  "sentence-embeddings",
  "transformers",
];

test("all 12 topic slugs return valid SVG strings", () => {
  const fallback = getTopicIcon("unknown-slug");
  for (const slug of allSlugs) {
    const svg = getTopicIcon(slug);
    assert.ok(svg.startsWith("<svg"), `SVG for ${slug} should start with <svg`);
    assert.ok(svg.includes('viewBox="0 0 24 24"'), `SVG for ${slug} should have viewBox`);
    assert.ok(svg.endsWith("</svg>"), `SVG for ${slug} should end with </svg>`);
    assert.notEqual(svg, fallback, `SVG for ${slug} should not be the default fallback`);
  }
});

test("getTopicIcon injects custom className", () => {
  const svg = getTopicIcon("tfidf", "custom-class");
  assert.ok(svg.includes('class="custom-class"'));
});

test("unknown slug returns fallback SVG", () => {
  const fallback = getTopicIcon("unknown-slug");
  assert.ok(fallback.startsWith("<svg"));
});
