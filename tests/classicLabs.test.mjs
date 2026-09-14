import assert from "node:assert/strict";
import test from "node:test";
import { createEditDistanceTrace } from "../src/editDistanceLab.ts";
import { createBigramTrace } from "../src/bigramLab.ts";

test("edit trace retains tied paths and their final cost", () => {
  const trace = createEditDistanceTrace("BIO", "BIFE");
  assert.equal(trace.result, "Minimum edit distance: 2");
  assert.match(trace.summary, /2 minimum paths/);
  const alternate = trace.choose("path", "1");
  assert.match(alternate.render(alternate.steps.length - 1).detail, /operations cost 2/);
  assert.notDeepEqual(trace.steps, alternate.steps);
  assert.match(trace.render(trace.steps.length - 1).visual, /<strong>BIFO<\/strong>/);
  assert.match(alternate.render(alternate.steps.length - 1).visual, /<strong>BIF<\/strong>/);
});

test("edit trace supports empty strings and Unicode code points", () => {
  const empty = createEditDistanceTrace("", "");
  assert.equal(empty.result, "Minimum edit distance: 0");
  assert.match(empty.render(empty.steps.length - 1).detail, /operations cost 0/);
  const unicode = createEditDistanceTrace("", "🐈");
  assert.equal(unicode.result, "Minimum edit distance: 1");
  assert.match(unicode.summary, /0 source characters → 1 target characters/);
  assert.match(unicode.render(1).detail, /1 insertion/);
});

test("edit trace caps path enumeration without changing the distance", () => {
  const trace = createEditDistanceTrace("a".repeat(10), "a".repeat(20));
  assert.match(trace.summary, /More than 64 minimum paths/);
  const last = trace.choose("path", "63");
  assert.match(last.render(last.steps.length - 1).detail, /operations cost 10/);
  assert.equal((trace.render(0).visual.match(/<option /g) ?? []).length, 64);
});

test("edit trace escapes characters in cells and operations", () => {
  const trace = createEditDistanceTrace("<", ">");
  const rendered = trace.render(trace.steps.length - 1);
  assert.match(rendered.visual, /&lt;/);
  assert.match(rendered.detail, /“&lt;” with “&gt;”/);
});

test("bigram trace includes sentence boundaries and the complete product", () => {
  const trace = createBigramTrace("cats sleep\ncats play\ndogs play", "cats play");
  assert.equal(trace.result, "Sentence probability: 0.333333");
  assert.deepEqual(trace.steps.map((step) => step.title), ["<s> → cats", "cats → play", "play → </s>"]);
  assert.match(trace.render(2).detail, /Complete sentence probability/);
  assert.match(trace.render(2).detail, /title="0\.3333333333333333"/);
  assert.match(trace.render(1).visual, /class="ml-sentence-prefix">&lt;s&gt; cats play<\/p>/);
  assert.match(trace.render(2).visual, /class="ml-sentence-prefix">&lt;s&gt; cats play &lt;\/s&gt;<\/p>/);
  assert.match(trace.render(2).visual, /aria-label="Conditional probability product"/);
  assert.match(trace.render(2).visual, /aria-label="Count ratio product"/);
  assert.match(trace.render(2).visual, /<strong>= 0.3333333333333333<\/strong>/);
});

test("bigram trace distinguishes unseen transitions from unseen contexts", () => {
  const trace = createBigramTrace("cats sleep", "birds fly");
  assert.equal(trace.result, "Sentence probability: 0");
  assert.match(trace.render(0).detail, /Without smoothing this factor is 0/);
  assert.match(trace.render(1).detail, /maximum likelihood estimate is undefined/);
  assert.doesNotMatch(trace.render(1).detail, /frac\{0\}\{0\}/);
});

test("bigram count inspection does not replace sentence factors", () => {
  const trace = createBigramTrace("cats sleep\ncats play\ndogs play", "cats play");
  const inspected = trace.choose("pair", "1,2");
  assert.equal(inspected.result, trace.result);
  assert.equal(inspected.selectedStep, 3);
  assert.match(inspected.render(3).detail, /cats → sleep/);
  assert.match(inspected.render(3).detail, /<dt>Context count<\/dt><dd>2<\/dd>/);
  assert.doesNotMatch(inspected.render(0).detail, /Selected corpus pair/);
});
