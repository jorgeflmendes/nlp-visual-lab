import assert from "node:assert/strict";
import test from "node:test";
import { viterbiTrace } from "../src/viterbiLab.ts";

test("Viterbi trace recovers the path through stored backpointers", () => {
  const trace = viterbiTrace(["Noun", "Verb"], ["will", "cook", "will"], [0.7, 0.3], [[0.3, 0.7], [0.6, 0.4]], [[0.2, 0.8], [0.8, 0.2], [0.2, 0.8]]);
  assert.equal(trace.result, "Verb → Noun → Verb · joint probability 0.064512");
  assert.equal(trace.steps.length, 10);
  assert.match(trace.render(5).detail, /Noun · chosen/);
  const visual = trace.render(5).visual;
  assert.match(visual, /aria-label="Sequence scores SS"/);
  assert.match(visual, /aria-label="Backpointers BP"/);
  assert.match(visual, /aria-label="SS\(Verb, 3\), will: 0.064512"/);
  assert.match(visual, /aria-label="BP\(Verb, 3\), will: Noun"/);
  assert.equal((visual.match(/data-jump="5"/g) ?? []).length, 3);
  assert.match(trace.render(7).detail, /pointer leads to Noun at position 2/);
  assert.match(trace.render(8).detail, /pointer leads to Verb at position 1/);
  assert.match(trace.render(9).detail, /entire state sequence has been recovered/);
});

test("Viterbi keeps the incoming maximum when every emission candidate is zero", () => {
  const trace = viterbiTrace(["A", "B"], ["x", "y"], [0.2, 0.8], [[0.5, 0.5], [0.5, 0.5]], [[1, 1], [0, 0]]);
  assert.equal(trace.result, "B → A · joint probability 0");
  assert.match(trace.render(2).detail, /B · chosen/);
  assert.match(trace.render(2).detail, /zero emission/);
  assert.match(trace.summary, /No positive final score/);
});

test("Viterbi initializes and terminates a single observation without a previous state", () => {
  const trace = viterbiTrace(["Only"], ["word"], [1], [[1]], [[1]]);
  assert.equal(trace.steps.length, 3);
  assert.equal(trace.result, "Only · joint probability 1");
  assert.match(trace.render(0).detail, /no previous state/);
  assert.match(trace.render(1).detail, /Selected maximum/);
  assert.match(trace.render(2).detail, /entire state sequence has been recovered/);
});
