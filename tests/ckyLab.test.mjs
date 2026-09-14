import assert from "node:assert/strict";
import test from "node:test";
import { ckyChart, parseCkyGrammar } from "../src/algorithms.ts";
import { recordCky } from "../src/ckyLab.ts";

const grammar = parseCkyGrammar(`S -> NP VP
NP -> Det N | NP PP | I
VP -> V NP | VP PP
PP -> P NP
Det -> the
N -> man | telescope
V -> saw
P -> with`).rules;

test("CKY replay preserves all chart symbols and competing derivations", () => {
  const words = "I saw the man with the telescope".split(" ");
  const { operations } = recordCky(words, grammar);
  const expected = ckyChart(words, grammar);
  const replay = Array.from({ length: words.length }, () =>
    Array.from({ length: words.length + 1 }, () => ({ symbols: new Set(), derivations: [] })),
  );
  for (const operation of operations) {
    if (operation.left !== undefined) {
      assert.ok(replay[operation.start][operation.split].symbols.has(operation.left));
      assert.ok(replay[operation.split][operation.end].symbols.has(operation.right));
    }
    const cell = replay[operation.start][operation.end];
    for (const derivation of operation.additions) {
      cell.symbols.add(derivation.parent);
      cell.derivations.push(derivation);
    }
  }
  for (let start = 0; start < words.length; start += 1) {
    for (let end = start + 1; end <= words.length; end += 1) {
      assert.deepEqual([...replay[start][end].symbols].sort(), [...expected[start][end].symbols].sort());
      assert.deepEqual(replay[start][end].derivations.map(JSON.stringify).sort(), expected[start][end].derivations.map(JSON.stringify).sort());
    }
  }
  assert.equal(replay[1][words.length].derivations.filter(({ parent }) => parent === "VP").length, 2);
});

test("CKY records failed binary pairs and empty sources without adding symbols", () => {
  const { chart, operations } = recordCky(["I", "I", "bird"], grammar);
  const failedPair = operations.find(({ start, end }) => start === 0 && end === 2);
  assert.equal(failedPair.left, "NP");
  assert.equal(failedPair.right, "NP");
  assert.deepEqual(failedPair.additions, []);
  const emptySource = operations.find(({ start, end }) => start === 1 && end === 3);
  assert.equal(emptySource.split, 2);
  assert.equal(emptySource.left, undefined);
  assert.deepEqual(emptySource.additions, []);
  assert.deepEqual(chart[0][3].symbols, []);
});

test("CKY lexical replay keeps multiple terminal categories", () => {
  const { operations } = recordCky(["book"], parseCkyGrammar("N -> book\nV -> book").rules);
  assert.equal(operations.length, 1);
  assert.deepEqual(operations[0].additions, [
    { parent: "N", terminal: "book" },
    { parent: "V", terminal: "book" },
  ]);
});
