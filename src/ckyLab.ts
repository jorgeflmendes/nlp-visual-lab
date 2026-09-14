import { ckyChart, parseCkyGrammar, type CkyCell, type CkyDerivation, type CkyRule } from "./algorithms.ts";
import type { Topic } from "./data/topics.ts";
import { bindMethodWorkspace, escapeHtml, math, methodPage, type MethodTrace } from "./methodWorkspace.ts";

const exampleGrammar = `S -> NP VP
NP -> Det N | NP PP | I
VP -> V NP | VP PP
PP -> P NP
Det -> the
N -> man | telescope
V -> saw
P -> with`;

export function ckyPage(topic: Topic): string {
  return methodPage(topic, {
    title: "Build a sentence, one span at a time",
    description: "Watch lexical categories become phrases. Inspect each split, the rules it matches, and the competing derivations in the chart.",
    inputs: `<label>Sentence<input id="cky-sentence" value="I saw the man with the telescope" spellcheck="false" /></label>
      <label>Start symbol<input id="cky-start" value="S" spellcheck="false" /></label>
      <label>CNF grammar<textarea id="cky-grammar" rows="10" spellcheck="false">${exampleGrammar}</textarea><small>Use A -&gt; B C or A -&gt; word. Separate alternatives with |. Matching is case sensitive.</small></label>`,
    notes: `<p>CKY fills a chart from short spans to long spans. V(i, j) covers tokens i through j − 1. A sentence is accepted when the start symbol appears in V(0, n).</p>
      ${math("V_{i,i+1}=\\{A\\mid A\\rightarrow w_i\\in G\\}")}
      ${math("V_{i,j}=\\{A\\mid\\exists k,B,C: A\\rightarrow BC\\in G, B\\in V_{i,k}, C\\in V_{k,j}\\}")}
      <p>This recognizer uses Chomsky normal form: a rule produces either two nonterminals or one terminal. It does not assign probabilities or prefer one parse. Different derivations remain in the chart.</p>`,
    examples: [
      { label: "Attachment ambiguity", values: { "cky-grammar": exampleGrammar, "cky-sentence": "I saw the man with the telescope", "cky-start": "S" } },
      { label: "Short sentence", values: { "cky-grammar": exampleGrammar, "cky-sentence": "I saw the man", "cky-start": "S" } },
      { label: "Unknown word", values: { "cky-grammar": exampleGrammar, "cky-sentence": "I saw the bird", "cky-start": "S" } },
    ],
  });
}

interface CkyOperation {
  start: number;
  end: number;
  split?: number;
  left?: string;
  right?: string;
  additions: CkyDerivation[];
}

export function recordCky(words: string[], rules: CkyRule[]): { chart: CkyCell[][]; operations: CkyOperation[] } {
  const chart = ckyChart(words, rules);
  const operations: CkyOperation[] = words.map((_, start) => ({
    start, end: start + 1, additions: chart[start][start + 1].derivations,
  }));
  for (let span = 2; span <= words.length; span += 1) {
    for (let start = 0; start + span <= words.length; start += 1) {
      const end = start + span;
      for (let split = start + 1; split < end; split += 1) {
        const leftSymbols = chart[start][split].symbols;
        const rightSymbols = chart[split][end].symbols;
        if (leftSymbols.length === 0 || rightSymbols.length === 0) {
          operations.push({ start, end, split, additions: [] });
        } else {
          for (const left of leftSymbols) {
            for (const right of rightSymbols) {
              operations.push({ start, end, split, left, right, additions: chart[start][end].derivations.filter(
                (derivation) => derivation.split === split && derivation.left === left && derivation.right === right,
              ) });
            }
          }
        }
      }
    }
  }
  return { chart, operations };
}

function symbolSet(symbols: string[]): string {
  return symbols.length ? `{ ${escapeHtml(symbols.join(", "))} }` : "∅";
}

function createCkyTrace(source: string, sentence: string, startSymbol: string): MethodTrace {
  const grammar = parseCkyGrammar(source);
  const words = sentence.trim().split(/\s+/).filter(Boolean);
  if (grammar.errors.length) throw new Error(grammar.errors.join(" "));
  if (!grammar.rules.length || !words.length || !startSymbol) throw new Error("Enter a grammar, a sentence, and a start symbol.");
  if (/\s/.test(startSymbol) || grammar.rules.some((rule) => /\s/.test(rule.left))) {
    throw new Error("Each nonterminal, including the start symbol, must be a single symbol without spaces.");
  }
  const { chart, operations } = recordCky(words, grammar.rules);
  const accepted = chart[0][words.length].symbols.includes(startSymbol);
  const lastSteps = new Map<string, number>();
  operations.forEach((operation, index) => lastSteps.set(`${operation.start},${operation.end}`, index));
  const constituents = chart.flat().reduce((count, cell) => count + cell.symbols.length, 0);

  return {
    result: accepted ? "Sentence accepted" : "Sentence rejected",
    summary: `${startSymbol} ${accepted ? "∈" : "∉"} V(0, ${words.length}). ${constituents} constituents across ${words.length * (words.length + 1) / 2} spans; ${operations.length} recorded tests.`,
    steps: operations.map((operation) => ({
      title: operation.split === undefined
        ? `Token ${operation.start}: ${words[operation.start]}`
        : `V(${operation.start}, ${operation.end}) · split ${operation.split}${operation.left ? ` · ${operation.left} + ${operation.right}` : " · empty source"}`,
      stage: operation.split === undefined ? "Lexical lookup" : `Span ${operation.end - operation.start}`,
    })),
    render(index) {
      const operation = operations[index];
      const { start, end, split } = operation;
      const progress: CkyCell[][] = Array.from({ length: words.length }, () =>
        Array.from({ length: words.length + 1 }, () => ({ symbols: [], derivations: [] })),
      );
      const filled = new Set<string>();
      const found = new Map<string, { symbol: string; start: number; end: number; step: number }>();
      for (let step = 0; step <= index; step += 1) {
        const current = operations[step];
        const cell = progress[current.start][current.end];
        filled.add(`${current.start},${current.end}`);
        for (const derivation of current.additions) {
          cell.derivations.push(derivation);
          if (!cell.symbols.includes(derivation.parent)) {
            cell.symbols.push(derivation.parent);
            found.set(`${current.start},${current.end},${derivation.parent}`, { symbol: derivation.parent, start: current.start, end: current.end, step });
          }
        }
      }
      const cell = progress[start][end];
      const parents = [...new Set(operation.additions.map((derivation) => derivation.parent))];
      const previousSymbols = [...new Set(cell.derivations.slice(0, cell.derivations.length - operation.additions.length).map((derivation) => derivation.parent))];
      const newSymbols = parents.filter((parent) => !previousSymbols.includes(parent));
      const rows = Array.from({ length: words.length }, (_, row) => words.length - row).map((span) => {
        const cells = Array.from({ length: words.length - span + 1 }, (_, left) => {
          const right = left + span;
          const key = `${left},${right}`;
          const active = start === left && end === right;
          const sourceCell = split !== undefined && ((left === start && right === split) || (left === split && right === end));
          const shown = filled.has(key);
          return `<button type="button" class="cky-cell ${shown ? "filled" : "pending"} ${active ? "current" : ""} ${sourceCell ? `source-${parents.length ? "success" : "failure"}` : ""}" data-jump="${lastSteps.get(key)}" aria-pressed="${active}" aria-label="Inspect span ${left} to ${right}: ${escapeHtml(words.slice(left, right).join(" "))}"><small>V(${left}, ${right})</small><strong>${shown ? symbolSet(progress[left][right].symbols) : "·"}</strong></button>`;
        }).join("");
        return `<div class="cky-level">${cells}</div>`;
      }).join("");
      const visual = `<div class="ml-visual-heading"><h2>Constituent chart</h2><small>${index + 1} / ${operations.length} tests · spans [i, j)</small></div>
        <p class="ml-legend">Gold: selected span · outlined: source spans · dashed: not visited. Select a cell to inspect its last test.</p>
        <div class="cky-scroll"><div class="cky-pyramid" style="--cky-columns:${words.length}">${rows}<div class="cky-tokens">${words.map((word, token) => `<span class="${token >= start && token < end ? "covered" : ""}"><b>${escapeHtml(word)}</b><small>${token}</small></span>`).join("")}</div></div></div>
        <div class="cky-constituents"><h3>Constituents discovered · ${found.size}</h3><p>Select a constituent to return to the test that first added it.</p><div>${[...found.values()].reverse().map((item) => `<button type="button" data-jump="${item.step}" class="${item.start === start && item.end === end ? "current" : ""}"><strong>${escapeHtml(item.symbol)}</strong><small>V(${item.start}, ${item.end})</small><q>${escapeHtml(words.slice(item.start, item.end).join(" "))}</q></button>`).join("") || `<p>No constituents have been discovered yet.</p>`}</div></div>`;
      const candidates = grammar.rules.filter((rule) => rule.right.length === (split === undefined ? 1 : 2));
      const candidateRows = candidates.map((rule) => {
        const matches = split === undefined
          ? rule.right[0] === words[start]
          : rule.right[0] === operation.left && rule.right[1] === operation.right;
        return `<tr class="${matches ? "selected" : ""}"><td>${escapeHtml(`${rule.left} → ${rule.right.join(" ")}`)}</td><td>${matches ? `Add ${escapeHtml(rule.left)}` : "No match"}</td></tr>`;
      }).join("");
      const derivations = cell.derivations.map((derivation) => `<li>${escapeHtml(derivation.terminal !== undefined
        ? `${derivation.parent} → ${derivation.terminal}`
        : `${derivation.parent} → ${derivation.left} ${derivation.right} at k = ${derivation.split}`)}</li>`).join("");
      const detail = `<h2>${split === undefined ? "Lexical lookup" : "Binary rule test"}</h2>
        <dl class="ml-facts"><div><dt>Target span</dt><dd>V(${start}, ${end}) · “${escapeHtml(words.slice(start, end).join(" "))}”</dd></div>
        ${split === undefined ? `<div><dt>Terminal</dt><dd>${escapeHtml(words[start])}</dd></div>` : `<div><dt>Split point</dt><dd>k = ${split}</dd></div><div><dt>Left source V(${start}, ${split})</dt><dd>${symbolSet(progress[start][split].symbols)}</dd></div><div><dt>Right source V(${split}, ${end})</dt><dd>${symbolSet(progress[split][end].symbols)}</dd></div><div><dt>Selected pair</dt><dd>${operation.left ? escapeHtml(`${operation.left} + ${operation.right}`) : "None: one source is empty"}</dd></div>`}
        <div><dt>Matched parents</dt><dd>${symbolSet(parents)}</dd></div><div><dt>Target before this test</dt><dd>${symbolSet(previousSymbols)}</dd></div><div><dt>New symbols</dt><dd>${symbolSet(newSymbols)}</dd></div><div><dt>Target after this test</dt><dd>${symbolSet(cell.symbols)}</dd></div></dl>
        <p>${parents.length ? "Every matching rule records a derivation; an existing parent symbol is kept once in the chart." : split === undefined ? "No lexical rule produces this token, so this span stays empty." : "This test adds no constituent to the target span."}</p>
        <h3>Grammar candidates</h3><div class="ml-table-scroll"><table class="ml-candidates"><thead><tr><th scope="col">${split === undefined ? "Lexical" : "Binary"} rule</th><th scope="col">This test</th></tr></thead><tbody>${candidateRows || `<tr><td colspan="2">No ${split === undefined ? "lexical" : "binary"} rules.</td></tr>`}</tbody></table></div>
        <h3>Derivations recorded in V(${start}, ${end}) · ${cell.derivations.length}</h3>${derivations ? `<ul class="ml-derivations">${derivations}</ul>` : `<p>No derivations recorded for this span.</p>`}`;
      return { visual, detail };
    },
  };
}

export function bindCky(): void {
  bindMethodWorkspace(() => createCkyTrace(
    document.querySelector<HTMLTextAreaElement>("#cky-grammar")!.value,
    document.querySelector<HTMLInputElement>("#cky-sentence")!.value,
    document.querySelector<HTMLInputElement>("#cky-start")!.value.trim(),
  ));
}
