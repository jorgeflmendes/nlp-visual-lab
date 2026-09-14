import { bigramStatistics, bigramVocabulary, sentenceBigramProbability } from "./algorithms.ts";
import type { Topic } from "./data/topics.ts";
import { bindMethodWorkspace, escapeHtml, math, methodPage, type MethodTrace } from "./methodWorkspace.ts";

const practiceCorpus = `in the summer it is hot
in the summer it is very hot
the butterflies are very hot
the butterflies fly a lot in the summer
in the summer the butterflies fly
the butterflies fly when it is hot`;

export function ngramsPage(topic: Topic): string {
  return methodPage(topic, {
    title: "Count context. Measure a sentence.",
    description: "Estimate each next word from your corpus, then trace how those probabilities combine from start to end.",
    inputs: `<label>Training corpus<textarea id="ngram-corpus" name="ngram-corpus" rows="7" spellcheck="false">${practiceCorpus}</textarea></label><label>Query sentence<input id="ngram-sentence" name="ngram-sentence" value="the butterflies fly" spellcheck="false" /></label>`,
    notes: "Each non-empty line is a sentence. Words are lowercased; punctuation is ignored. Sentence boundaries count as tokens. No smoothing is applied. Up to 80 vocabulary tokens and 80 query words.",
    examples: [
      { label: "Butterflies", values: { "ngram-corpus": practiceCorpus, "ngram-sentence": "the butterflies fly" } },
      { label: "Unseen transition", values: { "ngram-corpus": practiceCorpus, "ngram-sentence": "the summer flies" } },
      { label: "Small corpus", values: { "ngram-corpus": "cats sleep\ncats play\ndogs play", "ngram-sentence": "cats play" } },
    ],
  });
}

function probabilityText(value: number): string {
  return value === 0 || value === 1 ? String(value) : Number(value.toPrecision(6)).toString();
}

function latexToken(token: string): string {
  return token === "<s>" ? "\\langle s\\rangle" : token === "</s>" ? "\\langle /s\\rangle" : `\\text{${token}}`;
}

export function createBigramTrace(corpus: string, sentence: string): MethodTrace {
  if (!corpus.trim()) throw new Error("Enter at least one corpus sentence to estimate its bigrams.");
  const vocabulary = bigramVocabulary(corpus, sentence);
  if (vocabulary.length > 80) throw new Error("Use up to 80 distinct vocabulary tokens, including sentence boundaries, to keep the count matrix inspectable.");
  const calculation = sentenceBigramProbability(corpus, sentence);
  if (calculation.factors.length > 81) throw new Error("Use a query of up to 80 words to keep each factor inspectable.");
  const statistics = bigramStatistics(corpus);
  const factorIndexes = new Map(calculation.factors.map((factor, index) => [`${factor.previous}\u0000${factor.next}`, index]));
  const prefixProbabilities: number[] = [];
  calculation.factors.forEach((factor, index) => prefixProbabilities.push((prefixProbabilities[index - 1] ?? 1) * factor.probability));
  const zeroCount = calculation.factors.filter((factor) => factor.probability === 0).length;
  const trace = (inspectedPair?: [string, string]): MethodTrace => ({
    result: `Sentence probability: ${probabilityText(calculation.probability)}`,
    summary: `${calculation.factors.length} conditional factors · ${vocabulary.length} vocabulary tokens · ${statistics.contexts.get("<s>") ?? 0} corpus sentences${zeroCount ? ` · ${zeroCount} zero ${zeroCount === 1 ? "factor" : "factors"}` : ""}`,
    steps: [
      ...calculation.factors.map((factor) => ({ title: `${factor.previous} → ${factor.next}`, stage: factor.next === "</s>" ? "End sentence" : factor.previous === "<s>" ? "Start sentence" : "Estimate next word" })),
      ...(inspectedPair ? [{ title: `${inspectedPair[0]} → ${inspectedPair[1]}`, stage: "Inspect corpus count" }] : []),
    ],
    selectedStep: inspectedPair ? calculation.factors.length : undefined,
    choose(name, value) {
      if (name !== "pair") return trace();
      const [row, column] = value.split(",").map(Number);
      return vocabulary[row] !== undefined && vocabulary[column] !== undefined ? trace([vocabulary[row], vocabulary[column]]) : trace();
    },
    render(index) {
      const inspecting = index === calculation.factors.length && inspectedPair !== undefined;
      const current = calculation.factors[Math.min(index, calculation.factors.length - 1)];
      const pair = inspecting ? inspectedPair : [current.previous, current.next];
      const inspectedCount = statistics.bigrams.get(`${pair[0]}\u0000${pair[1]}`) ?? 0;
      const inspectedContext = statistics.contexts.get(pair[0]) ?? 0;
      const conditional = (previous: string, next: string, count: number, context: number): string => math(`P(${latexToken(next)}\\mid ${latexToken(previous)})=${context === 0 ? "\\text{undefined (unseen context)}" : `\\frac{${count}}{${context}}`}`);
      const reason = current.previousCount === 0
        ? `The context “${current.previous}” never precedes another token in this corpus. Its maximum likelihood estimate is undefined; this calculator reports 0 for an unseen context.`
        : current.bigramCount === 0
          ? `“${current.previous}” occurs ${current.previousCount} times as a context, but never before “${current.next}”. Without smoothing this factor is 0, so the full sentence probability is 0.`
          : `Of ${current.previousCount} occurrences of “${current.previous}” as a context, ${current.bigramCount} are followed by “${current.next}”.`;
      const through = inspecting ? calculation.factors.length - 1 : index;
      const prefix = calculation.factors.slice(0, through + 1);
      const sequence = [prefix[0].previous, ...prefix.map((factor) => factor.next)].join(" ");
      const expansion = `<section class="ml-expansion" data-view="bigram-product" aria-label="Expanded sentence probability"><h2>${through === calculation.factors.length - 1 ? "Complete sentence" : "Sentence so far"}</h2><p class="ml-sentence-prefix">${escapeHtml(sequence)}</p>
        <div class="ml-product-line" aria-label="Conditional probability product"><span>P(${escapeHtml(prefix.map((factor) => factor.next).join(" "))}) =</span>${prefix.map((factor, i) => `<span class="ml-product-term">${i ? '<span aria-hidden="true">×</span>' : ""}<button type="button" data-jump="${i}" aria-label="Inspect factor ${i + 1}: ${escapeHtml(factor.previous)} to ${escapeHtml(factor.next)}" aria-pressed="${i === index}">${math(`P(${latexToken(factor.next)}\\mid ${latexToken(factor.previous)})`)}</button></span>`).join("")}</div>
        <div class="ml-product-line" aria-label="Count ratio product"><span>=</span>${prefix.map((factor, i) => `${i ? '<span aria-hidden="true">×</span>' : ""}${math(factor.previousCount ? `\\frac{${factor.bigramCount}}{${factor.previousCount}}` : '\\underbrace{0}_{\\text{unseen context}}')}`).join("")}</div>
        <div class="ml-product-line" aria-label="Numerical product"><span>=</span>${prefix.map((factor, i) => `${i ? '<span aria-hidden="true">×</span>' : ""}<code>${factor.probability}</code>`).join("")}<strong>= ${prefixProbabilities[through]}</strong></div>
        <p class="ml-hint">Each factor predicts the next token from the previous one. The prefix begins after &lt;s&gt;; the complete sentence includes &lt;/s&gt;. ${prefix.some((factor) => !factor.previousCount) ? "An unseen context has no defined estimate; its factor is set to 0 by this calculator." : ""}</p></section>`;
      const visual = `<div class="ml-visual-heading"><h2>Sentence factors</h2><small>${inspecting ? "Inspecting a corpus count" : `${index + 1} of ${calculation.factors.length} · boundaries included`}</small></div>
        <div class="ml-token-row" data-view="bigram-factors" aria-label="Conditional probability factors">${calculation.factors.map((factor, i) => `<button type="button" data-jump="${i}" class="${i === index ? "current" : ""}" aria-pressed="${i === index}"><small>${escapeHtml(factor.previous)} → ${escapeHtml(factor.next)}</small><strong>${probabilityText(factor.probability)}</strong><small>${factor.previousCount ? `${factor.bigramCount} / ${factor.previousCount}` : "unseen context"}</small></button>`).join("")}</div>
        ${expansion}
        <div class="ml-visual-heading"><h2>Corpus counts</h2><small>Previous token ↓ · next token →</small></div>
        <p class="ml-hint">Bigram counts; the last column is the sum of each context row.</p><div class="ml-table-scroll"><table class="ml-matrix" data-view="bigram-counts" aria-label="Bigram counts"><thead><tr><th scope="col">previous / next</th>${vocabulary.map((token) => `<th scope="col" class="${token === pair[1] ? "active" : ""}">${escapeHtml(token)}</th>`).join("")}<th scope="col">Context total</th></tr></thead><tbody>${vocabulary.map((previous, r) => `<tr><th scope="row" class="${previous === pair[0] ? "active" : ""}">${escapeHtml(previous)}</th>${vocabulary.map((next, c) => {
          const count = statistics.bigrams.get(`${previous}\u0000${next}`) ?? 0;
          const selected = previous === pair[0] && next === pair[1];
          const factorIndex = factorIndexes.get(`${previous}\u0000${next}`);
          return `<td class="${selected ? "current" : ""}"><button type="button" ${factorIndex !== undefined ? `data-jump="${factorIndex}"` : `data-choice="pair" value="${r},${c}"`} aria-pressed="${selected}" aria-label="Count of ${escapeHtml(previous)} followed by ${escapeHtml(next)}: ${count}"><strong>${count}</strong></button></td>`;
        }).join("")}<td class="context-count">${statistics.contexts.get(previous) ?? 0}</td></tr>`).join("")}</tbody></table></div><p class="ml-legend">Select a sentence pair to jump to its factor, or another count to inspect its estimate. Context totals count outgoing transitions; the end token has none. Sentence factors are rounded; the inspector shows stored values.</p>`;
      const detail = inspecting
        ? `<h3>Selected corpus pair</h3><p>${escapeHtml(pair[0])} → ${escapeHtml(pair[1])}</p>${conditional(pair[0], pair[1], inspectedCount, inspectedContext)}<dl class="ml-facts"><div><dt>Pair count</dt><dd>${inspectedCount}</dd></div><div><dt>Context count</dt><dd>${inspectedContext}</dd></div><div><dt>Estimate</dt><dd>${inspectedContext ? probabilityText(inspectedCount / inspectedContext) : "Undefined; reported as 0"}</dd></div></dl><p>${inspectedContext ? `The row contains ${inspectedContext} outgoing transitions; ${inspectedCount} go to this next token.` : "This context has no outgoing transitions in the corpus, so the maximum likelihood estimate is undefined."} Select a sentence factor to return to its running product.</p>`
        : `<h3>Estimate factor ${index + 1}</h3>${conditional(current.previous, current.next, current.bigramCount, current.previousCount)}<dl class="ml-facts"><div><dt>Pair count</dt><dd>${current.bigramCount}</dd></div><div><dt>Context count</dt><dd>${current.previousCount}</dd></div><div><dt>Conditional probability</dt><dd title="${current.probability}">${current.probability}</dd></div></dl><p>${escapeHtml(reason)}</p>
        <h3>${index === calculation.factors.length - 1 ? "Complete sentence probability" : "Probability through this factor"}</h3>${math(`p_{${index + 1}}=p_{${index}}\\times P(w_{${index + 1}}\\mid w_{${index}})`)}<dl class="ml-facts"><div><dt>Previous product</dt><dd title="${prefixProbabilities[index - 1] ?? 1}">${prefixProbabilities[index - 1] ?? 1}</dd></div><div><dt>× Current factor</dt><dd>${current.probability}</dd></div><div><dt>= Running product</dt><dd title="${prefixProbabilities[index]}">${prefixProbabilities[index]}</dd></div></dl><p>The product begins at 1. The last factor includes the end-of-sentence token; every displayed value comes from the edited corpus.</p>
        `;
      return { visual, detail };
    },
  });
  return trace();
}

export function bindNgrams(): void {
  bindMethodWorkspace(() => createBigramTrace(document.querySelector<HTMLTextAreaElement>("#ngram-corpus")!.value, document.querySelector<HTMLInputElement>("#ngram-sentence")!.value));
}
