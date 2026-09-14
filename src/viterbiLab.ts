import { viterbi } from "./algorithms.ts";
import type { Topic } from "./data/topics.ts";
import { bindMethodWorkspace, escapeHtml, math, methodPage, type MethodTrace } from "./methodWorkspace.ts";

export function viterbiPage(topic: Topic): string {
  return methodPage(topic, {
    title: "Find the most likely hidden sequence",
    description: "Set the observations and model, then follow every maximum and backpointer through the trellis.",
    examples: [
      { label: "Another sequence", values: { "viterbi-sentence": "cook will cook" } },
      { label: "One observation", values: { "viterbi-sentence": "will" } },
    ],
    inputs: `<label>Hidden states · comma separated<input id="viterbi-states" name="viterbi-states" value="Noun, Verb" spellcheck="false" /></label>
      <label>Observation vocabulary · comma separated<input id="viterbi-vocabulary" name="viterbi-vocabulary" value="will, cook" spellcheck="false" /></label>
      <label>Observation sequence<input id="viterbi-sentence" name="viterbi-sentence" value="will cook will" spellcheck="false" /></label>
      <p class="ml-hint">Words are lowercased and split at punctuation. Every observed word must be in the vocabulary.</p>
      <details class="ml-model-editor"><summary>Model probabilities</summary><p>These example probabilities are set by hand. The start distribution and each transition and emission row must sum to 1. Changing the sequence keeps the model.</p><div id="hmm-probabilities"></div></details>`,
    notes: `${math("\\delta_1(s)=\\pi_s b_s(o_1)")}${math("\\delta_t(s)=b_s(o_t)\\max_r[\\delta_{t-1}(r)a_{rs}]")}${math("\\psi_t(s)=\\arg\\max_r[\\delta_{t-1}(r)a_{rs}]")}
      <p>A score is the joint probability of the observations and the best state prefix ending at that node. It is not a normalized probability over states.</p>
      <p>The backpointer records the first state with the maximum incoming score. Termination chooses the first final maximum; traceback follows those pointers. Calculations use JavaScript numbers and direct multiplication, so sufficiently small products can underflow to zero.</p>`,
  });
}

export function bindViterbi(): void {
  const statesInput = document.querySelector<HTMLInputElement>("#viterbi-states")!;
  const vocabularyInput = document.querySelector<HTMLInputElement>("#viterbi-vocabulary")!;
  const sentenceInput = document.querySelector<HTMLInputElement>("#viterbi-sentence")!;
  const editor = document.querySelector<HTMLDivElement>("#hmm-probabilities")!;
  const initialValues = new Map<string, string>([["Noun", "0.7"], ["Verb", "0.3"]]);
  const transitionValues = new Map<string, string>([["Noun\u0000Noun", "0.3"], ["Noun\u0000Verb", "0.7"], ["Verb\u0000Noun", "0.6"], ["Verb\u0000Verb", "0.4"]]);
  const emissionValues = new Map<string, string>([["Noun\u0000will", "0.2"], ["Noun\u0000cook", "0.8"], ["Verb\u0000will", "0.8"], ["Verb\u0000cook", "0.2"]]);
  const defaults = [initialValues, transitionValues, emissionValues].map((values) => new Map(values));

  const readStructure = () => ({
    states: statesInput.value.split(",").map((value) => value.trim()).filter(Boolean),
    vocabulary: vocabularyInput.value.toLocaleLowerCase("en").split(",").map((value) => value.trim()).filter(Boolean),
    words: sentenceInput.value.toLocaleLowerCase("en").match(/[\p{L}\p{N}]+/gu) ?? [],
  });

  const renderEditor = () => {
    const { states, vocabulary } = readStructure();
    if (!states.length || !vocabulary.length || new Set(states).size !== states.length || new Set(vocabulary).size !== vocabulary.length) {
      editor.innerHTML = `<p class="ml-hint">Enter unique state names and vocabulary words to edit their probabilities.</p>`;
      return;
    }
    const field = (kind: "initial" | "transition" | "emission", key: string, label: string, fallback: number) => {
      const values = kind === "initial" ? initialValues : kind === "transition" ? transitionValues : emissionValues;
      if (!values.has(key)) values.set(key, String(fallback));
      return `<td><input type="number" min="0" max="1" step="any" aria-label="${escapeHtml(label)}" data-probability="${kind}" data-key="${escapeHtml(encodeURIComponent(key))}" value="${escapeHtml(values.get(key)!)}" /></td>`;
    };
    editor.innerHTML = `<div class="ml-table-scroll"><table class="ml-matrix"><caption>Start · P(state)</caption><thead><tr>${states.map((state) => `<th scope="col">${escapeHtml(state)}</th>`).join("")}</tr></thead><tbody><tr>${states.map((state) => field("initial", state, `Start probability of ${state}`, 1 / states.length)).join("")}</tr></tbody></table></div>
      <div class="ml-table-scroll"><table class="ml-matrix"><caption>Transitions · P(next state | current state)</caption><thead><tr><th scope="col">Current ↓ / Next →</th>${states.map((state) => `<th scope="col">${escapeHtml(state)}</th>`).join("")}</tr></thead><tbody>${states.map((from) => `<tr><th scope="row">${escapeHtml(from)}</th>${states.map((to) => field("transition", `${from}\u0000${to}`, `Transition from ${from} to ${to}`, 1 / states.length)).join("")}</tr>`).join("")}</tbody></table></div>
      <div class="ml-table-scroll"><table class="ml-matrix"><caption>Emissions · P(word | state)</caption><thead><tr><th scope="col">State ↓ / Word →</th>${vocabulary.map((word) => `<th scope="col">${escapeHtml(word)}</th>`).join("")}</tr></thead><tbody>${states.map((state) => `<tr><th scope="row">${escapeHtml(state)}</th>${vocabulary.map((word) => field("emission", `${state}\u0000${word}`, `Emission of ${word} from ${state}`, 1 / vocabulary.length)).join("")}</tr>`).join("")}</tbody></table></div>`;
  };

  editor.addEventListener("input", (event) => {
    const input = event.target as HTMLInputElement;
    const kind = input.dataset.probability;
    if (!kind || input.dataset.key === undefined) return;
    const values = kind === "initial" ? initialValues : kind === "transition" ? transitionValues : emissionValues;
    values.set(decodeURIComponent(input.dataset.key), input.value);
  });
  statesInput.addEventListener("input", renderEditor);
  vocabularyInput.addEventListener("input", renderEditor);
  document.querySelector("#method-workspace")!.addEventListener("method-reset", () => {
    [initialValues, transitionValues, emissionValues].forEach((values, index) => {
      values.clear();
      defaults[index].forEach((value, key) => values.set(key, value));
    });
    renderEditor();
  });
  renderEditor();

  bindMethodWorkspace(() => {
    const { states, vocabulary, words } = readStructure();
    if (!states.length || !vocabulary.length || !words.length) throw new Error("Enter at least one state, vocabulary word, and observation.");
    if (new Set(states).size !== states.length) throw new Error("State names must be unique.");
    if (new Set(vocabulary).size !== vocabulary.length) throw new Error("Vocabulary words must be unique.");
    if (vocabulary.some((word) => !/^[\p{L}\p{N}]+$/u.test(word))) throw new Error("Each vocabulary entry must be one word containing letters or numbers.");
    const unknown = words.find((word) => !vocabulary.includes(word));
    if (unknown) throw new Error(`Add “${unknown}” to the vocabulary and set its emission probabilities.`);
    const distribution = (values: string[], label: string): number[] => {
      const numbers = values.map((value) => value.trim() === "" ? NaN : Number(value));
      if (numbers.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) throw new Error(`${label}: enter a probability from 0 to 1 in every cell.`);
      const total = numbers.reduce((sum, value) => sum + value, 0);
      if (Math.abs(total - 1) > 1e-9) throw new Error(`${label}: probabilities sum to ${total}; they must sum to 1.`);
      return numbers;
    };
    const initial = distribution(states.map((state) => initialValues.get(state) ?? ""), "Start distribution");
    const transitions = states.map((from) => distribution(states.map((to) => transitionValues.get(`${from}\u0000${to}`) ?? ""), `Transitions from ${from}`));
    const emissionRows = states.map((state) => distribution(vocabulary.map((word) => emissionValues.get(`${state}\u0000${word}`) ?? ""), `Emissions from ${state}`));
    const emissions = words.map((word) => states.map((_, state) => emissionRows[state][vocabulary.indexOf(word)]));
    return viterbiTrace(states, words, initial, transitions, emissions);
  });
}

export function viterbiTrace(states: string[], words: string[], initial: number[], transitions: number[][], emissions: number[][]): MethodTrace {
  const model = viterbi(emissions, transitions, initial);
  const size = states.length;
  const forwardCount = words.length * size;
  const finalScore = model.scores.at(-1)![model.path.at(-1)!];
  const steps = [
    ...words.flatMap((word, time) => states.map((state) => ({ title: `${time + 1}. ${word} → ${state}`, stage: time === 0 ? "Initialize" : "Compare candidates" }))),
    { title: "Choose the final state", stage: "Terminate" },
    ...words.map((_, offset) => ({ title: `Recover state ${words.length - offset}`, stage: "Trace back" })),
  ];
  const writtenPath = model.path.map((state, time) => `<span><small>${escapeHtml(words[time])}</small><strong>${escapeHtml(states[state])}</strong></span>`).join(`<span aria-hidden="true">→</span>`);
  return {
    result: `${model.path.map((state) => states[state]).join(" → ")} · joint probability ${finalScore}`,
    summary: `${words.length} observations · ${states.length} states · ${forwardCount} scores${finalScore === 0 ? ". No positive final score was represented. Zero probabilities or numeric underflow can cause this; the path follows the stored backpointers and first-maximum tie rule." : ""}`,
    steps,
    render(index) {
      const isForward = index < forwardCount;
      const isTermination = index === forwardCount;
      const time = isForward ? Math.floor(index / size) : isTermination ? words.length - 1 : words.length - 1 - (index - forwardCount - 1);
      const state = isForward ? index % size : model.path[time];
      const previous = time > 0 ? model.backpointers[time][state] : -1;
      const revealed = (t: number, s: number) => !isForward || t * size + s <= index;
      const tables = `<div class="ml-viterbi-tables">${(["SS", "BP"] as const).map((kind) => `<section><h2>${kind === "SS" ? "Sequence scores (SS)" : "Backpointers (BP)"}</h2><div class="ml-table-scroll"><table class="ml-matrix ml-viterbi-table" data-view="${kind}" aria-label="${kind === "SS" ? "Sequence scores SS" : "Backpointers BP"}"><thead><tr><th scope="col">State</th>${words.map((word, t) => `<th scope="col"><small>${t + 1}</small>${escapeHtml(word)}</th>`).join("")}</tr></thead><tbody>${states.map((name, s) => `<tr><th scope="row">${escapeHtml(name)}</th>${words.map((word, t) => {
        const shown = revealed(t, s);
        const active = t === time && s === state;
        const dependency = t === time - 1 && s === previous;
        const onPath = !isForward && model.path[t] === s;
        const value = kind === "SS" ? String(model.scores[t][s]) : t === 0 ? "—" : states[model.backpointers[t][s]];
        const label = `${kind}(${name}, ${t + 1}), ${word}: ${shown ? value : "not yet computed"}`;
        return `<td class="${active ? "current" : dependency ? "dependency" : onPath ? "on-path" : ""} ${shown ? "" : "pending"}"><button type="button" data-jump="${t * size + s}" aria-pressed="${active}" aria-label="${escapeHtml(label)}">${shown ? kind === "SS" ? String(Number(model.scores[t][s].toPrecision(6))) : escapeHtml(value) : "·"}</button></td>`;
      }).join("")}</tr>`).join("")}</tbody></table></div><p class="ml-hint">${kind === "SS" ? "SS(s, t): best prefix score ending in state s at position t. Values are rounded; select a cell for its stored value." : "BP(s, t): previous state selected by the maximum. — marks the first observation."}</p></section>`).join("")}</div>`;
      const nodeWidth = 120;
      const columnWidth = 150;
      const rowHeight = 100;
      const x = (t: number) => 90 + t * columnWidth;
      const y = (s: number) => 48 + s * rowHeight;
      const width = x(words.length - 1) + nodeWidth + 18;
      const height = y(size - 1) + 84;
      const links = words.slice(1).flatMap((_, offset) => states.map((__, destination) => {
        const t = offset + 1;
        if (!revealed(t, destination)) return "";
        const source = model.backpointers[t][destination];
        const selected = t === time && destination === state;
        const onPath = !isForward && model.path[t] === destination && model.path[t - 1] === source;
        return `<path class="${selected ? "selected" : onPath ? "on-path" : ""}" d="M ${x(t - 1) + nodeWidth} ${y(source) + 36} L ${x(t)} ${y(destination) + 36}" fill="none" stroke="${selected ? "#c26b2c" : onPath ? "#187e74" : "#b3bdbb"}" stroke-width="${selected || onPath ? 3 : 1}" />`;
      })).join("");
      const nodes = states.flatMap((name, s) => words.map((word, t) => {
        const shown = revealed(t, s);
        const active = t === time && s === state;
        const source = isForward && t === time - 1 && s === previous;
        const onPath = !isForward && model.path[t] === s;
        const score = model.scores[t][s];
        const shortScore = score === 0 ? "0" : score < 0.00001 ? score.toExponential(4) : String(Number(score.toPrecision(6)));
        return `<button type="button" data-jump="${t * size + s}" class="ml-trellis-node ${active ? "selected" : ""} ${source ? "dependency" : ""} ${onPath ? "on-path" : ""} ${shown ? "" : "pending"}" aria-pressed="${active}" aria-label="${escapeHtml(`${name}, observation ${t + 1}: ${word}. ${shown ? `Score ${score}. ${t > 0 ? `Backpointer ${states[model.backpointers[t][s]]}.` : "No previous state."}` : "Not yet shown. Select to inspect."}`)}" style="position:absolute;left:${x(t)}px;top:${y(s)}px;width:${nodeWidth}px;height:76px"><small>${escapeHtml(name)}</small><strong>${shown ? shortScore : "·"}</strong><small>${shown ? t === 0 ? "start" : `← ${escapeHtml(states[model.backpointers[t][s]])}` : "pending"}</small></button>`;
      })).join("");
      const trellis = `<div class="ml-visual-heading"><div><small>Hidden-state trellis</small><h2>${escapeHtml(words[time])} · ${escapeHtml(states[state])}</h2></div><span>Observation ${time + 1} / ${words.length}</span></div><p class="ml-hint">Select any node to inspect its calculation. Each line is a stored backpointer. Scores in nodes are rounded; the inspector shows stored values.</p><div class="ml-table-scroll"><div class="ml-trellis" data-view="trellis" style="position:relative;width:${width}px;height:${height}px"><svg aria-hidden="true" width="${width}" height="${height}" style="position:absolute;inset:0">${links}</svg>${words.map((word, t) => `<div class="ml-trellis-label" style="position:absolute;left:${x(t)}px;top:0;width:${nodeWidth}px"><small>${t + 1}</small><strong>${escapeHtml(word)}</strong></div>`).join("")}${states.map((name, s) => `<div class="ml-trellis-label" style="position:absolute;left:0;top:${y(s)}px;width:78px">${escapeHtml(name)}</div>`).join("")}${nodes}</div></div>`;
      let detail: string;
      if (isTermination) {
        detail = `<h3>Choose the final maximum</h3><div class="ml-table-scroll"><table class="ml-candidates"><thead><tr><th>Final state</th><th>Stored score</th><th>Decision</th></tr></thead><tbody>${states.map((name, s) => `<tr class="${s === state ? "selected" : ""}"><th>${escapeHtml(name)}</th><td>${model.scores[time][s]}</td><td>${s === state ? "Selected maximum" : model.scores[time][s] === finalScore ? "Tied; later in state order" : "Lower score"}</td></tr>`).join("")}</tbody></table></div><p>Traceback starts at <strong>${escapeHtml(states[state])}</strong> and follows the stored pointers from right to left.</p>`;
      } else if (!isForward) {
        detail = `<h3>Recover position ${time + 1}</h3><dl class="ml-facts"><dt>Observation</dt><dd>${escapeHtml(words[time])}</dd><dt>Recovered state</dt><dd>${escapeHtml(states[state])}</dd><dt>Stored score</dt><dd>${model.scores[time][state]}</dd><dt>Backpointer</dt><dd>${previous < 0 ? "None · start of sequence" : escapeHtml(states[previous])}</dd></dl><p>${previous < 0 ? "The entire state sequence has been recovered." : `The pointer leads to ${escapeHtml(states[previous])} at position ${time}. No scores are recomputed during traceback.`}</p><div class="ml-path">${model.path.slice(time).map((s, offset) => `<span><small>${time + offset + 1} · ${escapeHtml(words[time + offset])}</small><strong>${escapeHtml(states[s])}</strong></span>`).join(`<span aria-hidden="true">→</span>`)}</div>`;
      } else if (time === 0) {
        detail = `<h3>Initialize ${escapeHtml(states[state])}</h3>${math(`\\delta_1(${state + 1})=${initial[state]}\\times${emissions[0][state]}=${model.scores[0][state]}`)}<dl class="ml-facts"><dt>Start probability</dt><dd>${initial[state]}</dd><dt>Emission of ${escapeHtml(words[0])}</dt><dd>${emissions[0][state]}</dd><dt>Stored score</dt><dd>${model.scores[0][state]}</dd><dt>Backpointer</dt><dd>None · no previous state</dd></dl>`;
      } else {
        const incoming = model.scores[time - 1].map((score, source) => score * transitions[source][state]);
        detail = `<h3>Incoming candidates → ${escapeHtml(states[state])}</h3><p>Compare previous score × transition first. Multiply the selected maximum by the shared emission probability, ${emissions[time][state]}.</p><table class="ml-candidates"><thead><tr><th>Previous state</th><th>Score × transition</th><th>× Emission</th></tr></thead><tbody>${states.map((name, source) => `<tr class="${source === previous ? "selected" : ""}"><th>${escapeHtml(name)}${source === previous ? " · chosen" : ""}</th><td><small>${model.scores[time - 1][source]} × ${transitions[source][state]}</small><strong>= ${incoming[source]}</strong></td><td><small>× ${emissions[time][state]}</small><strong>= ${incoming[source] * emissions[time][state]}</strong></td></tr>`).join("")}</tbody></table><dl class="ml-facts"><dt>Stored score</dt><dd>${model.scores[time][state]}</dd><dt>Backpointer</dt><dd>${escapeHtml(states[previous])} · state ${previous + 1}</dd></dl><p>${incoming.filter((score) => score === incoming[previous]).length > 1 ? "Incoming maxima are tied. The first state in the model order wins." : `${escapeHtml(states[previous])} has the largest incoming product.`}${emissions[time][state] === 0 ? " The zero emission makes every completed candidate zero; the backpointer still follows the incoming maximum." : ""}</p>`;
      }
      return { visual: `${tables}${trellis}<p class="ml-hint">Best complete sequence</p><div class="ml-path" aria-label="Best complete state sequence">${writtenPath}</div>`, detail };
    },
  };
}
