import { groupExecutionSteps, type Candidate, type ExecutionStep, type ExecutionTrace, type TraceTensor } from "./modelTrace.ts";
import { loadModel, runModel, isModelLoaded, type RealModelKind } from "./realModels.ts";

type TypesetMath = (root: ParentNode) => Promise<void>;

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]!);
}

function visibleToken(value: string): string {
  return value.replace(/ /g, "␠").replace(/\n/g, "↵").replace(/\t/g, "⇥") || "∅";
}

function number(value: number): string {
  return Number.isInteger(value) ? String(value) : Math.abs(value) > 0 && Math.abs(value) < .0001 ? value.toExponential(3) : value.toFixed(4);
}

function shape(tensor: TraceTensor): string {
  return `[${tensor.shape.join(" × ")}]`;
}

function cellColor(value: number, scale: number): string {
  const intensity = scale ? Math.min(1, Math.abs(value) / scale) : 0;
  return `color-mix(in srgb, ${value < 0 ? "#ef4444" : "#6366f1"} ${Math.round(intensity * 75)}%, var(--soft))`;
}

function coordinates(index: number, dimensions: number[]): number[] {
  return dimensions.slice().reverse().map(size => {
    const coordinate = index % size;
    index = Math.floor(index / size);
    return coordinate;
  }).reverse();
}

function statistics(values: ArrayLike<number>): { min: number; max: number; mean: number; norm: number } {
  let min = Infinity, max = -Infinity, total = 0, squared = 0;
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    min = Math.min(min, value);
    max = Math.max(max, value);
    total += value;
    squared += value * value;
  }
  return { min, max, mean: total / values.length, norm: Math.sqrt(squared) };
}

export function bindExecutionWorkspace(kind: RealModelKind, typesetMath: TypesetMath): void {
  const workspace = document.querySelector<HTMLElement>(".dl-workspace")!;
  const input = workspace.querySelector<HTMLTextAreaElement>("#neural-input")!;
  const runButton = workspace.querySelector<HTMLButtonElement>("#run-real-model")!;
  const loadButton = workspace.querySelector<HTMLButtonElement>("#load-real-model");
  const resetButton = workspace.querySelector<HTMLButtonElement>("#reset-real-model")!;
  const status = workspace.querySelector<HTMLElement>("#model-status")!;
  const state = workspace.querySelector<HTMLElement>("#model-state")!;
  const output = workspace.querySelector<HTMLElement>("#neural-output")!;
  const runtime = workspace.querySelector<HTMLElement>("#model-runtime")!;
  const result = workspace.querySelector<HTMLElement>("#neural-result")!;
  const initialInput = input.value;
  const emptyOutput = output.innerHTML;
  const emptyResult = result.innerHTML;
  let revision = 0;
  let busy = false;
  let ready = isModelLoaded(kind);
  let disposeTrace: (() => void) | undefined;

  const setState = (label: string, message: string) => {
    state.textContent = label;
    state.dataset.state = label.toLowerCase();
    status.textContent = message;
    status.classList.remove("dl-error");
  };
  if (ready) { setState("Ready", "The model is already loaded. Run your input to capture a trace."); if (loadButton) { loadButton.disabled = true; loadButton.textContent = "Model loaded"; } }
  const invalidate = () => {
    revision++;
    disposeTrace?.();
    disposeTrace = undefined;
    result.innerHTML = emptyResult;
    output.innerHTML = emptyOutput;
    runtime.textContent = "";
    setState(busy ? "Busy" : ready ? "Ready" : "Not loaded", busy ? "Input changed. The current computation will be discarded." : "Run this input to inspect its computation.");
  };
  input.addEventListener("input", invalidate);
  resetButton.onclick = () => { input.value = initialInput; invalidate(); input.focus(); };
  workspace.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach(button => {
    button.onclick = () => { input.value = button.dataset.preset!; invalidate(); input.focus(); };
  });
  const execute = async (inference: boolean) => {
    if (busy) return;
    if (inference && !input.value.trim()) { setState(ready ? "Ready" : "Not loaded", "Enter an input first."); input.focus(); return; }
    busy = true;
    runButton.disabled = true;
    if (loadButton) loadButton.disabled = true;
    const runRevision = ++revision;
    const current = () => workspace.isConnected && runRevision === revision;
    const text = input.value;
    disposeTrace?.();
    disposeTrace = undefined;
    result.innerHTML = emptyResult;
    output.innerHTML = emptyOutput;
    runtime.textContent = "";
    const progress = (message: string, percent?: number) => {
      if (!current()) return;
      status.innerHTML = `<span>${escapeHtml(message)}</span>${percent === undefined ? '<progress aria-label="Working"></progress>' : `<progress aria-label="Model download" max="100" value="${percent}"></progress><span>${percent.toFixed(0)}%</span>`}`;
    };
    try {
      setState(ready ? "Ready" : "Loading", ready ? "Using the loaded model." : "Loading weights for this experiment. You can read the model notes below.");
      const backend = await loadModel(kind, progress);
      ready = true;
      if (!current()) return;
      runtime.textContent = backend;
      if (!inference) { setState("Ready", "Weights loaded. Run your input when ready."); return; }
      setState("Running", "Computing the forward pass locally…");
      status.innerHTML += '<progress aria-label="Computing the forward pass"></progress>';
      await new Promise(resolve => setTimeout(resolve, 0));
      const trace = await runModel(kind, text, progress);
      if (!current()) return;
      output.innerHTML = `<small>${escapeHtml(trace.outputLabel)}</small><strong></strong>`;
      runtime.textContent = `${trace.backend} · ${trace.elapsed.toFixed(1)} ms including trace capture`;
      disposeTrace = bindTrace(result, trace, typesetMath, output);
      setState("Result available", "Computation complete. Playback explores the recorded forward pass.");
      runButton.textContent = "Run again";
    } catch (error) {
      if (!(error instanceof Error && error.name === "ModelInputError")) console.error(`Deep learning ${kind}:`, error);
      if (!current()) return;
      const message = error instanceof Error && error.name === "ModelInputError" ? error.message : "The model could not run. Check your connection and available memory, then retry.";
      setState("Error", message);
      status.classList.add("dl-error");
      runButton.textContent = "Retry";
    } finally {
      busy = false;
      runButton.disabled = false;
      if (loadButton) { loadButton.disabled = ready; loadButton.textContent = ready ? "Model loaded" : "Load model"; }
      if (workspace.isConnected && runRevision !== revision) setState(ready ? "Ready" : "Not loaded", "Input changed. Run it to capture a new trace.");
    }
  };
  runButton.onclick = () => { void execute(true); };
  if (loadButton) loadButton.onclick = () => { void execute(false); };
  input.onkeydown = event => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); void execute(true); }
  };
  window.addEventListener("hashchange", () => { revision++; disposeTrace?.(); }, { once: true });
}

function bindTrace(root: HTMLElement, trace: ExecutionTrace, typesetMath: TypesetMath, outputEl?: HTMLElement): () => void {
  const sliceSize = 32;
  let selected = 0;
  let tensorIndex = 0;
  let tensorName = "";
  let axes: number[] = [];
  let page = 0;
  let coordinate = 0;
  let sourceIndex = 0;
  let candidateIndex = 0;
  let temperature = 1.0;
  let playing = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const stages = [...new Set(trace.steps.map(step => step.stage))];
  const stepGroups = groupExecutionSteps(trace.steps);
  const summaries = new WeakMap<TraceTensor, ReturnType<typeof statistics>>();
  const summary = (tensor: TraceTensor) => {
    let value = summaries.get(tensor);
    if (!value) { value = statistics(tensor.values); summaries.set(tensor, value); }
    return value;
  };
  root.innerHTML = `<section class="dl-execution" aria-label="Recorded forward pass">
    <div class="dl-transport"><button type="button" data-restart aria-label="Restart walkthrough (R)" title="Restart walkthrough (R)">↺</button><button type="button" data-prev aria-label="Previous step ([ or ←)" title="Previous step ([ or ←)">← Previous</button><button type="button" data-play aria-label="Play / Pause (Space)" title="Play / Pause (Space)">Play</button><button type="button" data-next aria-label="Next step (] or →)" title="Next step (] or →)">Next →</button><label class="dl-position"><span></span><input id="execution-step" aria-label="Execution step" type="range" min="0" max="${trace.steps.length - 1}" value="0"></label></div>
    <nav class="dl-stages" aria-label="Execution stages">${stages.map(stage => `<button type="button" data-stage="${escapeHtml(stage)}">${escapeHtml(stage)}</button>`).join("")}</nav>
    <div class="dl-layout"><nav class="dl-step-list" aria-label="Forward pass steps">${stepGroups.map((group, groupIndex) => `<div class="dl-step-group" data-group="${groupIndex}" data-stage="${escapeHtml(group.stage)}"><button type="button" class="dl-group-header" data-group-toggle="${groupIndex}" aria-expanded="false"><span class="dl-group-chevron" aria-hidden="true">▾</span><span class="dl-group-title">${escapeHtml(group.stage)}</span><span class="dl-group-count">${group.steps.length}</span></button><div class="dl-group-items">${group.steps.map(({ step, index }) => `<button type="button" data-step="${index}"><small>${String(index + 1).padStart(2, "0")}</small><span>${escapeHtml(step.name)}</span></button>`).join("")}</div></div>`).join("")}</nav><section class="dl-detail" aria-label="Current operation"></section></div>
    ${trace.note ? `<p class="dl-note">${escapeHtml(trace.note)}</p>` : ""}</section>`;
  const detail = root.querySelector<HTMLElement>(".dl-detail")!;
  const slider = root.querySelector<HTMLInputElement>("#execution-step")!;
  const playButton = root.querySelector<HTMLButtonElement>("[data-play]")!;
  const stepNow = () => trace.steps[selected];
  const tensorNow = () => stepNow().tensors[tensorIndex];
  const stop = () => { playing = false; clearTimeout(timer); playButton.textContent = "Play"; playButton.setAttribute("aria-pressed", "false"); };
  const restoreFocus = (selector: string | undefined) => { if (selector) root.querySelector<HTMLElement>(selector)?.focus({ preventScroll: true }); };
  const resetCandidateIndex = () => {
    const winningCandidateIndex = stepNow().probabilities?.findIndex(candidate => candidate.selected) ?? -1;
    candidateIndex = winningCandidateIndex >= 0 ? winningCandidateIndex : 0;
  };
  const move = (index: number, focus?: string) => {
    selected = Math.max(0, Math.min(trace.steps.length - 1, index));
    const matching = stepNow().tensors.findIndex(tensor => tensor.name === tensorName);
    tensorIndex = Math.max(0, matching);
    resetCandidateIndex();
    if (selected === trace.steps.length - 1) stop();
    drawStep();
    restoreFocus(focus);
  };
  const transport = () => {
    root.querySelector<HTMLButtonElement>("[data-prev]")!.disabled = selected === 0;
    root.querySelector<HTMLButtonElement>("[data-next]")!.disabled = selected === trace.steps.length - 1;
    root.querySelector<HTMLElement>(".dl-position span")!.textContent = `Step ${selected + 1} / ${trace.steps.length}`;
    slider.value = String(selected);
    slider.setAttribute("aria-valuetext", `${selected + 1}: ${stepNow().name}`);

    const activeGroupIndex = stepGroups.findIndex(group => selected >= group.startIndex && selected <= group.endIndex);
    root.querySelectorAll<HTMLElement>(".dl-step-group").forEach((groupEl, index) => {
      const isCurrentGroup = index === activeGroupIndex;
      const header = groupEl.querySelector<HTMLButtonElement>(".dl-group-header");
      const items = groupEl.querySelector<HTMLElement>(".dl-group-items");
      if (header && items) {
        header.setAttribute("aria-expanded", String(isCurrentGroup));
        items.hidden = !isCurrentGroup;
      }
      if (isCurrentGroup) {
        groupEl.setAttribute("data-active-group", "true");
      } else {
        groupEl.removeAttribute("data-active-group");
      }
    });

    root.querySelectorAll<HTMLButtonElement>("[data-step]").forEach(button => {
      if (Number(button.dataset.step) === selected) button.setAttribute("aria-current", "step"); else button.removeAttribute("aria-current");
    });
    root.querySelectorAll<HTMLButtonElement>("[data-stage]").forEach(button => {
      if (button.dataset.stage === stepNow().stage) button.setAttribute("aria-current", "step"); else button.removeAttribute("aria-current");
    });
    const list = root.querySelector<HTMLElement>(".dl-step-list")!;
    const active = list.querySelector<HTMLElement>('[aria-current="step"]');
    if (active) {
      if (list.scrollHeight > list.clientHeight) list.scrollTop = active.offsetTop - list.offsetTop - list.clientHeight / 2;
      if (list.scrollWidth > list.clientWidth) list.scrollLeft = active.offsetLeft - list.offsetLeft - list.clientWidth / 2;
    }
    playButton.textContent = playing ? "Pause" : selected === trace.steps.length - 1 ? "Replay" : "Play";
    playButton.setAttribute("aria-pressed", String(playing));

    if (outputEl) {
      const strongEl = outputEl.querySelector("strong");
      if (strongEl) {
        if (selected === trace.steps.length - 1) {
          strongEl.textContent = trace.output;
        } else {
          const currentStage = stepNow().stage;
          if (currentStage === "Input" || currentStage === "Embedding" || currentStage === "Encoder" || currentStage === "Memory") {
            strongEl.textContent = "…";
          } else {
            const step = stepNow();
            if (step.stage === "Output") {
              strongEl.textContent = trace.output;
            } else if (step.tokens && step.tokens.length > 0) {
              const emittedTokens = step.tokens.filter(t => t.text !== "[start]" && t.text !== "[pad]");
              if (emittedTokens.length > 0) {
                strongEl.textContent = emittedTokens.map(t => t.text).join("");
              } else {
                strongEl.textContent = "…";
              }
            } else {
              strongEl.textContent = "…";
            }
          }
        }
      }
    }
  };
  const tensorChoices = (step: ExecutionStep) => `<div class="dl-flow" aria-label="Tensors in this operation">${step.tensors.map((tensor, index) => {
    const stats = summary(tensor);
    const scale = Math.max(Math.abs(stats.min), Math.abs(stats.max));
    return `<button type="button" class="dl-tensor-choice" data-tensor="${index}" aria-pressed="${tensorIndex === index}"><span>${escapeHtml(tensor.name)}</span><small>${shape(tensor)}</small><span class="dl-mini-vector" aria-hidden="true" title="First ${Math.min(32, tensor.values.length)} coordinates">${Array.from(tensor.values.slice(0, 32)).map(value => `<i style="background:${cellColor(value, scale)}"></i>`).join("")}</span></button>`;
  }).join("")}</div>`;
  const computeProbabilitiesWithTemp = (step: ExecutionStep, temp: number): { candidates: Candidate[]; rawProbabilities: number[] } => {
    const list = step.probabilities ?? [];
    if (!list.length) return { candidates: [], rawProbabilities: [] };
    const hasLogits = list.some(c => c.logit !== undefined);
    if (!hasLogits || temp === 1.0) {
      return { candidates: list, rawProbabilities: list.map(c => c.probability) };
    }
    const safeTemp = Math.max(0.05, temp);
    const logits = list.map(c => c.logit ?? 0);
    const maxLogit = Math.max(...logits);
    const exps = logits.map(l => Math.exp((l - maxLogit) / safeTemp));
    const sumExps = exps.reduce((acc, v) => acc + v, 0);
    const newProbs = exps.map(v => sumExps > 0 ? v / sumExps : 0);
    const updatedCandidates: Candidate[] = list.map((c, i) => ({
      ...c,
      probability: newProbs[i] ?? c.probability,
    }));
    return { candidates: updatedCandidates, rawProbabilities: newProbs };
  };

  const candidates = (step: ExecutionStep) => {
    if (!step.probabilities?.length) return "";
    const hasLogits = step.probabilities.some(candidate => candidate.logit !== undefined);
    const { candidates: currentCandidates } = computeProbabilitiesWithTemp(step, temperature);
    return `<section class="dl-candidates">
      <header class="dl-candidates-header">
        <div>
          <h3>Output distribution <small>Top ${step.probabilities.length} candidates</small></h3>
          ${hasLogits ? `<p class="dl-candidates-formula">Softmax with temperature: <code>p_i = \\exp(z_i / \\tau) / \\sum_j \\exp(z_j / \\tau)</code></p>` : ""}
        </div>
        ${hasLogits ? `
        <div class="dl-temperature-control" aria-label="Decoding temperature">
          <label for="dl-temperature">Temperature &tau; = <strong id="dl-temp-val">${temperature.toFixed(2)}</strong></label>
          <input type="range" id="dl-temperature" min="0.10" max="2.00" step="0.05" value="${temperature.toFixed(2)}">
          <button type="button" data-temp-reset title="Reset temperature to 1.0">Reset</button>
        </div>` : ""}
      </header>
      <div class="dl-table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Token</th>
              <th scope="col">ID</th>
              ${hasLogits ? '<th scope="col">Logit</th>' : ""}
              <th scope="col">Probability ${hasLogits && temperature !== 1.0 ? `(&tau; = ${temperature.toFixed(2)})` : ""}</th>
              <th scope="col">Decision</th>
            </tr>
          </thead>
          <tbody>
            ${currentCandidates.map((candidate, index) => `<tr>
              <th scope="row"><button type="button" data-candidate="${index}" aria-pressed="${candidateIndex === index}">${escapeHtml(visibleToken(candidate.token))}</button></th>
              <td>${candidate.id ?? '<span class="dl-empty-val" aria-label="Not applicable">—</span>'}</td>
              ${hasLogits ? `<td>${candidate.logit === undefined ? '<span class="dl-empty-val" aria-label="Not applicable">—</span>' : number(candidate.logit)}</td>` : ""}
              <td><span class="dl-probability-track"><span class="dl-probability-fill" style="width:${candidate.probability * 100}%"></span></span>${(candidate.probability * 100).toFixed(3)}%</td>
              <td>${candidate.selected ? "✓ Selected (Greedy)" : ""}</td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>
      <div class="dl-candidate-detail dl-value-detail" aria-live="polite"></div>
    </section>`;
  };

  const drawCandidate = () => {
    const step = stepNow();
    const { candidates: currentCandidates } = computeProbabilitiesWithTemp(step, temperature);
    const candidate = currentCandidates[candidateIndex];
    const container = detail.querySelector<HTMLElement>(".dl-candidate-detail");
    if (!candidate || !container) return;
    container.innerHTML = `<strong>${escapeHtml(visibleToken(candidate.token))}</strong><span>p = <code>${candidate.probability.toFixed(6)}</code>${candidate.logit === undefined ? "" : ` · logit = <code>${candidate.logit}</code>`}${temperature !== 1.0 ? ` · &tau; = <code>${temperature.toFixed(2)}</code>` : ""}</span>`;
    detail.querySelectorAll<HTMLButtonElement>("[data-candidate]").forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.candidate) === candidateIndex)));
  };
  const attention = (step: ExecutionStep) => {
    const map = step.attention;
    if (!map) return "";
    sourceIndex = Math.min(sourceIndex, map.source.length - 1);
    const activeWeights = map.weights[map.row] ?? [];
    let maxWeight = -1;
    let peakColumn = 0;
    activeWeights.forEach((w, col) => {
      if (w > maxWeight) {
        maxWeight = w;
        peakColumn = col;
      }
    });
    const peakPct = (maxWeight * 100).toFixed(1);
    const attendedSourceToken = map.source[peakColumn] ?? "";
    const generatedChar = map.target[map.row] ?? "";

    return `<section class="dl-attention">
      <header class="dl-attention-header">
        <div>
          <h3>Attention alignment <small>Step ${map.row + 1}: output “${escapeHtml(visibleToken(generatedChar))}”</small></h3>
          <p class="dl-attention-callout">Top attended input position is <strong>${peakColumn} (“${escapeHtml(visibleToken(attendedSourceToken))}”)</strong> with <strong>${peakPct}%</strong> attention weight.</p>
        </div>
      </header>
      <div class="dl-attention-source-strip" aria-label="Input tokens weighted by attention">
        <span class="dl-strip-label">Input focus:</span>
        <div class="dl-source-chips">
          ${map.source.map((token, col) => {
            const w = activeWeights[col] ?? 0;
            const isPeak = col === peakColumn;
            const isSelected = col === sourceIndex;
            return `<button type="button" class="dl-source-chip ${isPeak ? "is-peak" : ""} ${isSelected ? "is-selected" : ""}" data-source="${col}" aria-pressed="${isSelected}" title="Source ${col} (${escapeHtml(visibleToken(token))}): ${(w * 100).toFixed(1)}% weight"><small>${col}</small><strong>${escapeHtml(visibleToken(token))}</strong><span class="dl-chip-weight">${(w * 100).toFixed(0)}%</span></button>`;
          }).join("")}
        </div>
      </div>
      <div class="dl-attention-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Output</th>
              ${map.source.map((token, column) => `<th scope="col" class="${column === peakColumn ? "is-peak-col" : ""}"><button type="button" data-source="${column}" aria-pressed="${column === sourceIndex}"><small>${column}</small>${escapeHtml(visibleToken(token))}</button></th>`).join("")}
            </tr>
          </thead>
          <tbody>
            ${map.weights.map((row, rowIndex) => `<tr class="${rowIndex === map.row ? "is-active" : ""}">
              <th scope="row"><button type="button" data-row="${rowIndex}" aria-pressed="${rowIndex === map.row}"><small>${rowIndex}</small>${escapeHtml(visibleToken(map.target[rowIndex] ?? ""))}</button></th>
              ${row.map((weight, column) => {
                const isCurrentPeak = rowIndex === map.row && column === peakColumn;
                return `<td class="${column === sourceIndex ? "is-source" : ""} ${isCurrentPeak ? "is-peak-cell" : ""}"><button type="button" data-cell="${rowIndex},${column}" tabindex="${rowIndex === map.row && column === sourceIndex ? 0 : -1}" aria-pressed="${rowIndex === map.row && column === sourceIndex}" aria-label="Output ${rowIndex} ${escapeHtml(visibleToken(map.target[rowIndex] ?? ""))}, source ${column} ${escapeHtml(visibleToken(map.source[column]))}, weight ${weight}" style="background:${cellColor(weight, 1)};color:${weight > .55 ? "white" : "var(--ink)"}">${weight.toFixed(2)}</button></td>`;
              }).join("")}
            </tr>`).join("")}
          </tbody>
        </table>
      </div>
      <div class="dl-attention-detail dl-value-detail" aria-live="polite"></div>
      <p class="dl-note">Colours represent normalized attention weights α ∈ [0, 1]. Cells are rounded for display; select any cell to inspect its exact floating-point value.</p>
    </section>`;
  };
  const drawAttentionValue = (row?: number, column = sourceIndex) => {
    const map = stepNow().attention;
    const container = detail.querySelector<HTMLElement>(".dl-attention-detail");
    if (!map || !container) return;
    const selectedRow = row ?? map.row;
    container.innerHTML = `<strong>${escapeHtml(visibleToken(map.source[column]))} → ${escapeHtml(visibleToken(map.target[selectedRow] ?? ""))}</strong><span>α[${selectedRow}, ${column}] = <code>${map.weights[selectedRow][column]}</code></span><span>Row sum = <code>${map.weights[selectedRow].reduce((sum, value) => sum + value, 0)}</code></span>`;
  };
  const drawRecurrence = () => {
    const step = stepNow();
    const container = detail.querySelector<HTMLElement>(".dl-recurrence");
    const memory = step.tensors.find(tensor => tensor.name === "Cell memory");
    if (!container || !memory || step.operation !== "LSTM cell") return;
    const unit = coordinate % memory.values.length;
    const value = (name: string) => step.tensors.find(tensor => tensor.name === name)!.values[unit];
    const term = (name: string, symbol: string) => {
      const index = step.tensors.findIndex(tensor => tensor.name === name);
      return `<button type="button" data-tensor="${index}" aria-pressed="${index === tensorIndex}" title="${escapeHtml(name)} [${unit}] = ${value(name)}"><small>${symbol} · ${escapeHtml(name)}</small><strong>${number(value(name))}</strong></button>`;
    };

    const meanTensor = (name: string) => {
      const t = step.tensors.find(tensor => tensor.name === name);
      if (!t || !t.values.length) return 0;
      let sum = 0;
      for (let i = 0; i < t.values.length; i++) sum += t.values[i];
      return sum / t.values.length;
    };
    const avgForget = meanTensor("Forget gate");
    const avgInput = meanTensor("Input gate");
    const avgOutput = meanTensor("Output gate");

    container.innerHTML = `<header class="dl-recurrence-header">
      <div>
        <strong>Memory update</strong>
        <span>Unit ${unit} of ${memory.values.length}</span>
      </div>
      <div class="dl-gate-gauges" aria-label="Mean gate activations">
        <span class="dl-gate-gauge" title="Average forget gate activation across all ${memory.values.length} units: higher means stronger retention">
          <small>Memory retention (f&#772;)</small>
          <strong>${(avgForget * 100).toFixed(1)}%</strong>
        </span>
        <span class="dl-gate-gauge" title="Average input gate activation across all ${memory.values.length} units: higher means stronger candidate absorption">
          <small>Input absorption (i&#772;)</small>
          <strong>${(avgInput * 100).toFixed(1)}%</strong>
        </span>
        <span class="dl-gate-gauge" title="Average output gate activation across all ${memory.values.length} units: exposure of tanh(c) to h">
          <small>Hidden exposure (o&#772;)</small>
          <strong>${(avgOutput * 100).toFixed(1)}%</strong>
        </span>
      </div>
    </header>
    <div class="dl-memory-lane"><span class="dl-lane-label">Cell memory</span>${term("Previous cell memory", "cₜ₋₁")}<b>×</b>${term("Forget gate", "fₜ")}<b>+</b>${term("Input gate", "iₜ")}<b>×</b>${term("Candidate memory", "c̃ₜ")}<b>→</b>${term("Cell memory", "cₜ")}</div>
    <div class="dl-hidden-lane"><span class="dl-lane-label">Exposed state</span>${term("Output gate", "oₜ")}<b>× tanh(</b>${term("Cell memory", "cₜ")}<b>) →</b>${term("Hidden state", "hₜ")}<span class="dl-memory-delta">Δ c[${unit}] = <code>${number(value("Cell memory") - value("Previous cell memory"))}</code></span></div>`;
  };
  const alignSelection = () => {
    for (const selector of [".dl-step-list", ".dl-token-strip", ".dl-flow"]) {
      const container = root.querySelector<HTMLElement>(selector);
      const active = container?.querySelector<HTMLElement>('[aria-current="step"], [aria-pressed="true"]');
      if (container && active && container.scrollWidth > container.clientWidth) container.scrollLeft = active.offsetLeft - container.offsetLeft - container.clientWidth / 2 + active.clientWidth / 2;
    }
  };
  window.addEventListener("resize", alignSelection);
  function drawStep(): void {
    const step = stepNow();
    transport();
    detail.innerHTML = `<header class="dl-step-heading"><div><small>${escapeHtml(step.stage)}</small><h2>${escapeHtml(step.name)}</h2></div><code>${escapeHtml(step.operation)}</code></header>
      ${step.tokens?.length ? `<div class="dl-token-strip" aria-label="Tokens">${step.tokens.map((token, index) => `<button type="button" data-token="${index}" aria-pressed="${index === step.selectedToken}"><small>${token.position ?? index}${token.id === undefined ? "" : ` · ID ${token.id}`}</small>${escapeHtml(visibleToken(token.text))}</button>`).join("")}</div>` : ""}
      <div class="dl-math">${step.formula ? `<div class="dl-equation" data-latex="${escapeHtml(step.formula)}" data-display="true"></div>` : ""}<p>${escapeHtml(step.description)}</p></div>
      ${step.operation === "LSTM cell" ? '<section class="dl-recurrence" aria-label="Memory calculation"></section>' : ""}${attention(step)}${candidates(step)}${tensorChoices(step)}<section class="dl-inspector" aria-label="Tensor inspector"></section>`;
    drawInspector();
    drawCandidate();
    drawAttentionValue();
    alignSelection();
    void typesetMath(detail);
  }
  function drawInspector(): void {
    const tensor = tensorNow();
    const container = detail.querySelector<HTMLElement>(".dl-inspector")!;
    if (!tensor) { container.innerHTML = ""; return; }
    tensorName = tensor.name;
    detail.querySelectorAll<HTMLButtonElement>(".dl-flow [data-tensor]").forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.tensor) === tensorIndex)));
    const dimensions = tensor.shape.slice(0, -1);
    axes = dimensions.map((size, index) => Math.min(axes[index] ?? 0, size - 1));
    const width = tensor.shape.at(-1) ?? 1;
    page = Math.min(page, Math.floor((width - 1) / sliceSize));
    const offset = axes.reduce((index, value, axis) => index * dimensions[axis] + value, 0) * width;
    const start = offset + page * sliceSize;
    const end = Math.min(offset + width, start + sliceSize);
    coordinate = Math.max(start, Math.min(end - 1, coordinate));
    const stats = summary(tensor);
    const scale = Math.max(Math.abs(stats.min), Math.abs(stats.max));
    const step = stepNow();
    const formatAxisOption = (axis: number, idx: number): string => {
      const axisName = (tensor.axes?.[axis] ?? "").toLowerCase();
      if (axisName.includes("head")) {
        return `Head ${idx}`;
      }
      if (axisName.includes("position") || axisName.includes("source") || axisName.includes("target")) {
        const token = step.tokens?.[idx]?.text;
        if (token !== undefined) {
          const clean = visibleToken(token);
          return `${idx} (${clean.length > 8 ? clean.slice(0, 7) + "…" : clean})`;
        }
        return `Pos ${idx}`;
      }
      return `${idx}`;
    };
    container.innerHTML = `<header><div><small>Tensor inspector</small><h3>${escapeHtml(tensor.name)}</h3></div><code>${shape(tensor)}${tensor.dtype ? ` · ${escapeHtml(tensor.dtype)}` : ""}</code></header>${tensor.description ? `<p>${escapeHtml(tensor.description)}</p>` : ""}
      <div class="dl-axis-controls">${dimensions.map((size, axis) => `<label>${escapeHtml(tensor.axes?.[axis] ?? `Axis ${axis}`)}<select data-axis="${axis}" ${size === 1 ? "disabled" : ""}>${Array.from({ length: size }, (_, index) => `<option value="${index}" ${axes[axis] === index ? "selected" : ""}>${escapeHtml(formatAxisOption(axis, index))}</option>`).join("")}</select></label>`).join("")}<span>${escapeHtml(tensor.axes?.at(-1) ?? "Last axis")} · ${page * sliceSize}–${end - offset - 1} / ${width}</span><label>Flat index<input id="tensor-index" type="number" min="0" max="${tensor.values.length - 1}" value="${coordinate}" step="1"></label><button type="button" data-jump>Inspect index</button></div>
      <dl class="dl-statistics"><div><dt>Minimum</dt><dd title="${stats.min}">${number(stats.min)}</dd></div><div><dt>Maximum</dt><dd title="${stats.max}">${number(stats.max)}</dd></div><div><dt>Mean</dt><dd title="${stats.mean}">${number(stats.mean)}</dd></div><div><dt>L2 norm</dt><dd title="${stats.norm}">${number(stats.norm)}</dd></div></dl>
      <div class="dl-vector-grid" aria-label="${escapeHtml(tensor.name)} coordinates">${Array.from(tensor.values.slice(start, end)).map((value, index) => `<button type="button" data-coordinate="${start + index}" tabindex="${coordinate === start + index ? 0 : -1}" aria-pressed="${coordinate === start + index}" aria-label="${escapeHtml(tensor.name)} [${coordinates(start + index, tensor.shape).join(", ")}] = ${value}" title="[${coordinates(start + index, tensor.shape).join(", ")}] = ${value}" style="--cell-color:${cellColor(value, scale)}"><span>${page * sliceSize + index}</span></button>`).join("")}</div>
      <div class="dl-value-detail" id="tensor-value" aria-live="polite"></div><div class="dl-tensor-pagination"><button type="button" data-values="-1" ${page === 0 ? "disabled" : ""}>← Coordinates</button><span>Colour relative to max |value| in this tensor</span><button type="button" data-values="1" ${end === offset + width ? "disabled" : ""}>Coordinates →</button></div>
      <details class="dl-raw"><summary>Raw values for this slice · ${end - start} coordinates</summary><p>Stored values below; summary statistics above are rounded and use the whole tensor.</p><div class="dl-table-scroll"><table><thead><tr><th scope="col">Coordinate</th><th scope="col">Value</th></tr></thead><tbody>${Array.from(tensor.values.slice(start, end)).map((value, index) => `<tr><th scope="row">[${coordinates(start + index, tensor.shape).join(", ")}]</th><td>${value}</td></tr>`).join("")}</tbody></table></div></details>`;
    drawCoordinate();
    alignSelection();
  }
  function drawCoordinate(): void {
    const tensor = tensorNow();
    const container = detail.querySelector<HTMLElement>("#tensor-value");
    if (!tensor || !container) return;
    drawRecurrence();
    container.innerHTML = `<strong>[${coordinates(coordinate, tensor.shape).join(", ")}]</strong><code>${tensor.values[coordinate]}</code><span>Stored value</span>`;
    detail.querySelectorAll<HTMLButtonElement>("[data-coordinate]").forEach(button => {
      const active = Number(button.dataset.coordinate) === coordinate;
      button.setAttribute("aria-pressed", String(active));
      button.tabIndex = active ? 0 : -1;
    });
  }
  const jumpToIndex = () => {
    const input = detail.querySelector<HTMLInputElement>("#tensor-index")!;
    if (!input.value || !Number.isFinite(input.valueAsNumber)) return;
    const tensor = tensorNow();
    coordinate = Math.max(0, Math.min(tensor.values.length - 1, Math.trunc(input.valueAsNumber)));
    const indices = coordinates(coordinate, tensor.shape);
    axes = indices.slice(0, -1); page = Math.floor(indices.at(-1)! / sliceSize);
    drawInspector(); restoreFocus("#tensor-index");
  };
  root.onclick = event => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button");
    if (!button || button.disabled) return;
    if (button.hasAttribute("data-play")) {
      if (playing) { stop(); return; }
      if (selected === trace.steps.length - 1) move(0);
      playing = true;
      transport();
      const tick = () => { if (!root.isConnected || !playing) return; move(selected + 1); if (playing) timer = setTimeout(tick, 1400); };
      timer = setTimeout(tick, 1400);
      return;
    }
    stop();
    if (button.hasAttribute("data-jump")) jumpToIndex();
    else if (button.dataset.groupToggle !== undefined) {
      const isExpanded = button.getAttribute("aria-expanded") === "true";
      button.setAttribute("aria-expanded", String(!isExpanded));
      const items = button.nextElementSibling as HTMLElement | null;
      if (items) {
        items.hidden = isExpanded;
      }
    }
    else if (button.hasAttribute("data-next")) move(selected + 1, selected + 1 === trace.steps.length - 1 ? "[data-prev]" : "[data-next]");
    else if (button.hasAttribute("data-prev")) move(selected - 1, selected === 1 ? "[data-next]" : "[data-prev]");
    else if (button.hasAttribute("data-restart")) move(0);
    else if (button.dataset.step !== undefined) move(Number(button.dataset.step));
    else if (button.dataset.stage) move(trace.steps.findIndex(step => step.stage === button.dataset.stage));
    else if (button.dataset.tensor !== undefined) {
      tensorIndex = Number(button.dataset.tensor); axes = []; page = 0; coordinate = 0;
      detail.querySelectorAll<HTMLButtonElement>("[data-tensor]").forEach(item => item.setAttribute("aria-pressed", String(Number(item.dataset.tensor) === tensorIndex)));
      drawInspector();
      restoreFocus(`[data-tensor="${tensorIndex}"]`);
    } else if (button.dataset.coordinate !== undefined) { coordinate = Number(button.dataset.coordinate); drawCoordinate(); }
    else if (button.dataset.values) { page += Number(button.dataset.values); drawInspector(); restoreFocus(`[data-values="${button.dataset.values}"]:not(:disabled)`); }
    else if (button.dataset.candidate !== undefined) {
      candidateIndex = Number(button.dataset.candidate);
      const logitsIndex = stepNow().tensors.findIndex(tensor => tensor.name === "Vocabulary logits");
      const id = stepNow().probabilities?.[candidateIndex].id;
      if (logitsIndex >= 0 && id !== undefined) {
        tensorIndex = logitsIndex; coordinate = id; page = Math.floor(id / sliceSize); axes = [];
        drawInspector();
      }
      drawCandidate();
    }
    else if (button.dataset.token !== undefined) {
      const index = Number(button.dataset.token);
      const step = stepNow();
      const target = trace.steps.findIndex(candidate => candidate.stage === step.stage && candidate.operation === step.operation && candidate.selectedToken !== undefined && candidate.tokens?.[candidate.selectedToken]?.position === step.tokens?.[index].position);
      if (target >= 0) move(target, `[data-token="${index}"]`);
      else {
        const tensor = tensorNow();
        const axis = tensor?.axes?.findIndex(label => label.includes("position"));
        const position = step.tokens?.[index].position ?? index;
        if (axis !== undefined && axis >= 0) {
          if (axis === tensor.shape.length - 1) { page = Math.floor(position / sliceSize); coordinate = position; }
          else { axes[axis] = position; page = 0; }
          drawInspector();
        }
        detail.querySelectorAll<HTMLButtonElement>("[data-token]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
      }
    } else if (button.dataset.row !== undefined || button.dataset.cell) {
      const [row, column] = button.dataset.cell ? button.dataset.cell.split(",").map(Number) : [Number(button.dataset.row), sourceIndex];
      sourceIndex = column;
      const target = trace.steps.findIndex(step => step.stage === "Decoder" && step.attention?.row === row);
      if (target >= 0) move(target, button.dataset.cell ? `[data-cell="${row},${column}"]` : `[data-row="${row}"]`);
    } else if (button.dataset.source !== undefined) {
      sourceIndex = Number(button.dataset.source);
      drawStep(); restoreFocus(`[data-source="${sourceIndex}"]`);
    } else if (button.hasAttribute("data-temp-reset")) {
      temperature = 1.0;
      drawStep();
      restoreFocus("#dl-temperature");
    }
  };
  root.oninput = event => {
    const target = event.target as HTMLElement;
    if (target === slider) { stop(); move(Number(slider.value)); }
    else if (target.id === "dl-temperature") {
      const input = target as HTMLInputElement;
      temperature = Number(input.value);
      const valLabel = detail.querySelector<HTMLElement>("#dl-temp-val");
      if (valLabel) valLabel.textContent = temperature.toFixed(2);
      const step = stepNow();
      const { candidates: currentCandidates } = computeProbabilitiesWithTemp(step, temperature);
      const rows = detail.querySelectorAll<HTMLTableRowElement>(".dl-candidates tbody tr");
      currentCandidates.forEach((c, idx) => {
        const row = rows[idx];
        if (row) {
          const trackFill = row.querySelector<HTMLElement>(".dl-probability-fill");
          const probCell = row.cells[row.cells.length - 2];
          if (trackFill) trackFill.style.width = `${c.probability * 100}%`;
          if (probCell) {
            probCell.innerHTML = `<span class="dl-probability-track"><span class="dl-probability-fill" style="width:${c.probability * 100}%"></span></span>${(c.probability * 100).toFixed(3)}%`;
          }
        }
      });
      const thProb = detail.querySelector<HTMLTableCellElement>(".dl-candidates thead th:nth-last-child(2)");
      if (thProb) {
        thProb.textContent = `Probability ${temperature !== 1.0 ? `(τ = ${temperature.toFixed(2)})` : ""}`;
      }
      drawCandidate();
    }
  };
  root.onchange = event => {
    const select = event.target as HTMLSelectElement;
    if (select.dataset.axis !== undefined) { axes[Number(select.dataset.axis)] = Number(select.value); page = 0; drawInspector(); restoreFocus(`[data-axis="${select.dataset.axis}"]`); }
  };
  const onFocus = (event: FocusEvent) => {
    const button = event.target as HTMLElement;
    if (button.dataset.coordinate !== undefined) { coordinate = Number(button.dataset.coordinate); drawCoordinate(); }
    if (button.dataset.cell) { const [row, column] = button.dataset.cell.split(",").map(Number); drawAttentionValue(row, column); }
  };
  root.addEventListener("focusin", onFocus);
  root.onmouseover = event => {
    const cell = (event.target as HTMLElement).closest<HTMLElement>("[data-cell]");
    if (cell?.dataset.cell) { const [row, column] = cell.dataset.cell.split(",").map(Number); drawAttentionValue(row, column); }
  };
  root.onkeydown = event => {
    const target = event.target as HTMLElement;
    if (target.id === "tensor-index" && event.key === "Enter") { event.preventDefault(); jumpToIndex(); return; }
    if (target.matches("input, select, textarea")) return;
    if (target.dataset.coordinate !== undefined && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
      event.preventDefault();
      const grid = detail.querySelector<HTMLElement>(".dl-vector-grid")!;
      const columns = getComputedStyle(grid).gridTemplateColumns.split(" ").length;
      const delta = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : event.key === "ArrowUp" ? -columns : columns;
      detail.querySelector<HTMLElement>(`[data-coordinate="${coordinate + delta}"]`)?.focus();
    } else if (target.dataset.cell && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
      event.preventDefault();
      const [row, column] = target.dataset.cell.split(",").map(Number);
      const nextRow = row + (event.key === "ArrowUp" ? -1 : event.key === "ArrowDown" ? 1 : 0);
      const nextColumn = column + (event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0);
      detail.querySelector<HTMLElement>(`[data-cell="${nextRow},${nextColumn}"]`)?.focus();
    } else if (!event.altKey && !event.ctrlKey && (event.key === "ArrowLeft" || event.key === "ArrowRight" || event.key === "[" || event.key === "]" || event.code === "KeyJ" || event.code === "KeyK")) {
      event.preventDefault(); stop();
      const delta = (event.key === "ArrowLeft" || event.key === "[" || event.code === "KeyJ") ? -1 : 1;
      move(selected + delta, delta > 0 ? "[data-next]" : "[data-prev]");
    } else if (!event.altKey && !event.ctrlKey && event.code === "Space") {
      event.preventDefault();
      if (playing) {
        stop();
      } else {
        if (selected === trace.steps.length - 1) move(0);
        playing = true;
        transport();
        const tick = () => {
          if (!root.isConnected || !playing) return;
          move(selected + 1);
          if (playing) timer = setTimeout(tick, 1400);
        };
        timer = setTimeout(tick, 1400);
      }
    } else if (!event.altKey && !event.ctrlKey && (event.code === "KeyR" || event.key === "r" || event.key === "R")) {
      event.preventDefault(); stop(); move(0, "[data-restart]");
    }
  };
  resetCandidateIndex();
  drawStep();
  return () => { stop(); window.removeEventListener("resize", alignSelection); root.onclick = null; root.oninput = null; root.onchange = null; root.removeEventListener("focusin", onFocus); root.onmouseover = null; root.onkeydown = null; };
}
