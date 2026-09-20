import type { Topic } from "./data/topics.ts";
import { getTopicIcon } from "./data/icons.ts";

export interface MethodStep { title: string; stage: string; }
export interface MethodTrace {
  result: string;
  summary: string;
  steps: MethodStep[];
  selectedStep?: number;
  render(index: number): { visual: string; detail: string };
  choose?: (name: string, value: string) => MethodTrace;
}

export function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]!);
}

export function math(formula: string): string {
  return `<div class="dl-math" data-latex="${escapeHtml(formula)}" data-display="true"></div>`;
}

async function typeset(root: ParentNode): Promise<void> {
  const elements = root.querySelectorAll<HTMLElement>("[data-latex]");
  if (!elements.length) return;
  const [{ default: katex }] = await Promise.all([import("katex"), import("katex/dist/katex.min.css")]);
  elements.forEach((element) => katex.render(element.dataset.latex ?? "", element, { displayMode: true, throwOnError: false, strict: false }));
}

export function methodPage(topic: Topic, options: {
  title: string; description: string; inputs: string; notes: string;
  examples?: Array<{ label: string; values: Record<string, string> }>;
}): string {
  const isFoundations = topic.slug === "edit-distance" || topic.slug === "tfidf";
  const parentCategorySlug = isFoundations ? "foundations" : "structured";
  const parentCategoryLabel = isFoundations ? "Foundations" : "Structured NLP";
  const links = isFoundations
    ? [["bpe", "BPE"], ["edit-distance", "Edit distance"], ["tfidf", "TF-IDF"]]
    : [["n-grams", "Bigrams"], ["viterbi", "Viterbi"], ["cky", "CKY"]];

  return `<article id="method-workspace" class="ml-workspace dl-workspace">
    <nav class="dl-model-nav" aria-label="${escapeHtml(parentCategoryLabel)}"><a class="dl-back" href="#/${parentCategorySlug}" aria-label="All ${escapeHtml(parentCategoryLabel.toLowerCase())} laboratories">← <span>${parentCategoryLabel}</span></a><div>${links.map(([slug, label]) => `<a href="#/${slug}" ${topic.slug === slug ? 'aria-current="page"' : ""}>${label}</a>`).join("")}</div></nav>
    <header class="dl-page-heading"><div><p class="dl-eyebrow">${escapeHtml(parentCategoryLabel)} / interactive lab</p><div class="dl-heading-title-row">${getTopicIcon(topic.slug, "dl-heading-icon")}<h1>${escapeHtml(options.title)}</h1></div><p>${escapeHtml(options.description)}</p></div><span class="dl-state" id="method-state">Ready to calculate</span></header>
    <section class="dl-experiment" aria-label="Experiment">
      <div class="dl-input-panel"><div class="dl-section-label"><span>Input</span><span>Editable experiment</span></div><div id="method-inputs">${options.inputs}</div>
      ${options.examples?.length ? `<div class="dl-presets"><span>Try</span>${options.examples.map((example) => `<button type="button" data-preset="${escapeHtml(JSON.stringify(example.values))}">${escapeHtml(example.label)}</button>`).join("")}</div>` : ""}
      <div class="dl-run-controls"><button class="dl-primary" id="method-run" type="button">Calculate <span aria-hidden="true">→</span></button><button id="method-reset" type="button">Reset</button></div></div>
      <div class="dl-output-panel"><div class="dl-section-label"><span>Result</span><span>Computed from your input</span></div><div id="method-output" aria-live="polite"><p class="dl-output-empty">Calculate to see the result.</p></div><p class="dl-note">Explore the recorded calculation below. Selecting a step changes the explanation, while the final result stays here.</p></div>
      <div id="method-status" role="status">Set the inputs, then calculate.</div>
    </section>
    <section id="method-trace" aria-label="Calculation"><div class="dl-empty"><p class="dl-eyebrow">Calculation workspace</p><h2>Follow each decision.</h2><p class="dl-note">Run the experiment to inspect its values, compare candidates and trace how the result was obtained.</p></div></section>
    <details class="ml-notes"><summary>Method notes</summary>${options.notes}</details>
  </article>`;
}

export function bindMethodWorkspace(build: () => MethodTrace): void {
  const root = document.querySelector<HTMLElement>("#method-workspace")!;
  const inputs = root.querySelector<HTMLElement>("#method-inputs")!;
  const output = root.querySelector<HTMLElement>("#method-output")!;
  const status = root.querySelector<HTMLElement>("#method-status")!;
  const state = root.querySelector<HTMLElement>("#method-state")!;
  const panel = root.querySelector<HTMLElement>("#method-trace")!;
  const empty = panel.innerHTML;
  const defaults = Array.from(inputs.querySelectorAll<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>("input, textarea, select"), (field) => ({ id: field.id, value: field.value }));
  let trace: MethodTrace | undefined;
  let index = 0;
  let timer: ReturnType<typeof setInterval> | undefined;
  const stop = () => {
    clearInterval(timer);
    timer = undefined;
    const play = panel.querySelector<HTMLButtonElement>("[data-play]");
    if (play) { play.textContent = "Play"; play.setAttribute("aria-pressed", "false"); }
  };
  const invalidate = () => {
    stop(); trace = undefined; index = 0;
    output.innerHTML = '<p class="dl-output-empty">Calculate to see the updated result.</p>';
    panel.innerHTML = empty;
    status.textContent = "Input changed. Calculate to record a new trace.";
    state.textContent = "Input changed";
  };
  const show = () => {
    if (!trace) return;
    const active = document.activeElement as HTMLElement | null;
    const focusedJump = active?.closest(".ml-trace-layout") ? active.dataset.jump : undefined;
    const focusedView = active?.closest<HTMLElement>("[data-view]")?.dataset.view;
    const scrolls = Array.from(panel.querySelectorAll<HTMLElement>(".ml-table-scroll, .cky-scroll, .cky-constituents > div"), (element) => ({ left: element.scrollLeft, top: element.scrollTop }));
    index = Math.max(0, Math.min(trace.steps.length - 1, index));
    const step = trace.steps[index];
    const content = trace.render(index);
    panel.querySelector<HTMLElement>(".ml-visual")!.innerHTML = content.visual;
    panel.querySelector<HTMLElement>(".ml-inspector")!.innerHTML = content.detail;
    panel.querySelector<HTMLElement>("#method-step-title")!.textContent = step.title;
    panel.querySelector<HTMLSelectElement>("#method-operation")!.value = String(index);
    panel.querySelector<HTMLInputElement>("#method-position")!.value = String(index);
    panel.querySelector<HTMLElement>(".dl-position")!.textContent = `${index + 1} / ${trace.steps.length}`;
    panel.querySelector<HTMLButtonElement>('[data-move="-1"]')!.disabled = index === 0;
    panel.querySelector<HTMLButtonElement>('[data-move="1"]')!.disabled = index === trace.steps.length - 1;
    panel.querySelectorAll<HTMLElement>("[data-stage]").forEach((button) => {
      if (button.dataset.stage === step.stage) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    panel.querySelectorAll<HTMLElement>(".ml-table-scroll, .cky-scroll, .cky-constituents > div").forEach((element, position) => {
      element.scrollLeft = scrolls[position]?.left ?? 0;
      element.scrollTop = scrolls[position]?.top ?? 0;
    });
    panel.querySelectorAll<HTMLElement>(".ml-visual .ml-table-scroll, .cky-scroll").forEach((container) => {
      const selected = container.querySelector<HTMLElement>('[aria-pressed="true"]');
      if (!selected) return;
      const bounds = container.getBoundingClientRect();
      const cell = selected.getBoundingClientRect();
      if (cell.left < bounds.left || cell.right > bounds.right) container.scrollLeft += cell.left - bounds.left - container.clientWidth / 2 + cell.width / 2;
      if (cell.top < bounds.top || cell.bottom > bounds.bottom) container.scrollTop += cell.top - bounds.top - container.clientHeight / 2 + cell.height / 2;
    });
    if (focusedJump !== undefined) {
      const replacement = Array.from(panel.querySelectorAll<HTMLElement>("[data-jump]")).find((element) => element.dataset.jump === focusedJump && (!focusedView || element.closest<HTMLElement>("[data-view]")?.dataset.view === focusedView));
      (replacement ?? panel.querySelector<HTMLElement>("#method-operation"))?.focus({ preventScroll: true });
    }
    void typeset(panel);
    if (index === trace.steps.length - 1) stop();
  };
  const mount = () => {
    if (!trace?.steps.length) throw new Error("The calculation did not produce any steps.");
    output.innerHTML = `<output>${escapeHtml(trace.result)}</output><p>${escapeHtml(trace.summary)}</p>`;
    state.textContent = "Calculation ready";
    status.textContent = `${trace.steps.length} recorded steps. Select a cell or use the playback controls to inspect the calculation.`;
    panel.innerHTML = `<div class="dl-execution"><div class="dl-transport"><button type="button" data-restart aria-label="Restart trace">↺</button><button type="button" data-move="-1" aria-label="Previous step">←</button><button type="button" data-play aria-pressed="false">Play</button><button type="button" data-move="1" aria-label="Next step">→</button><label for="method-position">Step <input id="method-position" type="range" min="0" max="${trace.steps.length - 1}" value="${index}"></label><span class="dl-position"></span></div>
      <nav class="dl-stages" aria-label="Calculation stages">${Array.from(new Set(trace.steps.map((step) => step.stage)), (stage) => `<button type="button" data-stage="${escapeHtml(stage)}">${escapeHtml(stage)}</button>`).join("")}</nav>
      <div class="ml-operation-heading"><div><p class="dl-eyebrow">Selected operation</p><h2 id="method-step-title"></h2></div><label>Jump to step<select id="method-operation">${trace.steps.map((step, position) => `<option value="${position}">${position + 1}. ${escapeHtml(step.title)}</option>`).join("")}</select></label></div>
      <div class="ml-trace-layout"><div class="ml-visual"></div><aside class="ml-inspector" aria-label="Selected calculation"></aside></div></div>`;
    show();
  };
  const choose = (name: string, value: string) => {
    if (!trace?.choose) return;
    stop(); trace = trace.choose(name, value); index = trace.selectedStep ?? index; mount();
    const replacement = Array.from(panel.querySelectorAll<HTMLSelectElement | HTMLButtonElement>("[data-choice]")).find((element) => element.dataset.choice === name && (element instanceof HTMLSelectElement || element.value === value));
    replacement?.focus({ preventScroll: true });
  };
  inputs.addEventListener("input", invalidate);
  inputs.addEventListener("change", invalidate);
  root.querySelector("#method-run")!.addEventListener("click", () => {
    stop();
    try { trace = build(); index = 0; mount(); }
    catch (error) {
      trace = undefined; panel.innerHTML = empty;
      output.innerHTML = '<p class="dl-output-empty">Check the input to continue.</p>';
      status.textContent = error instanceof Error ? error.message : "The calculation could not be completed.";
      state.textContent = "Input needs attention";
    }
  });
  const setValues = (values: Record<string, string>) => {
    Object.entries(values).forEach(([id, value]) => {
      const field = document.getElementById(id) as HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement | null;
      if (field && inputs.contains(field)) {
        field.value = value;
        field.dispatchEvent(new Event("input", { bubbles: true }));
        field.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
    invalidate();
  };
  root.querySelector("#method-reset")!.addEventListener("click", () => {
    setValues(Object.fromEntries(defaults.filter((field) => field.id).map((field) => [field.id, field.value])));
    root.dispatchEvent(new Event("method-reset"));
    status.textContent = "Default inputs restored. Calculate to start again.";
    state.textContent = "Ready to calculate";
  });
  root.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((button) => button.addEventListener("click", () => setValues(JSON.parse(button.dataset.preset!))));
  panel.addEventListener("click", (event) => {
    const button = (event.target as Element).closest<HTMLButtonElement>("button");
    if (!button || !trace) return;
    if (button.hasAttribute("data-play")) {
      if (timer) { stop(); return; }
      if (index === trace.steps.length - 1) { index = 0; show(); }
      button.textContent = "Pause"; button.setAttribute("aria-pressed", "true");
      timer = setInterval(() => { if (!root.isConnected) { stop(); return; } index++; show(); }, 1100);
      return;
    }
    stop();
    if (button.dataset.choice) { choose(button.dataset.choice, button.value); return; }
    if (button.dataset.jump !== undefined) index = Number(button.dataset.jump);
    else if (button.dataset.move) index += Number(button.dataset.move);
    else if (button.hasAttribute("data-restart")) index = 0;
    else if (button.dataset.stage) index = trace.steps.findIndex((step) => step.stage === button.dataset.stage);
    else return;
    show();
  });
  panel.addEventListener("input", (event) => {
    const target = event.target as HTMLInputElement;
    if (target.id === "method-position") { stop(); index = Number(target.value); show(); }
  });
  panel.addEventListener("change", (event) => {
    const target = event.target as HTMLSelectElement;
    if (target.id === "method-operation") { stop(); index = Number(target.value); show(); }
    else if (target.dataset.choice) choose(target.dataset.choice, target.value);
  });
  window.addEventListener("hashchange", stop, { once: true });
  void typeset(root);
}
