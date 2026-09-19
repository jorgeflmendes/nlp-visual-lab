import type { DeepLearningTopic } from "./data/topics.ts";
import {
  type BpeModel,
  type BpeOptions,
  type BpeIterationState,
  trainBpe,
  tokenizeWithRules,
  parseCorpus,
} from "./bpeAlgorithm.ts";
import "./bpeLab.css";

function escapeHtml(s: string): string {
  return String(s).replace(/[&<>'"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[c] ?? c
  );
}

function visibleChar(s: string): string {
  return s === " " ? "␠" : s === "\n" ? "↵" : s === "\t" ? "⇥" : s;
}

const routes: [string, string][] = [
  ["bpe", "bpe"],
  ["lstm", "lstm"],
  ["seq2seq", "seq2seq"],
  ["attention", "attention"],
  ["transformers", "transformers"],
];

const labels: Record<string, string> = {
  bpe: "BPE",
  lstm: "LSTM",
  seq2seq: "Seq2Seq",
  attention: "Attention",
  transformers: "Transformer",
};

export function bpePage(topic: DeepLearningTopic): string {
  return `<article class="bpe-workspace dl-workspace" data-kind="bpe">
    <nav class="dl-model-nav" aria-label="Deep learning experiments">
      <a class="dl-back" href="#/deep-learning" aria-label="All deep learning experiments">← <span>Laboratory</span></a>
      <div>${routes.map(([kind, slug]) => `<a href="#/${slug}" ${kind === topic.slug ? 'aria-current="page"' : ""}>${labels[kind] ?? kind}</a>`).join("")}</div>
    </nav>
    <header class="dl-page-heading">
      <div>
        <p class="dl-eyebrow">BPE / Tokenizer Laboratory</p>
        <h1>Train subwords. Inspect every merge.</h1>
        <p>Follow how Byte Pair Encoding iteratively aggregates the most frequent adjacent character and subword pairs into a learned vocabulary.</p>
      </div>
      <span id="bpe-status" class="dl-state">Not trained</span>
    </header>

    <!-- Top experiment panel: Input & Config + Output Preview -->
    <section class="dl-experiment" aria-label="Corpus input and training configuration">
      <div class="dl-input-panel">
        <div class="dl-section-label">
          <label for="bpe-corpus">Training corpus</label>
          <span>01 / Corpus &amp; Configuration</span>
        </div>
        <textarea id="bpe-corpus" rows="4" spellcheck="false">low lower lowest new newer newest wide wider widest</textarea>
        <div class="dl-presets" aria-label="Example corpora">
          <span>Try</span>
          <button id="bpe-example" type="button">Vocabulary expansion</button>
          <button id="bpe-morphology" type="button">Morphology &amp; affixes</button>
          <button id="bpe-clear" type="button">Clear</button>
          <label style="cursor:pointer; color:var(--green); font-size:.74rem; text-decoration:underline; text-decoration-color:#a9c4b9; text-underline-offset:4px; padding-inline:8px;">
            Load file
            <input id="bpe-file" class="bpe-file-input" type="file" accept=".txt,.csv,.md,.tsv,text/plain,text/csv,text/markdown" />
          </label>
        </div>
        <p id="bpe-filename" class="dl-input-hint" style="min-height:1.4em; margin-bottom:10px;"></p>

        <!-- Advanced settings toggleable -->
        <details style="margin-bottom:16px; border:1px solid var(--line); border-radius:6px; background:#0e141a; padding:10px 14px;">
          <summary style="cursor:pointer; font-size:.78rem; font-weight:650; color:var(--green);">Segmentation &amp; Hyperparameters</summary>
          <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:10px; margin-top:12px;">
            <label>
              <span class="bpe-label">Sequences</span>
              <select id="bpe-split" style="font-size:.75rem; min-height:36px; padding:6px 8px;">
                <option value="whitespace">Whitespace</option>
                <option value="lines">Lines</option>
                <option value="comma">Commas</option>
                <option value="custom">Custom regex</option>
                <option value="whole">Whole text</option>
              </select>
            </label>
            <label>
              <span class="bpe-label">Custom regex</span>
              <input id="bpe-custom" type="text" value="[;,]+" disabled style="font-size:.75rem; min-height:36px; padding:6px 8px;" />
            </label>
            <label>
              <span class="bpe-label">Tie break</span>
              <select id="bpe-tie" style="font-size:.75rem; min-height:36px; padding:6px 8px;">
                <option value="first">First observed</option>
                <option value="lexical">Lexicographic</option>
              </select>
            </label>
            <label>
              <span class="bpe-label">Target |V|</span>
              <input id="bpe-target" type="number" min="1" max="4096" value="35" style="font-size:.75rem; min-height:36px; padding:6px 8px;" />
            </label>
            <label>
              <span class="bpe-label">Max merges</span>
              <input id="bpe-max" type="number" min="0" max="120" value="25" style="font-size:.75rem; min-height:36px; padding:6px 8px;" />
            </label>
            <label>
              <span class="bpe-label">Min frequency</span>
              <input id="bpe-min" type="number" min="1" max="999999" value="2" style="font-size:.75rem; min-height:36px; padding:6px 8px;" />
            </label>
          </div>
          <div class="bpe-checks" style="margin-top:10px;">
            <label><input id="bpe-nfc" type="checkbox" checked /> NFC</label>
            <label><input id="bpe-lower" type="checkbox" /> Lowercase</label>
            <label><input id="bpe-punct" type="checkbox" /> Remove punctuation</label>
          </div>
        </details>

        <div class="dl-run-controls">
          <button id="bpe-train" class="dl-primary" type="button">Train tokenizer <span aria-hidden="true">→</span></button>
          <button id="bpe-reset-btn" type="button">Reset</button>
        </div>
      </div>

      <!-- Output Panel -->
      <div class="dl-output-panel">
        <div class="dl-section-label">
          <span>Learned tokenizer summary</span>
          <span>02 / Status &amp; Result</span>
        </div>
        <div id="bpe-output" aria-live="polite">
          <p class="dl-output-empty">Click "Train tokenizer" to learn subwords.</p>
          <p class="dl-note">Configure the corpus and parameters, then train to start the step-by-step merge inspector.</p>
        </div>
        <div class="dl-model-meta">
          <span>Subword Tokenization (BPE)</span>
          <span id="bpe-summary-meta">Client-side execution</span>
        </div>
      </div>

      <div id="model-status" role="status" aria-live="polite">Ready to train. Click "Train tokenizer" to record the merge history.</div>
    </section>

    <!-- Execution Inspector: Empty before training, Full Inspector after training -->
    <div id="bpe-result-container" style="margin-top: 24px;">
      <section class="dl-empty" aria-label="Execution inspector">
        <div>
          <p class="dl-eyebrow">03 / Execution inspector</p>
          <h2>From corpus to learned vocabulary.</h2>
          <p>Inspect the pair leaderboard, follow every merge candidate, and test the resulting tokenizer on unseen text.</p>
        </div>
        <ol class="dl-empty-flow">
          <li>Corpus</li>
          <li>Pairs</li>
          <li>Merges</li>
          <li>Tokenizer</li>
        </ol>
        <p class="dl-note">Train the tokenizer with a corpus to activate the interactive merge inspector.</p>
      </section>
    </div>
    <details class="dl-model-notes"><summary>About Byte Pair Encoding</summary><p>Byte Pair Encoding (BPE) is an iterative subword tokenization algorithm originally introduced for data compression. In neural language models (such as GPT-2, RoBERTa, and LLaMA), BPE begins with basic unicode characters as its initial vocabulary and iteratively merges the most frequent adjacent pair across the entire corpus until reaching a target vocabulary size or maximum merge threshold. Subword tokenization balances character-level granularity (avoiding out-of-vocabulary words) with word-level efficiency.</p></details>
  </article>`;
}

function renderInspectorTemplate(inferText: string = "lowest newer"): string {
  return `
    <section class="dl-execution" aria-label="BPE execution walkthrough">
      <!-- Transport control bar directly matching deep learning standards -->
      <div class="dl-transport">
        <button id="bpe-restart" type="button" aria-label="Restart walkthrough">↺</button>
        <button id="bpe-prev" type="button" aria-label="Previous step" disabled>← Previous</button>
        <button id="bpe-play" type="button" data-play aria-pressed="false">Play</button>
        <button id="bpe-next" type="button" aria-label="Next step" disabled>Next →</button>
        <button id="bpe-final" type="button" aria-label="Final state" disabled>Final</button>
        <label class="dl-position">
          <span id="bpe-step-label">Step 0 / 0</span>
          <input id="bpe-slider" aria-label="Merge step" type="range" min="0" max="0" value="0" />
        </label>
      </div>

      <!-- Stages nav bar -->
      <nav class="dl-stages" aria-label="Walkthrough stages">
        <button id="bpe-d1" type="button" data-stage="1" aria-current="step">1. Characters</button>
        <button id="bpe-d2" type="button" data-stage="2">2. Count pairs</button>
        <button id="bpe-d3" type="button" data-stage="3">3. Merge winner</button>
        <button id="bpe-d4" type="button" data-stage="4">4. Tokenizer</button>
      </nav>

      <!-- Stats row -->
      <div class="bpe-stats-container">
        <div class="dl-statistics">
          <div><dt>SEQUENCES</dt><dd id="bpe-sSeq">—</dd></div>
          <div><dt>CHARACTERS</dt><dd id="bpe-sChar">—</dd></div>
          <div><dt>INITIAL |V|</dt><dd id="bpe-sInit">—</dd></div>
          <div><dt>FINAL |V|</dt><dd id="bpe-sFinal">—</dd></div>
          <div><dt>MERGES LEARNED</dt><dd id="bpe-sMerges">—</dd></div>
          <div><dt>STOP REASON</dt><dd id="bpe-sStop" style="font-family: inherit; font-size: 0.73rem;">—</dd></div>
        </div>
      </div>

      <!-- Main walkthrough split: Corpus evolution + Pair Leaderboard -->
      <div class="bpe-inspector-grid">
        <!-- Left column: Corpus & Current Merge Decision -->
        <div class="bpe-card-clean">
          <div class="bpe-card-header">
            <div>
              <p class="dl-eyebrow" style="margin: 0 0 3px;">Corpus state</p>
              <h3 id="bpe-caption">Initial character-level representation.</h3>
            </div>
            <span id="bpe-iter" class="dl-state">iteration 0</span>
          </div>

          <div id="bpe-corpusState" class="bpe-corpus">
            <div class="bpe-empty">The evolving corpus will appear here.</div>
          </div>
          <p id="bpe-hidden" class="dl-note" style="display:none"></p>

          <div class="bpe-decision-box">
            <div id="bpe-decision-panel">No merge selected yet.</div>
          </div>
        </div>

        <!-- Right column: Candidate pairs leaderboard -->
        <div class="bpe-card-clean bpe-border-left">
          <div class="bpe-card-header">
            <div>
              <p class="dl-eyebrow" style="margin: 0 0 3px;">Candidate pairs</p>
              <h3>Pair leaderboard</h3>
            </div>
            <span class="dl-state">Frequency</span>
          </div>
          <div id="bpe-pairs" class="bpe-pairs-list">
            <div class="bpe-empty">No counts yet.</div>
          </div>
        </div>
      </div>

      <!-- Secondary split: Vocabulary & History vs Test Playground -->
      <div class="bpe-secondary-grid">
        <div class="bpe-card-clean">
          <div class="bpe-card-header">
            <div>
              <p class="dl-eyebrow" style="margin: 0 0 3px;">Evolving vocabulary</p>
              <h3>Learned subwords</h3>
            </div>
            <span id="bpe-vocabPill" class="dl-state">|V| = —</span>
          </div>
          <div id="bpe-vocab" class="bpe-vocab">
            <span class="bpe-hint">Train a tokenizer first.</span>
          </div>

          <div class="bpe-rules-header">
            <p class="dl-eyebrow" style="margin: 0 0 3px;">Merge rules</p>
            <h3>Rules applied to this step</h3>
          </div>
          <div id="bpe-rules" class="bpe-rules">
            <div class="bpe-hint">No rules learned.</div>
          </div>
        </div>

        <div class="bpe-card-clean bpe-border-left">
          <div class="bpe-card-header">
            <div>
              <p class="dl-eyebrow" style="margin: 0 0 3px;">Interactive test</p>
              <h3>Tokenizer playground</h3>
            </div>
            <span id="bpe-count" class="dl-state">— tokens</span>
          </div>

          <div class="bpe-playground-form">
            <div>
              <label for="bpe-infer" class="bpe-label">Text to tokenize</label>
              <textarea id="bpe-infer" rows="2" spellcheck="false">${escapeHtml(inferText)}</textarea>
            </div>
            <div>
              <label for="bpe-inferMode" class="bpe-label">Rules to apply</label>
              <select id="bpe-inferMode">
                <option value="full">Full trained tokenizer</option>
                <option value="current">Current iteration only</option>
              </select>
            </div>
          </div>

          <div style="margin-top: 16px;">
            <span class="bpe-label">Tokenized output</span>
            <div id="bpe-result" class="bpe-result-tokens">
              <div class="bpe-hint">Train a tokenizer to test it.</div>
            </div>
          </div>

          <details class="bpe-trace-details">
            <summary>Step-by-step merge trace on input</summary>
            <div id="bpe-trace" class="bpe-trace"></div>
          </details>
        </div>
      </div>
    </section>`;
}

export function bindBpe(): void {
  const $ = (id: string) => document.getElementById(id);
  const resultContainer = $("bpe-result-container");
  const outputEl = $("bpe-output");
  const statusEl = $("bpe-status");
  const statusMsgEl = $("model-status");
  const summaryMetaEl = $("bpe-summary-meta");
  const corpusInput = $("bpe-corpus") as HTMLTextAreaElement;

  const initialCorpus = corpusInput?.value ?? "";
  const emptyResult = resultContainer?.innerHTML ?? "";
  const emptyOutput = outputEl?.innerHTML ?? "";

  let currentInferText = "lowest newer";
  let model: BpeModel | null = null;
  let step = 0;
  let playing = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const stopPlay = () => {
    playing = false;
    clearTimeout(timer);
    const playBtn = $("bpe-play");
    if (playBtn) {
      playBtn.textContent = "Play";
      playBtn.setAttribute("aria-pressed", "false");
    }
  };

  const invalidate = (message: string = "Corpus or parameters changed. Click 'Train tokenizer' to update.") => {
    stopPlay();
    model = null;
    step = 0;
    if (resultContainer) resultContainer.innerHTML = emptyResult;
    if (outputEl) outputEl.innerHTML = emptyOutput;
    if (statusEl) {
      statusEl.textContent = "Not trained";
      statusEl.dataset.state = "idle";
    }
    if (statusMsgEl) {
      statusMsgEl.textContent = message;
      statusMsgEl.classList.remove("dl-error");
    }
    if (summaryMetaEl) {
      summaryMetaEl.textContent = "Client-side execution";
    }
  };

  const startPlay = () => {
    if (!model || model.states.length <= 1) return;
    playing = true;
    const playBtn = $("bpe-play");
    if (playBtn) {
      playBtn.textContent = "Pause";
      playBtn.setAttribute("aria-pressed", "true");
    }
    const tick = () => {
      if (!playing || !model) return;
      if (step < model.states.length - 1) {
        step++;
        render();
        timer = setTimeout(tick, 900);
      } else {
        stopPlay();
      }
    };
    timer = setTimeout(tick, 900);
  };

  function getOptions(): BpeOptions {
    const splitMode = ($("bpe-split") as HTMLSelectElement).value as BpeOptions["splitMode"];
    const customSeparator = ($("bpe-custom") as HTMLInputElement).value;
    const tieBreak = ($("bpe-tie") as HTMLSelectElement).value as BpeOptions["tieBreak"];
    const normalizeNfc = ($("bpe-nfc") as HTMLInputElement).checked;
    const lowercase = ($("bpe-lower") as HTMLInputElement).checked;
    const removePunctuation = ($("bpe-punct") as HTMLInputElement).checked;
    const targetVocabSize = parseInt(($("bpe-target") as HTMLInputElement).value, 10) || 35;
    const maxMerges = parseInt(($("bpe-max") as HTMLInputElement).value, 10) ?? 25;
    const minPairFrequency = parseInt(($("bpe-min") as HTMLInputElement).value, 10) || 2;

    return {
      splitMode,
      customSeparator,
      tieBreak,
      normalizeNfc,
      lowercase,
      removePunctuation,
      targetVocabSize,
      maxMerges,
      minPairFrequency,
    };
  }

  function renderTokenSpans(tokens: string[], learnedSet: Set<string>): string {
    return tokens
      .map((t) => `<span class="bpe-token${learnedSet.has(t) ? " learned" : ""}">${escapeHtml(visibleChar(t))}</span>`)
      .join('<span class="bpe-sep">·</span>');
  }

  function renderPairs(st: BpeIterationState): void {
    const pairsEl = $("bpe-pairs");
    if (!pairsEl) return;

    if (!st.pairs.length) {
      pairsEl.innerHTML = '<div class="bpe-empty">No adjacent pairs remain.</div>';
      return;
    }

    const top = Math.max(1, st.pairs[0].count);
    pairsEl.innerHTML = st.pairs
      .slice(0, 15)
      .map((p, i) => {
        const isBest = !!st.candidate && i === 0;
        const pct = Math.max(5, Math.round((100 * p.count) / top));
        return `<div class="bpe-pair${isBest ? " best" : ""}">
          <div>
            <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
              <span style="font-family:Consolas,monospace;font-weight:600">${escapeHtml(visibleChar(p.left))}</span>
              <span style="color:var(--muted)">+</span>
              <span style="font-family:Consolas,monospace;font-weight:600">${escapeHtml(visibleChar(p.right))}</span>
              ${isBest ? '<span class="bpe-badge">next</span>' : ""}
            </div>
            <div class="bpe-bar"><div class="bpe-fill" style="width:${pct}%"></div></div>
          </div>
          <div style="text-align:right;font-family:Consolas,monospace;font-weight:700">${p.count}</div>
        </div>`;
      })
      .join("");
  }

  function renderDecision(st: BpeIterationState): void {
    const decEl = $("bpe-decision-panel");
    if (!decEl) return;

    if (st.candidate) {
      const c = st.candidate;
      decEl.innerHTML = `<div class="dl-value-detail">
        <small style="flex-basis: 100%; color: var(--green); font: 0.68rem Consolas, monospace; text-transform: uppercase; letter-spacing: .04em;">Highest-frequency pair</small>
        <span style="font-family: Consolas, monospace; font-size: 1.05rem; font-weight: 700;">
          <span>${escapeHtml(visibleChar(c.left))}</span>
          <span style="color: var(--dl-muted); font-weight: normal;">+</span>
          <span>${escapeHtml(visibleChar(c.right))}</span>
          <span style="color: var(--dl-muted); font-weight: normal;">→</span>
          <span style="color: var(--green);">${escapeHtml(visibleChar(c.left + c.right))}</span>
        </span>
        <code style="margin-left: auto; color: var(--dl-muted);">${c.count} occurrence${c.count === 1 ? "" : "s"} in current sequences</code>
      </div>`;
    } else {
      decEl.innerHTML = `<div class="dl-value-detail">
        <small style="flex-basis: 100%; color: var(--dl-muted); font: 0.68rem Consolas, monospace; text-transform: uppercase; letter-spacing: .04em;">Stopping condition reached</small>
        <strong style="font-size: 0.95rem;">${escapeHtml(st.reason || model?.stopReason || "Training complete")}</strong>
        <span style="margin-left: auto; color: var(--dl-muted); font-size: 0.75rem;">No further merges performed.</span>
      </div>`;
    }
  }

  function renderVocab(st: BpeIterationState): void {
    const vocabPill = $("bpe-vocabPill");
    const vocabEl = $("bpe-vocab");
    const rulesEl = $("bpe-rules");
    if (!vocabPill || !vocabEl || !rulesEl) return;

    const learnedSet = new Set(st.rules.map((r) => r.merged));
    vocabPill.textContent = `|V| = ${st.vocab.length}`;

    vocabEl.innerHTML =
      st.vocab
        .slice(0, 150)
        .map((v) => `<span class="bpe-token${learnedSet.has(v) ? " learned" : ""}">${escapeHtml(visibleChar(v))}</span>`)
        .join("") + (st.vocab.length > 150 ? `<span class="bpe-hint">+ ${st.vocab.length - 150} more</span>` : "");

    rulesEl.innerHTML = st.rules.length
      ? st.rules
          .map(
            (r, i) =>
              `<div class="bpe-rule">
                <div class="bpe-rule-num">${i + 1}</div>
                <div style="font-family:Consolas,monospace;font-size:0.8rem">
                  ${escapeHtml(visibleChar(r.left))} + ${escapeHtml(visibleChar(r.right))} → <strong>${escapeHtml(visibleChar(r.merged))}</strong>
                </div>
                <div style="font-family:Consolas,monospace;color:var(--muted);font-size:0.75rem">×${r.count}</div>
              </div>`
          )
          .join("")
      : '<div class="bpe-hint">No merge rules learned yet at iteration 0.</div>';
  }

  function renderInference(): void {
    if (!model) return;
    const inferText = ($("bpe-infer") as HTMLTextAreaElement)?.value ?? "";
    const inferMode = ($("bpe-inferMode") as HTMLSelectElement)?.value ?? "full";
    const resultEl = $("bpe-result");
    const traceEl = $("bpe-trace");
    const countEl = $("bpe-count");
    if (!resultEl || !traceEl || !countEl) return;

    let seqs: string[];
    try {
      seqs = parseCorpus(inferText, getOptions(), true);
    } catch (e: unknown) {
      resultEl.innerHTML = `<div class="bpe-hint">${escapeHtml((e as Error).message)}</div>`;
      return;
    }

    const rules = inferMode === "current" ? model.states[step].rules : model.rules;
    const learned = new Set(rules.map((r) => r.merged));

    const results = seqs.map((source) => ({
      source,
      ...tokenizeWithRules(Array.from(source), rules),
    }));

    const totalTokens = results.reduce((acc, r) => acc + r.tokens.length, 0);
    countEl.textContent = `${totalTokens} token${totalTokens === 1 ? "" : "s"}`;

    if (!results.length) {
      resultEl.innerHTML = '<div class="bpe-hint">Type text to tokenize.</div>';
      traceEl.innerHTML = "";
      return;
    }

    resultEl.innerHTML = results
      .map(
        (r, i) =>
          `<div class="bpe-seq">
            <div class="bpe-idx">${i + 1}</div>
            <div class="bpe-tokens">${renderTokenSpans(r.tokens, learned)}</div>
          </div>`
      )
      .join("");

    if (!rules.length) {
      traceEl.innerHTML = '<div class="bpe-hint">No merge rules are being applied.</div>';
      return;
    }

    const first = results[0];
    traceEl.innerHTML =
      `<div class="bpe-hint" style="margin-bottom:6px">Trace for first sequence: <code style="font-family:Consolas,monospace">${escapeHtml(first.source)}</code></div>` +
      rules
        .map(
          (r, i) =>
            `<div class="bpe-trace-item">
              <div class="bpe-meta">rule ${i + 1}: ${escapeHtml(visibleChar(r.left))} + ${escapeHtml(visibleChar(r.right))} → ${escapeHtml(visibleChar(r.merged))}</div>
              <div style="font-family:Consolas,monospace;font-size:0.8rem">${first.trace[i + 1].map((x) => escapeHtml(visibleChar(x))).join(" · ")}</div>
            </div>`
        )
        .join("");
  }

  function renderDots(st: BpeIterationState): void {
    const activeStep = !st.candidate ? 4 : step === 0 ? 2 : 3;
    for (let i = 1; i <= 4; i++) {
      const dot = $(`bpe-d${i}`);
      if (!dot) continue;
      if (i === activeStep) {
        dot.setAttribute("aria-current", "step");
      } else {
        dot.removeAttribute("aria-current");
      }
    }
  }

  function render(): void {
    if (!model) return;
    step = Math.max(0, Math.min(step, model.states.length - 1));
    const st = model.states[step];
    const learnedSet = new Set(st.rules.map((r) => r.merged));

    const statusEl = $("bpe-status");
    if (statusEl) {
      statusEl.textContent = `${model.rules.length} merges learned`;
      statusEl.dataset.state = "ready";
    }

    const setContent = (id: string, val: string | number) => {
      const el = $(id);
      if (el) el.textContent = String(val);
    };

    setContent("bpe-sSeq", model.source.length.toLocaleString());
    setContent("bpe-sChar", model.charCount.toLocaleString());
    setContent("bpe-sInit", model.initialVocab.length);
    setContent("bpe-sFinal", model.states.at(-1)?.vocab.length ?? 0);
    setContent("bpe-sMerges", model.rules.length);
    setContent("bpe-sStop", model.stopReason);

    setContent("bpe-iter", `iteration ${step} / ${model.rules.length}`);
    const captionEl = $("bpe-caption");
    if (captionEl) {
      captionEl.textContent =
        step === 0
          ? "Initial character-level representation."
          : `Corpus after ${step} merge${step === 1 ? "" : "s"}.`;
    }

    const shownSeqs = st.seqs.slice(0, 10);
    const corpusStateEl = $("bpe-corpusState");
    if (corpusStateEl) {
      corpusStateEl.innerHTML = shownSeqs
        .map(
          (t, i) =>
            `<div class="bpe-seq">
              <div class="bpe-idx">${i + 1}</div>
              <div class="bpe-tokens">${renderTokenSpans(t, learnedSet)}</div>
            </div>`
        )
        .join("");
    }

    const hiddenEl = $("bpe-hidden");
    if (hiddenEl) {
      if (st.seqs.length > 10) {
        hiddenEl.style.display = "block";
        hiddenEl.textContent = `Showing 10 of ${st.seqs.length.toLocaleString()} training sequences.`;
      } else {
        hiddenEl.style.display = "none";
      }
    }

    renderPairs(st);
    renderDecision(st);
    renderVocab(st);
    renderInference();
    renderDots(st);

    const prevBtn = $("bpe-prev") as HTMLButtonElement | null;
    const nextBtn = $("bpe-next") as HTMLButtonElement | null;
    const finalBtn = $("bpe-final") as HTMLButtonElement | null;
    const restartBtn = $("bpe-restart") as HTMLButtonElement | null;
    const playBtn = $("bpe-play") as HTMLButtonElement | null;
    const slider = $("bpe-slider") as HTMLInputElement | null;
    const stepLabel = $("bpe-step-label");

    if (prevBtn) prevBtn.disabled = step === 0;
    if (nextBtn) nextBtn.disabled = step >= model.states.length - 1;
    if (finalBtn) finalBtn.disabled = step >= model.states.length - 1;
    if (restartBtn) restartBtn.disabled = step === 0;
    if (playBtn) playBtn.disabled = model.states.length <= 1;

    if (slider) {
      slider.max = String(model.states.length - 1);
      slider.value = String(step);
    }
    if (stepLabel) {
      stepLabel.textContent = `Step ${step} / ${model.states.length - 1}`;
    }

    if (step === model.states.length - 1) {
      stopPlay();
    }
  }

  function bindInspectorEvents(): void {
    $("bpe-restart")?.addEventListener("click", () => {
      stopPlay();
      if (model) {
        step = 0;
        render();
      }
    });

    $("bpe-prev")?.addEventListener("click", () => {
      stopPlay();
      if (model && step > 0) {
        step--;
        render();
      }
    });

    $("bpe-next")?.addEventListener("click", () => {
      stopPlay();
      if (model && step < model.states.length - 1) {
        step++;
        render();
      }
    });

    $("bpe-final")?.addEventListener("click", () => {
      stopPlay();
      if (model) {
        step = model.states.length - 1;
        render();
      }
    });

    $("bpe-play")?.addEventListener("click", () => {
      if (playing) {
        stopPlay();
      } else {
        if (model && step >= model.states.length - 1) {
          step = 0;
          render();
        }
        startPlay();
      }
    });

    $("bpe-slider")?.addEventListener("input", (e: Event) => {
      stopPlay();
      if (model) {
        const val = parseInt((e.target as HTMLInputElement).value, 10);
        step = Math.max(0, Math.min(val, model.states.length - 1));
        render();
      }
    });

    $("bpe-infer")?.addEventListener("input", (e: Event) => {
      currentInferText = (e.target as HTMLTextAreaElement).value;
      renderInference();
    });
    $("bpe-inferMode")?.addEventListener("change", renderInference);

    // Stages navigation buttons (1: Characters, 2: Count pairs, 3: Merge winner, 4: Tokenizer)
    $("bpe-d1")?.addEventListener("click", () => {
      stopPlay();
      if (model) {
        step = 0;
        render();
      }
    });

    $("bpe-d2")?.addEventListener("click", () => {
      stopPlay();
      if (model) {
        step = 0;
        render();
      }
    });

    $("bpe-d3")?.addEventListener("click", () => {
      stopPlay();
      if (model) {
        // Jump to first merge step if available
        step = Math.min(1, model.states.length - 1);
        render();
      }
    });

    $("bpe-d4")?.addEventListener("click", () => {
      stopPlay();
      if (model) {
        step = model.states.length - 1;
        render();
      }
    });
  }

  function handleTrain(): void {
    stopPlay();
    const corpusText = corpusInput?.value ?? "";
    const options = getOptions();

    try {
      model = trainBpe(corpusText, options);
      step = 0;

      // Update output summary panel
      if (outputEl) {
        const finalVocabSize = model.states.at(-1)?.vocab.length ?? 0;
        const topLearned = model.rules.slice(0, 8).map((r) => r.merged);
        outputEl.innerHTML = `
          <div class="dl-output-block">
            <span class="dl-prediction-title">Trained Vocabulary</span>
            <div class="dl-prediction-score" style="font-size: 1.35rem; font-family: Consolas, monospace; margin: 4px 0 8px;">
              ${finalVocabSize} subwords <span style="font-size: 0.82rem; color: var(--muted); font-weight: normal;">(${model.rules.length} merges)</span>
            </div>
            <p style="font-size: 0.8rem; color: var(--muted); margin: 0 0 6px;">
              Initial alphabet: <strong>${model.initialVocab.length}</strong> chars &nbsp;·&nbsp; Stop reason: <em>${escapeHtml(model.stopReason)}</em>
            </p>
            ${
              topLearned.length > 0
                ? `<div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:8px;">
                    ${topLearned.map((tok) => `<span class="bpe-token learned" style="min-height:24px; min-width:24px; font-size:0.75rem; padding:2px 6px;">${escapeHtml(visibleChar(tok))}</span>`).join("")}
                  </div>`
                : ""
            }
          </div>
        `;
      }

      if (statusEl) {
        statusEl.textContent = `${model.rules.length} merges learned`;
        statusEl.dataset.state = "ready";
      }
      if (statusMsgEl) {
        statusMsgEl.textContent = `Trained ${model.rules.length} merge rules across ${model.source.length} sequences. Showing step 0 below.`;
        statusMsgEl.classList.remove("dl-error");
      }
      if (summaryMetaEl) {
        summaryMetaEl.textContent = `${model.charCount} chars · ${model.source.length} seqs`;
      }

      // Mount the execution inspector template
      if (resultContainer) {
        resultContainer.innerHTML = renderInspectorTemplate(currentInferText);
      }

      bindInspectorEvents();
      render();
    } catch (err: unknown) {
      if (statusMsgEl) {
        statusMsgEl.textContent = (err as Error).message;
        statusMsgEl.classList.add("dl-error");
      }
      if (statusEl) {
        statusEl.textContent = "Error";
        statusEl.dataset.state = "error";
      }
    }
  }

  $("bpe-train")?.addEventListener("click", handleTrain);

  $("bpe-reset-btn")?.addEventListener("click", () => {
    if (corpusInput) corpusInput.value = initialCorpus;
    currentInferText = "lowest newer";
    const splitSelect = $("bpe-split") as HTMLSelectElement;
    const customInput = $("bpe-custom") as HTMLInputElement;
    const tieSelect = $("bpe-tie") as HTMLSelectElement;
    const nfcCheck = $("bpe-nfc") as HTMLInputElement;
    const lowerCheck = $("bpe-lower") as HTMLInputElement;
    const punctCheck = $("bpe-punct") as HTMLInputElement;
    const targetInput = $("bpe-target") as HTMLInputElement;
    const maxInput = $("bpe-max") as HTMLInputElement;
    const minInput = $("bpe-min") as HTMLInputElement;
    const fnEl = $("bpe-filename");

    if (splitSelect) splitSelect.value = "whitespace";
    if (customInput) { customInput.value = "[;,]+"; customInput.disabled = true; }
    if (tieSelect) tieSelect.value = "first";
    if (nfcCheck) nfcCheck.checked = true;
    if (lowerCheck) lowerCheck.checked = false;
    if (punctCheck) punctCheck.checked = false;
    if (targetInput) targetInput.value = "35";
    if (maxInput) maxInput.value = "25";
    if (minInput) minInput.value = "2";
    if (fnEl) fnEl.textContent = "";

    invalidate("Reset to initial example corpus. Click 'Train tokenizer' to run.");
  });

  // Invalidate when corpus or options are modified
  corpusInput?.addEventListener("input", () => {
    invalidate();
  });

  ["bpe-split", "bpe-custom", "bpe-tie", "bpe-nfc", "bpe-lower", "bpe-punct", "bpe-target", "bpe-max", "bpe-min"].forEach((id) => {
    $(id)?.addEventListener("change", () => {
      invalidate();
    });
  });

  $("bpe-split")?.addEventListener("change", () => {
    const splitSelect = $("bpe-split") as HTMLSelectElement;
    const customInput = $("bpe-custom") as HTMLInputElement;
    if (customInput && splitSelect) {
      customInput.disabled = splitSelect.value !== "custom";
    }
  });

  $("bpe-example")?.addEventListener("click", () => {
    if (corpusInput) corpusInput.value = "low lower lowest new newer newest wide wider widest low low lower newer wide";
    currentInferText = "lowest newer";
    const inferEl = $("bpe-infer") as HTMLTextAreaElement;
    const splitEl = $("bpe-split") as HTMLSelectElement;
    const customEl = $("bpe-custom") as HTMLInputElement;
    const fnEl = $("bpe-filename");
    if (inferEl) inferEl.value = currentInferText;
    if (splitEl) splitEl.value = "whitespace";
    if (customEl) customEl.disabled = true;
    if (fnEl) fnEl.textContent = "";
    invalidate();
  });

  $("bpe-morphology")?.addEventListener("click", () => {
    if (corpusInput) corpusInput.value = "play playing player plays replay walk walking walker walks rework teach teacher teaches reteach";
    currentInferText = "replaying walker reteaches";
    const inferEl = $("bpe-infer") as HTMLTextAreaElement;
    const splitEl = $("bpe-split") as HTMLSelectElement;
    const targetEl = $("bpe-target") as HTMLInputElement;
    const maxEl = $("bpe-max") as HTMLInputElement;
    const minEl = $("bpe-min") as HTMLInputElement;
    const customEl = $("bpe-custom") as HTMLInputElement;
    const fnEl = $("bpe-filename");

    if (inferEl) inferEl.value = currentInferText;
    if (splitEl) splitEl.value = "whitespace";
    if (customEl) customEl.disabled = true;
    if (targetEl) targetEl.value = "35";
    if (maxEl) maxEl.value = "25";
    if (minEl) minEl.value = "2";
    if (fnEl) fnEl.textContent = "";
    invalidate();
  });

  $("bpe-clear")?.addEventListener("click", () => {
    if (corpusInput) corpusInput.value = "";
    const fnEl = $("bpe-filename");
    if (fnEl) fnEl.textContent = "";
    invalidate("Corpus cleared.");
  });

  $("bpe-file")?.addEventListener("change", (e: Event) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;

    if (file.size > 1e6) {
      if (statusMsgEl) {
        statusMsgEl.textContent = "Choose a file under 1 MB for this demo.";
        statusMsgEl.classList.add("dl-error");
      }
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      if (corpusInput) corpusInput.value = String(reader.result || "");
      const fnEl = $("bpe-filename");
      if (fnEl) fnEl.textContent = file.name;
      invalidate(`File "${file.name}" loaded. Click "Train tokenizer" to train.`);
    };
    reader.onerror = () => {
      if (statusMsgEl) {
        statusMsgEl.textContent = "Could not read file.";
        statusMsgEl.classList.add("dl-error");
      }
    };
    reader.readAsText(file);
  });
}

