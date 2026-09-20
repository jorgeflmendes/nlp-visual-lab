import type { Topic } from "./data/topics.ts";
import { getTopicIcon } from "./data/icons.ts";
import { typesetMath } from "./main.ts";
import {
  type DocumentInput,
  type TfidfOptions,
  type TfidfState,
  type TfMode,
  type IdfMode,
  type VocabOrder,
  computeTfidfState,
  computeCosine,
  performRetrieval,
} from "./tfidfAlgorithm.ts";
import "./tfidfLab.css";

function escapeHtml(s: string): string {
  return String(s).replace(/[&<>'"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[c] ?? c
  );
}

const defaultDocs: DocumentInput[] = [
  { name: "D1", text: "deep learning architectures train neural language models with attention mechanisms" },
  { name: "D2", text: "information retrieval searches relevant documents using vector space similarity and ranking" },
  { name: "D3", text: "neural search combines language models with vector retrieval for document ranking" },
];

const foundationsRoutes: [string, string][] = [
  ["bpe", "BPE"],
  ["edit-distance", "Edit distance"],
  ["tfidf", "TF-IDF"],
];

export function tfidfPage(topic: Topic): string {
  return `<article class="tfidf-workspace dl-workspace" data-kind="tfidf">
    <nav class="dl-model-nav" aria-label="Foundations laboratories">
      <a class="dl-back" href="#/foundations" aria-label="All foundations laboratories">← <span>Foundations</span></a>
      <div>${foundationsRoutes.map(([slug, label]) => `<a href="#/${slug}" ${topic.slug === slug ? 'aria-current="page"' : ""}>${label}</a>`).join("")}</div>
    </nav>
    <header class="dl-page-heading">
      <div>
        <p class="dl-eyebrow">Foundations / Vector space laboratory</p>
        <div class="dl-heading-title-row">
          ${getTopicIcon(topic.slug, "dl-heading-icon")}
          <h1>Build TF-IDF from first principles.</h1>
        </div>
        <p>Inspect counts, TF, DF, IDF and TF-IDF values, then use the resulting vectors for cosine similarity and query retrieval.</p>
      </div>
      <span id="tfidf-status" class="dl-state">Not computed</span>
    </header>

    <section class="dl-experiment" aria-label="Document collection and configuration">
      <div class="dl-input-panel">
        <div class="dl-section-label">
          <label>Document collection</label>
          <span>01 / Documents &amp; Configuration</span>
        </div>

        <div class="dl-presets" aria-label="Collection presets and actions">
          <span>Actions</span>
          <button id="tfidf-add-doc" type="button">+ Add document</button>
          <button id="tfidf-example" type="button">Reset example</button>
          <label style="cursor:pointer; color:var(--green); font-size:.74rem; text-decoration:underline; text-decoration-color:#a9c4b9; text-underline-offset:4px; padding-inline:8px;">
            Load text files
            <input id="tfidf-file" class="tfidf-file-input" type="file" multiple accept=".txt,.md,text/plain,text/markdown" />
          </label>
        </div>

        <div id="tfidf-editors" class="tfidf-doc-grid"></div>

        <details class="tfidf-config-details">
          <summary>Representation &amp; Preprocessing</summary>
          <div class="tfidf-config-grid">
            <label>
              <span class="tfidf-label">Term frequency</span>
              <select id="tfidf-tf-mode" style="font-size:.75rem; min-height:36px; padding:6px 8px;">
                <option value="relative">Relative: count / length</option>
                <option value="raw">Raw count</option>
                <option value="binary">Binary (0 or 1)</option>
                <option value="log">Sublinear: 1 + ln(count)</option>
              </select>
            </label>
            <label>
              <span class="tfidf-label">Inverse doc frequency</span>
              <select id="tfidf-idf-mode" style="font-size:.75rem; min-height:36px; padding:6px 8px;">
                <option value="classic">Classic: ln(N / df)</option>
                <option value="smooth">Smoothed: ln((N+1)/(df+1)) + 1</option>
                <option value="log10">Base 10: log10(N / df)</option>
                <option value="log2">Base 2: log2(N / df)</option>
              </select>
            </label>
            <label>
              <span class="tfidf-label">Vocabulary order</span>
              <select id="tfidf-order" style="font-size:.75rem; min-height:36px; padding:6px 8px;">
                <option value="alpha">Alphabetical</option>
                <option value="freq">Corpus frequency</option>
                <option value="idf">Highest IDF first</option>
              </select>
            </label>
            <label>
              <span class="tfidf-label">Precision</span>
              <select id="tfidf-precision" style="font-size:.75rem; min-height:36px; padding:6px 8px;">
                <option value="2">2 decimals</option>
                <option value="3" selected>3 decimals</option>
                <option value="4">4 decimals</option>
              </select>
            </label>
          </div>
          <div class="tfidf-checks" style="margin-top:10px;">
            <label><input id="tfidf-lower" type="checkbox" checked /> Lowercase</label>
            <label><input id="tfidf-punct" type="checkbox" checked /> Ignore punctuation</label>
            <label><input id="tfidf-stop" type="checkbox" /> Remove English stopwords</label>
          </div>
        </details>

        <div class="dl-run-controls">
          <button id="tfidf-compute" class="dl-primary" type="button">Compute vectors <span aria-hidden="true">→</span></button>
          <button id="tfidf-reset-btn" type="button">Reset</button>
        </div>
      </div>

      <div class="dl-output-panel">
        <div class="dl-section-label">
          <span>Learned vector representation</span>
          <span>02 / Status &amp; Summary</span>
        </div>
        <div id="tfidf-summary-output" aria-live="polite">
          <p class="dl-output-empty">Click "Compute vectors" to evaluate the term-document matrix.</p>
          <p class="dl-note">Configure the documents and parameters, then compute to inspect the vector space representations.</p>
        </div>
        <div class="dl-model-meta">
          <span>Vector Space Model</span>
          <span>Exact local arithmetic</span>
          <span id="tfidf-runtime-meta">Client-side execution</span>
        </div>
      </div>

      <div id="tfidf-status-msg" role="status" aria-live="polite">Ready. Documents and settings are editable.</div>
    </section>

    <!-- Execution Container: Empty state before calculation, Interactive tools after calculation -->
    <div id="tfidf-result-container" style="margin-top: 24px;">
      <section class="dl-empty" aria-label="Execution inspector">
        <div>
          <p class="dl-eyebrow">03 / Execution inspector</p>
          <h2>From text corpus to vector space.</h2>
          <p>Inspect term counts, document frequencies, TF-IDF weights, cosine similarities, and query retrieval.</p>
        </div>
        <ol class="dl-empty-flow">
          <li>Corpus</li>
          <li>Vocabulary</li>
          <li>Weights (TF-IDF)</li>
          <li>Vector Space</li>
        </ol>
        <p class="dl-note">Click "Compute vectors" above to generate the term-document matrix and calculation tools.</p>
      </section>
    </div>

    <details class="dl-model-notes">
      <summary>About the Vector Space Model and TF-IDF</summary>
      <p>Term Frequency–Inverse Document Frequency (TF-IDF) converts textual documents into numerical vectors in a shared vocabulary space. The term frequency (TF) scores how salient a word is within an individual document, while the inverse document frequency (IDF) attenuates ubiquitous words that provide little discriminative power across the entire collection. Cosine similarity measures the normalized dot product between two vector directions, making it robust against document length differences.</p>
    </details>
  </article>`;
}

function renderExecutionTemplate(): string {
  return `
    <section class="dl-execution" aria-label="TF-IDF computation and vector inspection">
      <!-- Stats bar -->
      <div class="tfidf-stats-container">
        <div class="dl-statistics">
          <div><dt>DOCUMENTS</dt><dd id="tfidf-sDocs">—</dd></div>
          <div><dt>VOCABULARY</dt><dd id="tfidf-sVocab">—</dd></div>
          <div><dt>TOTAL TOKENS</dt><dd id="tfidf-sTokens">—</dd></div>
          <div><dt>NON-ZERO CELLS</dt><dd id="tfidf-sNnz">—</dd></div>
          <div><dt>SPARSITY</dt><dd id="tfidf-sSparsity">—</dd></div>
          <div><dt>SELECTED TERM</dt><dd id="tfidf-sTerm" style="font-family: Consolas, monospace;">—</dd></div>
        </div>
      </div>

      <!-- Term-document matrix section -->
      <div class="tfidf-matrix-card">
        <div class="tfidf-card-header">
          <div>
            <p class="dl-eyebrow" style="margin: 0 0 3px;">Term-document matrix</p>
            <h3>Weights &amp; Matrix representations</h3>
          </div>
          <div id="tfidf-matrix-tabs" class="dl-presets" style="margin:0;">
            <button type="button" data-mode="count">Counts</button>
            <button type="button" data-mode="tf">TF</button>
            <button type="button" data-mode="idf">DF / IDF</button>
            <button type="button" data-mode="tfidf" class="active">TF-IDF</button>
          </div>
        </div>

        <div id="tfidf-formula-desc" class="dl-input-hint" style="font-family: Consolas, monospace; margin-bottom: 8px;"></div>
        <div id="tfidf-matrix-box" class="tfidf-matrix-wrap"></div>
      </div>

      <!-- Split 1: Cell inspector + Term importance -->
      <div class="tfidf-inspector-grid">
        <div class="tfidf-card-clean">
          <div class="tfidf-card-header">
            <div>
              <p class="dl-eyebrow" style="margin: 0 0 3px;">Step-by-step arithmetic</p>
              <h3>Cell calculation inspector</h3>
            </div>
            <span id="tfidf-cell-pill" class="dl-state">Select a cell</span>
          </div>
          <div id="tfidf-cell-inspector">
            <p class="dl-note">Select any numeric cell in the matrix above to trace its exact count, document frequency, TF, IDF, and resulting weight.</p>
          </div>
        </div>

        <div class="tfidf-card-clean tfidf-border-left">
          <div class="tfidf-card-header">
            <div>
              <p class="dl-eyebrow" style="margin: 0 0 3px;">Corpus distribution</p>
              <h3>Term statistics</h3>
            </div>
          </div>
          <div id="tfidf-term-info">
            <p class="dl-note">Select a term in the matrix to view its document frequency across the collection.</p>
          </div>
        </div>
      </div>

      <!-- Split 2: Document cosine similarity + Query retrieval -->
      <div class="tfidf-tools-grid">
        <div class="tfidf-card-clean">
          <div class="tfidf-card-header">
            <div>
              <p class="dl-eyebrow" style="margin: 0 0 3px;">Vector angle</p>
              <h3>Document cosine similarity</h3>
            </div>
            <span id="tfidf-cos-pill" class="dl-state">cos = —</span>
          </div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">
            <label>
              <span class="tfidf-label">Document A</span>
              <select id="tfidf-doc-a"></select>
            </label>
            <label>
              <span class="tfidf-label">Document B</span>
              <select id="tfidf-doc-b"></select>
            </label>
          </div>
          <div id="tfidf-cos-details"></div>
        </div>

        <div class="tfidf-card-clean tfidf-border-left">
          <div class="tfidf-card-header">
            <div>
              <p class="dl-eyebrow" style="margin: 0 0 3px;">Information retrieval</p>
              <h3>Query ranking playground</h3>
            </div>
          </div>
          <div>
            <label for="tfidf-query" class="tfidf-label">Search query</label>
            <input id="tfidf-query" type="text" value="neural search" style="width:100%;" />
          </div>
          <div id="tfidf-query-info" class="dl-note" style="margin: 8px 0;"></div>
          <div id="tfidf-ranking" style="max-height: 220px; overflow-y: auto;"></div>
        </div>
      </div>
    </section>
  `;
}

export function bindTfidf(): void {
  const $ = (id: string) => document.getElementById(id);
  let docs: DocumentInput[] = defaultDocs.map((d) => ({ ...d }));
  let state: TfidfState | null = null;
  let matrixMode: "count" | "tf" | "idf" | "tfidf" = "tfidf";
  let selected = { term: null as string | null, doc: 0 };
  let isComputed = false;

  const getOptions = (): TfidfOptions => ({
    tfMode: (($("tfidf-tf-mode") as HTMLSelectElement)?.value ?? "relative") as TfMode,
    idfMode: (($("tfidf-idf-mode") as HTMLSelectElement)?.value ?? "classic") as IdfMode,
    order: (($("tfidf-order") as HTMLSelectElement)?.value ?? "alpha") as VocabOrder,
    precision: parseInt(($("tfidf-precision") as HTMLSelectElement)?.value ?? "3", 10),
    lowercase: ($("tfidf-lower") as HTMLInputElement)?.checked ?? true,
    removePunctuation: ($("tfidf-punct") as HTMLInputElement)?.checked ?? true,
    removeStopwords: ($("tfidf-stop") as HTMLInputElement)?.checked ?? false,
  });

  const fmt = (num: number, p?: number) => {
    const prec = p !== undefined ? p : getOptions().precision;
    return Number(num).toFixed(prec);
  };

  const invalidate = () => {
    if (!isComputed) return;
    const statusPill = $("tfidf-status");
    if (statusPill) {
      statusPill.textContent = "Input changed";
      statusPill.dataset.state = "dirty";
    }
    const statusMsg = $("tfidf-status-msg");
    if (statusMsg) {
      statusMsg.textContent = "Input changed. Click 'Compute vectors' to update results.";
    }
  };

  const renderEditors = () => {
    const box = $("tfidf-editors");
    if (!box) return;
    box.innerHTML = docs
      .map(
        (d, i) => `
        <div class="tfidf-doc-card">
          <div class="tfidf-doc-card-head">
            <input data-name="${i}" value="${escapeHtml(d.name)}" aria-label="Document name" />
            <button type="button" data-remove="${i}" ${docs.length <= 1 ? "disabled" : ""}>Remove</button>
          </div>
          <textarea data-text="${i}" rows="3" spellcheck="false" aria-label="Document text">${escapeHtml(d.text)}</textarea>
        </div>`
      )
      .join("");

    box.querySelectorAll<HTMLInputElement>("[data-name]").forEach((input) => {
      input.addEventListener("input", (e) => {
        const i = Number((e.target as HTMLElement).dataset.name);
        docs[i].name = (e.target as HTMLInputElement).value || `D${i + 1}`;
        invalidate();
      });
    });

    box.querySelectorAll<HTMLTextAreaElement>("[data-text]").forEach((textarea) => {
      textarea.addEventListener("input", (e) => {
        const i = Number((e.target as HTMLElement).dataset.text);
        docs[i].text = (e.target as HTMLTextAreaElement).value;
        invalidate();
      });
    });

    box.querySelectorAll<HTMLButtonElement>("[data-remove]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        if (docs.length <= 1) return;
        const i = Number((e.currentTarget as HTMLElement).dataset.remove);
        docs.splice(i, 1);
        renderEditors();
        invalidate();
      });
    });
  };

  const syncEditors = () => {
    docs.forEach((d, i) => {
      const nameEl = document.querySelector<HTMLInputElement>(`[data-name="${i}"]`);
      const textEl = document.querySelector<HTMLTextAreaElement>(`[data-text="${i}"]`);
      if (nameEl) d.name = nameEl.value || `D${i + 1}`;
      if (textEl) d.text = textEl.value;
    });
  };

  const renderFormula = () => {
    const desc = $("tfidf-formula-desc");
    if (!desc) return;
    const tfm = getOptions().tfMode;
    const idfm = getOptions().idfMode;

    const tfLatex: Record<TfMode, string> = {
      relative: "\\mathrm{TF}(t, d) = \\frac{\\mathrm{count}(t, d)}{|d|}",
      raw: "\\mathrm{TF}(t, d) = \\mathrm{count}(t, d)",
      binary: "\\mathrm{TF}(t, d) = \\begin{cases} 1 & \\text{if } \\mathrm{count}(t, d) > 0 \\\\ 0 & \\text{otherwise} \\end{cases}",
      log: "\\mathrm{TF}(t, d) = 1 + \\ln(\\mathrm{count}(t, d))",
    };

    const idfLatex: Record<IdfMode, string> = {
      classic: "\\mathrm{IDF}(t) = \\ln\\left(\\frac{N}{\\mathrm{df}(t)}\\right)",
      smooth: "\\mathrm{IDF}(t) = \\ln\\left(\\frac{N + 1}{\\mathrm{df}(t) + 1}\\right) + 1",
      log10: "\\mathrm{IDF}(t) = \\log_{10}\\left(\\frac{N}{\\mathrm{df}(t)}\\right)",
      log2: "\\mathrm{IDF}(t) = \\log_2\\left(\\frac{N}{\\mathrm{df}(t)}\\right)",
    };

    if (matrixMode === "count") {
      desc.innerHTML = `<span data-latex="\\mathrm{count}(t, d) = \\text{absolute occurrences of term } t \\text{ in document } d" data-display="false"></span>`;
    } else if (matrixMode === "tf") {
      desc.innerHTML = `<span data-latex="${tfLatex[tfm]}" data-display="false"></span>`;
    } else if (matrixMode === "idf") {
      desc.innerHTML = `<span data-latex="${idfLatex[idfm]}" data-display="false"></span>`;
    } else {
      desc.innerHTML = `<span data-latex="${tfLatex[tfm]} \\quad\\cdot\\quad ${idfLatex[idfm]} \\quad\\cdot\\quad \\mathrm{TF\\text{-}IDF}(t, d) = \\mathrm{TF}(t, d) \\times \\mathrm{IDF}(t)" data-display="false"></span>`;
    }
    void typesetMath(desc);
  };

  const heatClass = (val: number, max: number): string => {
    if (!max || !val) return "";
    const ratio = val / max;
    return ratio > 0.75 ? "tfidf-heat4" : ratio > 0.5 ? "tfidf-heat3" : ratio > 0.25 ? "tfidf-heat2" : "tfidf-heat1";
  };

  const renderMatrix = () => {
    renderFormula();
    const box = $("tfidf-matrix-box");
    if (!box || !state) return;
    const currentState = state;

    if (!currentState.vocab.length) {
      box.innerHTML = `<div class="tfidf-empty">No terms remaining after preprocessing. Adjust lowercase, punctuation, or stopwords.</div>`;
      return;
    }

    if (matrixMode === "idf") {
      const maxIdf = Math.max(...currentState.vocab.map((t) => currentState.idfs.get(t) || 0), 0);
      box.innerHTML = `
        <table class="tfidf-matrix-table">
          <thead>
            <tr>
              <th>Term</th>
              <th>Corpus Count</th>
              <th>df</th>
              <th>N</th>
              <th>IDF</th>
            </tr>
          </thead>
          <tbody>
            ${currentState.vocab
              .map(
                (t) => `
              <tr>
                <td>${escapeHtml(t)}</td>
                <td>${currentState.freq.get(t) || 0}</td>
                <td>${currentState.dfs.get(t) || 0}</td>
                <td>${currentState.N}</td>
                <td tabindex="0" class="tfidf-cell ${heatClass(currentState.idfs.get(t) || 0, maxIdf)} ${selected.term === t ? "selected" : ""}" data-term="${escapeHtml(t)}" data-doc="0">
                  ${fmt(currentState.idfs.get(t) || 0)}
                </td>
              </tr>`
              )
              .join("")}
          </tbody>
        </table>
      `;
    } else {
      const getter =
        matrixMode === "count"
          ? (t: string, j: number) => currentState.counts[j].get(t) || 0
          : matrixMode === "tf"
          ? (t: string, j: number) => currentState.tfs[j].get(t) || 0
          : (t: string, j: number) => currentState.weights[j].get(t) || 0;

      let max = 0;
      currentState.vocab.forEach((t) => {
        docs.forEach((_, j) => {
          max = Math.max(max, getter(t, j));
        });
      });

      box.innerHTML = `
        <table class="tfidf-matrix-table">
          <thead>
            <tr>
              <th>Term</th>
              ${docs.map((d) => `<th>${escapeHtml(d.name)}</th>`).join("")}
            </tr>
          </thead>
          <tbody>
            ${currentState.vocab
              .map(
                (t) => `
              <tr>
                <td>${escapeHtml(t)}</td>
                ${docs
                  .map((_, j) => {
                    const v = getter(t, j);
                    const isSel = selected.term === t && selected.doc === j;
                    return `<td tabindex="0" class="tfidf-cell ${heatClass(v, max)} ${isSel ? "selected" : ""}" data-term="${escapeHtml(t)}" data-doc="${j}">
                      ${matrixMode === "count" ? v : fmt(v)}
                    </td>`;
                  })
                  .join("")}
              </tr>`
              )
              .join("")}
          </tbody>
        </table>
      `;
    }

    box.querySelectorAll<HTMLElement>(".tfidf-cell").forEach((c) => {
      const activate = () => {
        selected.term = c.dataset.term || null;
        selected.doc = Number(c.dataset.doc || 0);
        renderMatrix();
        renderInspector();
        renderTermInfo();
      };
      c.addEventListener("click", activate);
      c.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          activate();
        }
      });
    });
  };

  const renderStats = () => {
    if (!state) return;
    const currentState = state;
    const totalTokens = currentState.tokenDocs.reduce((acc, toks) => acc + toks.length, 0);
    const totalCells = currentState.N * currentState.vocab.length;
    const nnz = currentState.vocab.reduce(
      (sum, t) => sum + docs.reduce((s, _, j) => s + (currentState.weights[j].get(t) !== 0 ? 1 : 0), 0),
      0
    );
    const sparsity = totalCells ? ((1 - nnz / totalCells) * 100).toFixed(1) + "%" : "—";

    const setContent = (id: string, text: string | number) => {
      const el = $(id);
      if (el) el.textContent = String(text);
    };

    setContent("tfidf-sDocs", currentState.N);
    setContent("tfidf-sVocab", currentState.vocab.length);
    setContent("tfidf-sTokens", totalTokens);
    setContent("tfidf-sNnz", nnz);
    setContent("tfidf-sSparsity", sparsity);
    setContent("tfidf-sTerm", selected.term || "—");

    const statusEl = $("tfidf-status");
    if (statusEl) {
      statusEl.textContent = `${currentState.N} docs · ${currentState.vocab.length} terms`;
      statusEl.dataset.state = "ready";
    }

    const summaryOutput = $("tfidf-summary-output");
    if (summaryOutput) {
      summaryOutput.innerHTML = `
        <div class="dl-output-block">
          <span class="dl-prediction-title">Vocabulary &amp; Sparsity</span>
          <div class="dl-prediction-score" style="font-size: 1.35rem; font-family: Consolas, monospace; margin: 4px 0 8px;">
            ${currentState.vocab.length} unique terms <span style="font-size: 0.82rem; color: var(--muted); font-weight: normal;">(${sparsity} sparsity)</span>
          </div>
          <p style="font-size: 0.8rem; color: var(--muted); margin: 0;">
            ${totalTokens} total tokens processed across ${currentState.N} documents.
          </p>
        </div>
      `;
    }
  };

  const renderInspector = () => {
    const el = $("tfidf-cell-inspector");
    const pill = $("tfidf-cell-pill");
    if (!el || !state || !selected.term) return;

    const t = selected.term;
    const j = selected.doc;
    const c = state.counts[j].get(t) || 0;
    const L = state.tokenDocs[j].length;
    const df = state.dfs.get(t) || 0;
    const T = state.tfs[j].get(t) || 0;
    const I = state.idfs.get(t) || 0;
    const W = state.weights[j].get(t) || 0;
    const tm = getOptions().tfMode;
    const im = getOptions().idfMode;

    const tfLatexCalc =
      tm === "relative"
        ? `\\mathrm{TF} = \\frac{${c}}{${L}} = ${fmt(T)}`
        : tm === "raw"
        ? `\\mathrm{TF} = ${c}`
        : tm === "binary"
        ? `\\mathrm{TF} = ${c ? 1 : 0}`
        : c
        ? `\\mathrm{TF} = 1 + \\ln(${c}) = ${fmt(T)}`
        : "\\mathrm{TF} = 0";

    const idfLatexCalc =
      im === "smooth"
        ? `\\mathrm{IDF} = \\ln\\left(\\frac{${state.N} + 1}{${df} + 1}\\right) + 1 = ${fmt(I)}`
        : im === "log10"
        ? `\\mathrm{IDF} = \\log_{10}\\left(\\frac{${state.N}}{${df}}\\right) = ${fmt(I)}`
        : im === "log2"
        ? `\\mathrm{IDF} = \\log_2\\left(\\frac{${state.N}}{${df}}\\right) = ${fmt(I)}`
        : `\\mathrm{IDF} = \\ln\\left(\\frac{${state.N}}{${df}}\\right) = ${fmt(I)}`;

    const tfidfLatexCalc = `\\mathrm{TF\\text{-}IDF} = ${fmt(T)} \\times ${fmt(I)} = \\mathbf{${fmt(W)}}`;

    if (pill) {
      pill.textContent = `${t} × ${docs[j].name}`;
    }

    el.innerHTML = `
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">
        <div style="background:var(--soft); border:1px solid var(--line); border-radius:5px; padding:10px 12px;">
          <div class="dl-eyebrow" style="margin-bottom:4px;">Within Document (${escapeHtml(docs[j].name)})</div>
          <div style="font-size:0.82rem;"><span data-latex="\\mathrm{count}(t, d) = ${c}" data-display="false"></span></div>
          <div style="font-size:0.82rem; margin-top:4px;"><span data-latex="|d| = ${L}" data-display="false"></span></div>
        </div>
        <div style="background:var(--soft); border:1px solid var(--line); border-radius:5px; padding:10px 12px;">
          <div class="dl-eyebrow" style="margin-bottom:4px;">Across Collection</div>
          <div style="font-size:0.82rem;"><span data-latex="\\mathrm{df}(t) = ${df}" data-display="false"></span></div>
          <div style="font-size:0.82rem; margin-top:4px;"><span data-latex="N = ${state.N}" data-display="false"></span></div>
        </div>
      </div>
      <div style="background:var(--soft); border:1px solid var(--line); border-radius:5px; padding:12px 14px; font-size:0.85rem; display:flex; flex-direction:column; gap:8px;">
        <div><span data-latex="${tfLatexCalc}" data-display="false"></span></div>
        <div><span data-latex="${idfLatexCalc}" data-display="false"></span></div>
        <div style="padding-top:8px; border-top:1px solid var(--line); color:var(--green); font-weight:600;">
          <span data-latex="${tfidfLatexCalc}" data-display="false"></span>
        </div>
      </div>
    `;
    void typesetMath(el);
  };

  const renderTermInfo = () => {
    const el = $("tfidf-term-info");
    if (!el || !state || !selected.term) return;
    const t = selected.term;
    const df = state.dfs.get(t) || 0;
    const idfVal = state.idfs.get(t) || 0;
    const count = state.freq.get(t) || 0;

    const note =
      df === state.N
        ? "The term occurs in all documents, carrying minimal discriminative value in this collection."
        : df === 1
        ? "The term occurs in only 1 document, making it highly specific."
        : "The IDF value increases as the term becomes rarer across documents.";

    el.innerHTML = `
      <div class="dl-statistics" style="margin:0 0 10px; border-block:none; padding:0;">
        <div><dt>CORPUS COUNT</dt><dd>${count}</dd></div>
        <div><dt>DOCUMENT FREQ</dt><dd>${df} / ${state.N}</dd></div>
        <div><dt>CALCULATED IDF</dt><dd>${fmt(idfVal)}</dd></div>
      </div>
      <p class="dl-note" style="margin-top:10px;">${note}</p>
    `;
  };

  const renderSelectors = () => {
    const docA = $("tfidf-doc-a") as HTMLSelectElement | null;
    const docB = $("tfidf-doc-b") as HTMLSelectElement | null;
    if (!docA || !docB) return;

    const html = docs.map((d, i) => `<option value="${i}">${escapeHtml(d.name)}</option>`).join("");
    const prevA = Math.min(Number(docA.value) || 0, docs.length - 1);
    const prevB = Math.min(Number.isFinite(Number(docB.value)) ? Number(docB.value) : 1, docs.length - 1);

    docA.innerHTML = html;
    docB.innerHTML = html;
    docA.value = String(prevA);
    docB.value = String(prevB);
  };

  const renderCosine = () => {
    if (!state) return;
    const docA = $("tfidf-doc-a") as HTMLSelectElement | null;
    const docB = $("tfidf-doc-b") as HTMLSelectElement | null;
    const cosPill = $("tfidf-cos-pill");
    const details = $("tfidf-cos-details");
    if (!docA || !docB || !details) return;

    const aIdx = Number(docA.value) || 0;
    const bIdx = Number(docB.value) || 0;
    const res = computeCosine(state, aIdx, bIdx);

    if (cosPill) {
      cosPill.textContent = `cos = ${fmt(res.cosine)}`;
    }

    const topShared = res.contributions.slice(0, 6);
    const maxContribution = topShared[0]?.value || 1;

    const dotLatex = `\\mathbf{d}_A \\cdot \\mathbf{d}_B = ${fmt(res.dotProduct)}`;
    const normLatex = `\\|\\mathbf{d}_A\\| = ${fmt(res.normA)} \\quad\\cdot\\quad \\|\\mathbf{d}_B\\| = ${fmt(res.normB)}`;
    const cosLatex = `\\cos(\\mathbf{d}_A, \\mathbf{d}_B) = \\frac{\\mathbf{d}_A \\cdot \\mathbf{d}_B}{\\|\\mathbf{d}_A\\| \\|\\mathbf{d}_B\\|} = \\mathbf{${fmt(res.cosine)}}`;

    details.innerHTML = `
      <div style="background:var(--soft); border:1px solid var(--line); border-radius:5px; padding:12px 14px; font-size:0.83rem; display:flex; flex-direction:column; gap:6px;">
        <div><span data-latex="${dotLatex}" data-display="false"></span></div>
        <div><span data-latex="${normLatex}" data-display="false"></span></div>
        <div style="margin-top:4px; padding-top:6px; border-top:1px solid var(--line); color:var(--green); font-weight:600;">
          <span data-latex="${cosLatex}" data-display="false"></span>
        </div>
      </div>
      <div style="margin-top:12px;">
        <span class="tfidf-label">Shared dimensions contributing to dot product</span>
        ${
          topShared.length
            ? topShared
                .map(
                  (x) => `
              <div class="tfidf-rank-item">
                <div>
                  <div style="font-family:Consolas,monospace; font-size:0.78rem;">${escapeHtml(x.term)}</div>
                  <div class="tfidf-rank-bar">
                    <div class="tfidf-rank-fill" style="width:${(x.value / maxContribution) * 100}%"></div>
                  </div>
                </div>
                <div style="text-align:right; font-family:Consolas,monospace; font-size:0.75rem;">${fmt(x.value)}</div>
              </div>`
                )
                .join("")
            : `<div class="dl-note" style="margin-top:6px;">No positively weighted dimensions are shared between these two documents.</div>`
        }
      </div>
    `;
    void typesetMath(details);
  };

  const renderRetrieval = () => {
    if (!state) return;
    const queryInput = $("tfidf-query") as HTMLInputElement | null;
    const infoEl = $("tfidf-query-info");
    const rankEl = $("tfidf-ranking");
    if (!queryInput || !infoEl || !rankEl) return;

    const queryText = queryInput.value;
    const res = performRetrieval(state, queryText, getOptions());

    infoEl.innerHTML = `
      Vocabulary matches: <span style="font-family:Consolas,monospace; font-weight:600;">${res.knownTokens.length ? res.knownTokens.map(escapeHtml).join(" · ") : "None"}</span>
      ${res.unknownTokens.length ? `<br>Out of vocabulary: <span style="font-family:Consolas,monospace; color:var(--muted);">${res.unknownTokens.map(escapeHtml).join(" · ")}</span>` : ""}
    `;

    const maxScore = Math.max(...res.ranking.map((r) => r.score), 0);

    rankEl.innerHTML = res.ranking
      .map(
        (r, i) => `
        <div class="tfidf-rank-item">
          <div>
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="tfidf-idx">#${i + 1}</span>
              <strong>${escapeHtml(r.name)}</strong>
            </div>
            <div class="tfidf-rank-bar">
              <div class="tfidf-rank-fill" style="width:${maxScore ? (r.score / maxScore) * 100 : 0}%"></div>
            </div>
          </div>
          <div style="text-align:right; font-family:Consolas,monospace; font-size:0.78rem; font-weight:650;">
            ${fmt(r.score)}
          </div>
        </div>`
      )
      .join("");
  };

  const bindExecutionTools = () => {
    $("tfidf-matrix-tabs")?.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        $("tfidf-matrix-tabs")?.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
        const target = e.currentTarget as HTMLButtonElement;
        target.classList.add("active");
        matrixMode = (target.dataset.mode as "count" | "tf" | "idf" | "tfidf") || "tfidf";
        renderMatrix();
      });
    });

    $("tfidf-doc-a")?.addEventListener("change", renderCosine);
    $("tfidf-doc-b")?.addEventListener("change", renderCosine);
    $("tfidf-query")?.addEventListener("input", renderRetrieval);
  };

  const compute = () => {
    syncEditors();
    const options = getOptions();
    state = computeTfidfState(docs, options);

    if (!selected.term || !state.vocab.includes(selected.term)) {
      selected.term = state.vocab[0] || null;
    }
    selected.doc = Math.min(selected.doc, state.N - 1);

    const resultContainer = $("tfidf-result-container");
    if (resultContainer) {
      resultContainer.innerHTML = renderExecutionTemplate();
    }

    isComputed = true;

    renderStats();
    renderMatrix();
    renderInspector();
    renderTermInfo();
    renderSelectors();
    renderCosine();
    renderRetrieval();
    bindExecutionTools();

    const statusMsg = $("tfidf-status-msg");
    if (statusMsg) {
      statusMsg.textContent = `Calculated representation across ${state.N} documents and ${state.vocab.length} vocabulary terms.`;
    }
  };

  const resetAll = () => {
    docs = defaultDocs.map((d) => ({ ...d }));
    state = null;
    isComputed = false;
    selected = { term: null, doc: 0 };
    matrixMode = "tfidf";

    const tfModeEl = $("tfidf-tf-mode") as HTMLSelectElement | null;
    const idfModeEl = $("tfidf-idf-mode") as HTMLSelectElement | null;
    const orderEl = $("tfidf-order") as HTMLSelectElement | null;
    const precEl = $("tfidf-precision") as HTMLSelectElement | null;
    const lowerEl = $("tfidf-lower") as HTMLInputElement | null;
    const punctEl = $("tfidf-punct") as HTMLInputElement | null;
    const stopEl = $("tfidf-stop") as HTMLInputElement | null;

    if (tfModeEl) tfModeEl.value = "relative";
    if (idfModeEl) idfModeEl.value = "classic";
    if (orderEl) orderEl.value = "alpha";
    if (precEl) precEl.value = "3";
    if (lowerEl) lowerEl.checked = true;
    if (punctEl) punctEl.checked = true;
    if (stopEl) stopEl.checked = false;

    renderEditors();

    const resultContainer = $("tfidf-result-container");
    if (resultContainer) {
      resultContainer.innerHTML = `
        <section class="dl-empty" aria-label="Execution inspector">
          <div>
            <p class="dl-eyebrow">03 / Execution inspector</p>
            <h2>From text corpus to vector space.</h2>
            <p>Inspect term counts, document frequencies, TF-IDF weights, cosine similarities, and query retrieval.</p>
          </div>
          <ol class="dl-empty-flow">
            <li>Corpus</li>
            <li>Vocabulary</li>
            <li>Weights (TF-IDF)</li>
            <li>Vector Space</li>
          </ol>
          <p class="dl-note">Click "Compute vectors" above to generate the term-document matrix and calculation tools.</p>
        </section>
      `;
    }

    const summaryOutput = $("tfidf-summary-output");
    if (summaryOutput) {
      summaryOutput.innerHTML = `
        <p class="dl-output-empty">Click "Compute vectors" to evaluate the term-document matrix.</p>
        <p class="dl-note">Configure the documents and parameters, then compute to inspect the vector space representations.</p>
      `;
    }

    const statusPill = $("tfidf-status");
    if (statusPill) {
      statusPill.textContent = "Not computed";
      statusPill.dataset.state = "idle";
    }

    const statusMsg = $("tfidf-status-msg");
    if (statusMsg) {
      statusMsg.textContent = "Ready. Documents and settings are editable.";
    }
  };

  // Bind main actions
  $("tfidf-compute")?.addEventListener("click", compute);
  $("tfidf-reset-btn")?.addEventListener("click", resetAll);

  $("tfidf-add-doc")?.addEventListener("click", () => {
    if (docs.length >= 12) {
      const msg = $("tfidf-status-msg");
      if (msg) msg.textContent = "The interactive matrix is limited to 12 documents for clarity.";
      return;
    }
    syncEditors();
    docs.push({ name: `D${docs.length + 1}`, text: "" });
    renderEditors();
    invalidate();
  });

  $("tfidf-example")?.addEventListener("click", () => {
    docs = defaultDocs.map((d) => ({ ...d }));
    renderEditors();
    invalidate();
  });

  $("tfidf-file")?.addEventListener("change", (e: Event) => {
    const files = Array.from((e.target as HTMLInputElement).files || []).slice(0, 12);
    if (!files.length) return;

    Promise.all(
      files.map(
        (f) =>
          new Promise<DocumentInput>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () =>
              resolve({
                name: f.name.replace(/\.[^.]+$/, ""),
                text: String(reader.result || ""),
              });
            reader.onerror = reject;
            reader.readAsText(f);
          })
      )
    )
      .then((loaded) => {
        docs = loaded;
        renderEditors();
        invalidate();
      })
      .catch(() => {
        const msg = $("tfidf-status-msg");
        if (msg) msg.textContent = "Could not read one or more files.";
      });
  });

  ["tfidf-tf-mode", "tfidf-idf-mode", "tfidf-order", "tfidf-precision", "tfidf-lower", "tfidf-punct", "tfidf-stop"].forEach(
    (id) => {
      $(id)?.addEventListener("change", () => {
        if (isComputed) {
          compute();
        } else {
          invalidate();
        }
      });
    }
  );

  // Initial render: documents editable, lower section cleanly in initial empty state
  renderEditors();
}
