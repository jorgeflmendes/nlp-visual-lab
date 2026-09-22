import type { DeepLearningTopic } from "./data/topics.ts";
import { getTopicIcon } from "./data/icons.ts";
import { bindExecutionWorkspace } from "./executionWorkspace.ts";
import {
  extractSentenceEmbedding,
  extractSentenceEmbeddingsBatch,
} from "./realModels.ts";
import {
  buildSentenceMatrix,
  rankBySimilarity,
  projectSentencePCA,
} from "./sentenceEmbeddingsAlgorithm.ts";
import "./sentenceEmbeddingsLab.css";

type TypesetMath = (root: ParentNode) => Promise<void>;

const routes: [DeepLearningTopic["kind"], string, string][] = [
  ["lstm", "lstm", "LSTM"],
  ["seq2seq", "seq2seq", "Seq2Seq"],
  ["attention", "attention", "Attention"],
  ["sentence-embeddings", "sentence-embeddings", "Sentence Embeddings"],
  ["transformer", "transformers", "Transformers"],
];

const PRESETS: Record<string, string[]> = {
  clusters: [
    "Natural language processing enables computers to understand human text.",
    "Computational linguistics studies mathematical models of syntax and semantics.",
    "Deep neural networks learn continuous vector representations from text.",
    "The solar system consists of eight planets orbiting the sun.",
    "Astronomers observe distant galaxies and cosmic phenomena using telescopes.",
  ],
  qa: [
    "What is the capital of France?",
    "Paris is the capital and largest city of France.",
    "How do transformer neural networks work?",
    "Transformers rely on self-attention mechanisms to process tokens in parallel.",
  ],
  paraphrase: [
    "The doctor examined the patient carefully.",
    "The physician thoroughly checked the sick individual.",
    "A rapid brown fox leaped over a lazy hound.",
    "The fast brown canine jumped across the sleeping dog.",
  ],
};

function escapeHtml(s: string): string {
  return String(s).replace(/[&<>'"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[c] ?? c
  );
}

export function sentenceEmbeddingsPage(topic: DeepLearningTopic): string {
  return `<article class="se-workspace dl-workspace" data-kind="sentence-embeddings">
    <nav class="dl-model-nav" aria-label="Neural architecture experiments">
      <a class="dl-back" href="#/neural" aria-label="All neural architecture experiments">← <span>Neural Architectures</span></a>
      <div>${routes.map(([kind, slug, label]) => `<a href="#/${slug}" ${kind === topic.kind ? 'aria-current="page"' : ""}>${label}</a>`).join("")}</div>
    </nav>

    <header class="dl-page-heading">
      <div>
        <p class="dl-eyebrow">Neural / Dense Semantic Representations</p>
        <div class="dl-heading-title-row">
          ${getTopicIcon(topic.slug, "dl-heading-icon")}
          <h1>all-MiniLM-L6-v2 · Sentence Embeddings</h1>
        </div>
        <p>Extract 384-dimensional dense semantic vectors using real in-browser ONNX inference. Test multi-sentence similarity matrices, semantic retrieval ranking, 2D PCA spatial projection, and trace the forward pass through tokens, encoder states, mean pooling, and L2 normalization.</p>
      </div>
      <span id="model-state" class="dl-state">Not loaded</span>
    </header>

    <!-- Top Section: Interactive Semantic Workbench -->
    <div class="se-grid-2">
      <!-- Sentence Corpus Editor -->
      <section class="se-card" aria-labelledby="se-corpus-heading">
        <div class="se-card-header">
          <h2 id="se-corpus-heading" class="se-card-title">Corpus Sentences</h2>
          <span class="se-card-badge" id="se-corpus-count">5 sentences</span>
        </div>

        <div class="dl-presets" style="margin: 0;" aria-label="Corpus presets">
          <span style="font-size: 0.75rem; color: var(--muted);">Presets:</span>
          <button type="button" class="se-preset-btn" data-corpus="clusters">NLP &amp; Astronomy (2 clusters)</button>
          <button type="button" class="se-preset-btn" data-corpus="qa">Questions &amp; Answers</button>
          <button type="button" class="se-preset-btn" data-corpus="paraphrase">Paraphrases</button>
        </div>

        <div id="se-sentence-list" class="se-sentence-list"></div>

        <div class="se-corpus-controls">
          <button id="se-add-sentence" type="button" class="se-btn">+ Add Sentence</button>
          <div class="se-btn-group">
            <button id="se-compute-btn" type="button" class="se-btn se-primary">Compute Embeddings →</button>
          </div>
        </div>
        <div id="se-compute-status" style="font-size: 0.76rem; color: var(--muted); min-height: 1.2em;"></div>
      </section>

      <!-- Semantic Analysis Tabs -->
      <section class="se-card" aria-labelledby="se-analysis-heading">
        <div class="se-card-header">
          <h2 id="se-analysis-heading" class="se-card-title">Semantic Evaluation</h2>
          <span class="se-card-badge" id="se-emb-dim">384d · L2 Unit Vectors</span>
        </div>

        <div class="se-tab-bar" role="tablist">
          <button type="button" class="se-tab-btn active" role="tab" data-tab="heatmap" aria-selected="true">Cosine Heatmap</button>
          <button type="button" class="se-tab-btn" role="tab" data-tab="retrieval" aria-selected="false">Semantic Search</button>
          <button type="button" class="se-tab-btn" role="tab" data-tab="pca" aria-selected="false">2D PCA Scatter</button>
        </div>

        <!-- Tab 1: Heatmap -->
        <div id="se-tab-heatmap" class="se-tab-pane active" role="tabpanel">
          <p style="font-size: 0.76rem; color: var(--muted); margin: 0 0 10px 0;">
            Pairwise cosine similarity matrix $S_{i,j} = e_i \\cdot e_j \\in [-1, 1]$. Cell colors indicate semantic alignment.
          </p>
          <div style="overflow-x: auto;">
            <table id="se-heatmap-table" class="se-heatmap-table">
              <tbody><tr><td style="color: var(--muted); padding: 24px;">Click "Compute Embeddings" to generate similarity matrix</td></tr></tbody>
            </table>
          </div>
        </div>

        <!-- Tab 2: Retrieval -->
        <div id="se-tab-retrieval" class="se-tab-pane" role="tabpanel">
          <p style="font-size: 0.76rem; color: var(--muted); margin: 0 0 10px 0;">
            Query the corpus in continuous vector space. Ranks sentences by descending cosine similarity with query embedding.
          </p>
          <div style="display: flex; gap: 8px; margin-bottom: 12px;">
            <input id="se-query-input" class="se-sentence-input" type="text" placeholder="Enter search query..." value="How do machines comprehend text?" autocomplete="off" />
            <button id="se-query-btn" type="button" class="se-btn se-primary">Search</button>
          </div>
          <ul id="se-retrieval-list" class="se-retrieval-list">
            <li class="se-retrieval-item" style="color: var(--muted);">Compute embeddings to enable semantic retrieval</li>
          </ul>
        </div>

        <!-- Tab 3: PCA -->
        <div id="se-tab-pca" class="se-tab-pane" role="tabpanel">
          <p style="font-size: 0.76rem; color: var(--muted); margin: 0 0 10px 0;">
            Unsupervised 2D projection via Principal Component Analysis (PC1 vs PC2) reveals semantic clustering.
          </p>
          <div class="se-pca-container">
            <svg id="se-pca-svg" class="se-pca-svg" viewBox="0 0 400 300"></svg>
          </div>
        </div>
      </section>
    </div>

    <!-- Section Divider -->
    <div class="se-section-divider">
      <span>Neural Forward Pass Inspector</span>
      <hr />
    </div>

    <!-- Bottom Section: Execution Inspector -->
    <section class="dl-experiment" aria-label="Model input and output">
      <div class="dl-input-panel">
        <div class="dl-section-label">
          <label for="neural-input">Sentence input</label>
          <span>01 / Input</span>
        </div>
        <textarea id="neural-input" rows="2" aria-describedby="neural-input-hint" spellcheck="false">Natural language processing enables computers to understand human language.</textarea>
        <div class="dl-presets" aria-label="Example inputs">
          <span>Try</span>
          <button type="button" data-preset="Natural language processing enables computers to understand human language.">NLP understanding</button>
          <button type="button" data-preset="Dense vectors capture semantic similarity between passages in geometric space.">Dense vectors</button>
          <button type="button" data-preset="Linguistic theory formalizes syntactic structures and compositional semantics.">Linguistics</button>
        </div>
        <p id="neural-input-hint" class="dl-input-hint">Enter any English sentence to extract its 384-dimensional embedding and trace each forward step.</p>
        <div class="dl-run-controls">
          <button id="run-real-model" class="dl-primary" type="button">Load &amp; run <span aria-hidden="true">→</span></button>
          <button id="load-real-model" type="button">Load only</button>
          <button id="reset-real-model" type="button">Reset</button>
        </div>
      </div>
      <div class="dl-output-panel">
        <div class="dl-section-label">
          <span>Model output</span>
          <span>02 / Embedding</span>
        </div>
        <div id="neural-output" aria-live="polite">
          <p class="dl-output-empty">Your 384-dimensional unit vector appears here.</p>
          <p class="dl-note">Run the model to compute and inspect the embedding.</p>
        </div>
        <div class="dl-model-meta">
          <a href="https://huggingface.co/Xenova/all-MiniLM-L6-v2" target="_blank" rel="noreferrer">all-MiniLM-L6-v2 model card ↗</a>
          <span>384-dimensional embeddings</span>
          <span id="model-runtime">Inference runs in your browser.</span>
        </div>
      </div>
      <div id="model-status" role="status" aria-live="polite">Ready to load. The first run downloads the quantized model weights.</div>
    </section>

    <div id="neural-result">
      <section class="dl-empty" aria-label="Execution inspector">
        <div>
          <p class="dl-eyebrow">03 / Execution inspector</p>
          <h2>From sentence to normalized vector.</h2>
          <p>Inspect subword token IDs, transformer encoder hidden states, masked mean pooling, and unit L2 Euclidean normalization.</p>
        </div>
        <ol class="dl-empty-flow">
          <li>Tokens</li>
          <li>Hidden states</li>
          <li>Mean pooling</li>
          <li>Normalized vector</li>
        </ol>
        <p class="dl-note">Run the model above to inspect this sentence’s actual activations.</p>
      </section>
    </div>

    <details class="dl-model-notes">
      <summary>About all-MiniLM-L6-v2 and sentence representations</summary>
      <p>all-MiniLM-L6-v2 maps sentences to a 384-dimensional dense vector space. It uses 6 transformer layers with 12 self-attention heads (hidden dimension 384). Unlike CLS-only pooling, mean pooling averages representations across all unmasked tokens, followed by unit $L_2$ normalization so that dot products directly measure cosine similarity.</p>
    </details>
  </article>`;
}

export function bindSentenceEmbeddings(_topic: DeepLearningTopic, typesetMath: TypesetMath): void {
  // 1. Bind the standard neural execution workspace for the bottom trace
  bindExecutionWorkspace("sentence-embeddings", typesetMath);

  // 2. Interactive Semantic Workbench State
  let sentences: string[] = [...PRESETS.clusters];
  let cachedEmbeddings: Float32Array[] | null = null;

  const sentenceListEl = document.getElementById("se-sentence-list");
  const countEl = document.getElementById("se-corpus-count");
  const addBtn = document.getElementById("se-add-sentence");
  const computeBtn = document.getElementById("se-compute-btn") as HTMLButtonElement | null;
  const computeStatusEl = document.getElementById("se-compute-status");
  const heatmapTable = document.getElementById("se-heatmap-table");
  const queryInput = document.getElementById("se-query-input") as HTMLInputElement | null;
  const queryBtn = document.getElementById("se-query-btn") as HTMLButtonElement | null;
  const retrievalList = document.getElementById("se-retrieval-list");
  const pcaSvg = document.getElementById("se-pca-svg") as unknown as SVGElement | null;
  const neuralInput = document.getElementById("neural-input") as HTMLTextAreaElement | null;
  const runModelBtn = document.getElementById("run-real-model") as HTMLButtonElement | null;

  function renderSentenceList(): void {
    if (!sentenceListEl) return;
    if (countEl) countEl.textContent = `${sentences.length} sentences`;

    sentenceListEl.innerHTML = sentences
      .map(
        (sentence, idx) => `
        <div class="se-sentence-row" data-index="${idx}">
          <span class="se-sentence-tag">S${idx + 1}</span>
          <input class="se-sentence-input" type="text" value="${escapeHtml(sentence)}" data-input-idx="${idx}" />
          <button type="button" class="se-inspect-btn" data-inspect-idx="${idx}" title="Trace this sentence in neural inspector">Inspect →</button>
          ${
            sentences.length > 2
              ? `<button type="button" class="se-remove-btn" data-remove-idx="${idx}" title="Remove sentence">✕</button>`
              : ""
          }
        </div>
      `
      )
      .join("");

    sentenceListEl.querySelectorAll<HTMLInputElement>(".se-sentence-input").forEach((inp) => {
      inp.addEventListener("input", (e) => {
        const idx = Number((e.target as HTMLInputElement).dataset.inputIdx);
        sentences[idx] = (e.target as HTMLInputElement).value;
        cachedEmbeddings = null; // Invalidate cached embeddings
        if (computeStatusEl) computeStatusEl.textContent = "Corpus modified. Recompute embeddings to update analysis.";
      });
    });

    sentenceListEl.querySelectorAll<HTMLButtonElement>(".se-inspect-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.dataset.inspectIdx);
        const text = sentences[idx];
        if (neuralInput) {
          neuralInput.value = text;
          neuralInput.focus();
          neuralInput.scrollIntoView({ behavior: "smooth", block: "center" });
        }
        if (runModelBtn) {
          runModelBtn.click();
        }
      });
    });

    sentenceListEl.querySelectorAll<HTMLButtonElement>(".se-remove-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.dataset.removeIdx);
        sentences.splice(idx, 1);
        cachedEmbeddings = null;
        renderSentenceList();
      });
    });
  }

  function renderHeatmap(matrix: number[][]): void {
    if (!heatmapTable) return;

    let html = `<thead><tr><th></th>`;
    for (let j = 0; j < sentences.length; j++) {
      html += `<th title="${escapeHtml(sentences[j])}">S${j + 1}</th>`;
    }
    html += `</tr></thead><tbody>`;

    for (let i = 0; i < sentences.length; i++) {
      html += `<tr><th title="${escapeHtml(sentences[i])}">S${i + 1}</th>`;
      for (let j = 0; j < sentences.length; j++) {
        const score = matrix[i][j];
        const intensity = Math.max(0, Math.min(1, score));
        const bg = `color-mix(in srgb, var(--accent) ${Math.round(intensity * 60)}%, var(--paper))`;
        html += `<td style="background: ${bg}; font-weight: ${i === j ? "700" : "500"}; cursor: pointer;" title="S${i + 1} vs S${j + 1}: ${score.toFixed(4)}">${score.toFixed(3)}</td>`;
      }
      html += `</tr>`;
    }
    html += `</tbody>`;
    heatmapTable.innerHTML = html;
  }

  function renderRetrieval(ranking: { index: number; score: number }[]): void {
    if (!retrievalList) return;

    retrievalList.innerHTML = ranking
      .map(({ index, score }) => {
        const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
        return `
        <li class="se-retrieval-item">
          <span style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--muted); min-width: 24px;">S${index + 1}</span>
          <span class="se-retrieval-text">${escapeHtml(sentences[index])}</span>
          <div class="se-retrieval-bar-bg">
            <div class="se-retrieval-bar-fill" style="width: ${pct}%;"></div>
          </div>
          <span class="se-retrieval-score">${score.toFixed(4)}</span>
        </li>
      `;
      })
      .join("");
  }

  function renderPCA(embeddings: Float32Array[]): void {
    if (!pcaSvg) return;

    const points = projectSentencePCA(embeddings);
    if (!points.length) {
      pcaSvg.innerHTML = "";
      return;
    }

    const width = 400;
    const height = 300;
    const pad = 40;
    const innerW = width - pad * 2;
    const innerH = height - pad * 2;

    const svgParts: string[] = [
      `<line x1="${pad}" y1="${height / 2}" x2="${width - pad}" y2="${height / 2}" stroke="var(--line)" stroke-dasharray="3 3" />`,
      `<line x1="${width / 2}" y1="${pad}" x2="${width / 2}" y2="${height - pad}" stroke="var(--line)" stroke-dasharray="3 3" />`,
      `<text x="${width - pad + 6}" y="${height / 2 + 4}" fill="var(--muted)" font-size="10" font-family="var(--font-mono)">PC1</text>`,
      `<text x="${width / 2 - 12}" y="${pad - 6}" fill="var(--muted)" font-size="10" font-family="var(--font-mono)">PC2</text>`,
    ];

    points.forEach((pt, idx) => {
      const cx = pad + ((pt.x + 1) / 2) * innerW;
      const cy = pad + ((1 - (pt.y + 1) / 2)) * innerH;
      const label = `S${idx + 1}`;
      const title = escapeHtml(sentences[idx]);

      svgParts.push(`
        <g style="cursor: pointer;">
          <title>${label}: ${title}</title>
          <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="7" fill="var(--accent)" stroke="var(--paper)" stroke-width="2" />
          <text x="${(cx + 10).toFixed(1)}" y="${(cy + 4).toFixed(1)}" fill="var(--ink)" font-size="11" font-weight="600" font-family="var(--font-mono)">${label}</text>
        </g>
      `);
    });

    pcaSvg.innerHTML = svgParts.join("");
  }

  async function computeAll(): Promise<void> {
    if (computeBtn) {
      computeBtn.disabled = true;
      computeBtn.textContent = "Computing…";
    }
    if (computeStatusEl) computeStatusEl.textContent = "Running MiniLM encoder over sentences in WebAssembly…";

    try {
      const validSentences = sentences.filter((s) => s.trim().length > 0);
      if (validSentences.length < 2) {
        throw new Error("Provide at least 2 non-empty sentences.");
      }

      cachedEmbeddings = await extractSentenceEmbeddingsBatch(validSentences, (msg, pct) => {
        if (computeStatusEl) {
          computeStatusEl.textContent = `${msg}${pct !== undefined ? ` (${pct}%)` : ""}`;
        }
      });

      const matrix = buildSentenceMatrix(cachedEmbeddings);
      renderHeatmap(matrix);
      renderPCA(cachedEmbeddings);

      // Also run query if query text is present
      if (queryInput && queryInput.value.trim()) {
        const queryEmb = await extractSentenceEmbedding(queryInput.value.trim());
        const ranking = rankBySimilarity(queryEmb, cachedEmbeddings);
        renderRetrieval(ranking);
      }

      if (computeStatusEl) {
        computeStatusEl.textContent = `Successfully computed ${cachedEmbeddings.length} sentence embeddings (384d).`;
      }
    } catch (err) {
      if (computeStatusEl) {
        computeStatusEl.textContent = `Error: ${String(err)}`;
        computeStatusEl.style.color = "var(--red)";
      }
    } finally {
      if (computeBtn) {
        computeBtn.disabled = false;
        computeBtn.textContent = "Recompute Embeddings →";
      }
    }
  }

  // Setup Tab Bar Switching
  document.querySelectorAll<HTMLButtonElement>(".se-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".se-tab-btn").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".se-tab-pane").forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      const target = btn.dataset.tab;
      const pane = document.getElementById(`se-tab-${target}`);
      if (pane) pane.classList.add("active");
    });
  });

  // Setup Preset Buttons
  document.querySelectorAll<HTMLButtonElement>(".se-preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.corpus;
      if (key && PRESETS[key]) {
        sentences = [...PRESETS[key]];
        cachedEmbeddings = null;
        renderSentenceList();
        void computeAll();
      }
    });
  });

  // Add Sentence Button
  if (addBtn) {
    addBtn.addEventListener("click", () => {
      sentences.push("A newly added sentence for semantic evaluation.");
      cachedEmbeddings = null;
      renderSentenceList();
    });
  }

  // Compute Button
  if (computeBtn) {
    computeBtn.addEventListener("click", () => {
      void computeAll();
    });
  }

  // Search Query Button
  if (queryBtn) {
    queryBtn.addEventListener("click", async () => {
      if (!queryInput || !queryInput.value.trim()) return;
      if (!cachedEmbeddings || cachedEmbeddings.length !== sentences.length) {
        await computeAll();
      }
      if (!cachedEmbeddings) return;

      queryBtn.disabled = true;
      queryBtn.textContent = "…";
      try {
        const queryEmb = await extractSentenceEmbedding(queryInput.value.trim());
        const ranking = rankBySimilarity(queryEmb, cachedEmbeddings);
        renderRetrieval(ranking);
      } finally {
        queryBtn.disabled = false;
        queryBtn.textContent = "Search";
      }
    });
  }

  if (queryInput) {
    queryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        queryBtn?.click();
      }
    });
  }

  // Initial render
  renderSentenceList();
}
