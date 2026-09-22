import type { Topic } from "./data/topics.ts";
import { getTopicIcon } from "./data/icons.ts";
import {
  type WordEmbeddingIndex,
  type ExpressionEvalResult,
  type ExtraPCAPoint,
  createEmbeddingIndex,
  getWordVector,
  evaluateExpression,
  findNearestNeighbors,
  computeSimilarityMatrix,
  project2DPCA,
} from "./wordEmbeddingsAlgorithm.ts";
import "./wordEmbeddingsLab.css";

const foundationsRoutes: [string, string][] = [
  ["bpe", "BPE"],
  ["edit-distance", "Edit distance"],
  ["tfidf", "TF-IDF"],
  ["word-embeddings", "Word embeddings"],
];

let cachedIndex: WordEmbeddingIndex | null = null;

function escapeHtml(s: string): string {
  return String(s).replace(/[&<>'"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[c] ?? c
  );
}

export function wordEmbeddingsPage(topic: Topic): string {
  return `<article class="we-workspace dl-workspace" data-kind="word-embeddings">
    <nav class="dl-model-nav" aria-label="Foundations laboratories">
      <a class="dl-back" href="#/foundations" aria-label="All foundations laboratories">← <span>Foundations</span></a>
      <div>${foundationsRoutes.map(([slug, label]) => `<a href="#/${slug}" ${topic.slug === slug ? 'aria-current="page"' : ""}>${label}</a>`).join("")}</div>
    </nav>

    <header class="dl-page-heading">
      <div>
        <p class="dl-eyebrow">Foundations / Semantic vector space</p>
        <div class="dl-heading-title-row">
          ${getTopicIcon(topic.slug, "dl-heading-icon")}
          <h1>Word Embeddings &amp; Vector Algebra.</h1>
        </div>
        <p>Explore 100-dimensional GloVe representations over a 20,000-word vocabulary. Evaluate arbitrary linear expressions, discover semantic neighborhoods, and project unconstrained vocabulary spaces via 2D PCA.</p>
      </div>
      <span id="we-status" class="dl-state">Loading vectors...</span>
    </header>

    <div id="we-loading" class="we-loading-overlay">
      <p id="we-loading-msg" style="font-weight: 500;">Downloading GloVe 100d vectors (8 MB)...</p>
      <div class="we-progress-bar">
        <div id="we-progress-fill" class="we-progress-fill"></div>
      </div>
      <span style="font-size: 0.78rem; color: var(--muted);">Unit-normalized float32 vectors processed directly in memory</span>
    </div>

    <div id="we-content" style="display: none; flex-direction: column; gap: 20px;">
      <!-- Row 1: Arbitrary Vector Arithmetic and Nearest Neighbors -->
      <div class="we-grid-2">
        <!-- Vector Arithmetic Box -->
        <section class="we-card" aria-labelledby="arithmetic-heading">
          <div class="we-card-header">
            <h2 id="arithmetic-heading" class="we-card-title">Vector Arithmetic Calculator</h2>
            <span class="we-card-badge">&sum; s_i w_i v_i</span>
          </div>

          <div class="we-analogy-presets">
            <span style="color: var(--muted); font-size: 0.75rem;">Presets:</span>
            <button type="button" class="we-preset-btn we-expr-preset" data-expr="king - man + woman">king - man + woman</button>
            <button type="button" class="we-preset-btn we-expr-preset" data-expr="paris - france + japan">paris - france + japan</button>
            <button type="button" class="we-preset-btn we-expr-preset" data-expr="2 * paris - france + 0.5 * tokyo">2*paris - france + 0.5*tokyo</button>
            <button type="button" class="we-preset-btn we-expr-preset" data-expr="walked - walking + swimming">walked - walking + swimming</button>
            <button type="button" class="we-preset-btn we-expr-preset" data-expr="doctor - man + woman">doctor - man + woman</button>
            <button type="button" class="we-preset-btn we-expr-preset" data-expr="german + beer - bavaria">german + beer - bavaria</button>
          </div>

          <div style="display: flex; flex-direction: column; gap: 6px;">
            <label for="we-expr-input" style="font-size: 0.75rem; color: var(--muted);">Arbitrary linear combination expression:</label>
            <div style="display: flex; gap: 8px;">
              <input id="we-expr-input" class="we-analogy-input" type="text" value="king - man + woman" placeholder="e.g. 2*paris - france + 0.5*tokyo" autocomplete="off" />
              <button id="we-expr-eval-btn" type="button" class="we-action-btn">Evaluate</button>
            </div>
          </div>

          <div id="we-terms-breakdown" class="we-terms-breakdown"></div>

          <div id="we-analogy-result" class="we-analogy-result-box">
            <div class="we-winner-row">
              <div>
                <span style="font-size: 0.75rem; color: var(--muted); display: block;">Closest vector projection:</span>
                <span id="we-winner-word" class="we-winner-word">—</span>
              </div>
              <span id="we-winner-score" class="we-winner-score">—</span>
            </div>
            <ul id="we-candidates-list" class="we-candidates-list"></ul>
          </div>
        </section>

        <!-- Nearest Neighbors Box -->
        <section class="we-card" aria-labelledby="neighbors-heading">
          <div class="we-card-header">
            <h2 id="neighbors-heading" class="we-card-title">Nearest Neighbors Explorer</h2>
            <span class="we-card-badge">Cosine Similarity</span>
          </div>

          <div style="display: flex; flex-direction: column; gap: 6px;">
            <label for="we-neighbor-input" style="font-size: 0.75rem; color: var(--muted);">Lookup query word:</label>
            <div style="display: flex; gap: 8px;">
              <input id="we-neighbor-input" class="we-analogy-input" type="text" value="science" placeholder="Enter any vocabulary word..." autocomplete="off" />
              <button id="we-neighbor-btn" type="button" class="we-action-btn">Search</button>
            </div>
          </div>

          <div class="we-analogy-presets">
            <span style="color: var(--muted); font-size: 0.75rem;">Try:</span>
            <button type="button" class="we-preset-btn we-nn-preset" data-word="computer">computer</button>
            <button type="button" class="we-preset-btn we-nn-preset" data-word="philosophy">philosophy</button>
            <button type="button" class="we-preset-btn we-nn-preset" data-word="galaxy">galaxy</button>
            <button type="button" class="we-preset-btn we-nn-preset" data-word="coffee">coffee</button>
            <button type="button" class="we-preset-btn we-nn-preset" data-word="quantum">quantum</button>
            <button type="button" class="we-preset-btn we-nn-preset" data-word="democracy">democracy</button>
          </div>

          <div class="we-analogy-result-box" style="flex: 1;">
            <ul id="we-neighbor-list" class="we-candidates-list"></ul>
          </div>
        </section>
      </div>

      <!-- Row 2: Cosine Similarity Heatmap and 2D PCA Scatter -->
      <div class="we-grid-2">
        <!-- Heatmap Box -->
        <section class="we-card" aria-labelledby="heatmap-heading">
          <div class="we-card-header">
            <h2 id="heatmap-heading" class="we-card-title">Pairwise Cosine Matrix</h2>
            <span class="we-card-badge">[-1, 1]</span>
          </div>

          <div class="we-analogy-presets">
            <span style="color: var(--muted); font-size: 0.75rem;">Sets:</span>
            <button type="button" class="we-preset-btn we-matrix-preset" data-words="king, queen, man, woman, prince, princess, dog, cat">Royalty &amp; Animals</button>
            <button type="button" class="we-preset-btn we-matrix-preset" data-words="paris, france, london, england, tokyo, japan, rome, italy">Capitals</button>
            <button type="button" class="we-preset-btn we-matrix-preset" data-words="computer, software, hardware, algorithm, network, internet">Technology</button>
            <button type="button" class="we-preset-btn we-matrix-preset" data-words="happy, sad, joyful, angry, cheerful, depressed">Emotions</button>
          </div>

          <div>
            <label for="we-matrix-input" style="font-size: 0.75rem; color: var(--muted); display: block; margin-bottom: 4px;">Words (comma-separated):</label>
            <input id="we-matrix-input" class="we-analogy-input" type="text" value="king, queen, man, woman, prince, princess, dog, cat" autocomplete="off" />
          </div>

          <div style="overflow-x: auto;">
            <table id="we-heatmap-table" class="we-heatmap-table"></table>
          </div>
        </section>

        <!-- PCA 2D Scatter Box -->
        <section class="we-card" aria-labelledby="pca-heading">
          <div class="we-card-header">
            <h2 id="pca-heading" class="we-card-title">2D PCA Spatial Projection</h2>
            <div style="display: flex; align-items: center; gap: 8px;">
              <label style="font-size: 0.72rem; color: var(--muted); display: flex; align-items: center; gap: 4px; cursor: pointer;">
                <input type="checkbox" id="we-pca-include-result" checked />
                <span>Plot arithmetic target</span>
              </label>
              <span class="we-card-badge">PC1 vs PC2</span>
            </div>
          </div>

          <div class="we-analogy-presets">
            <span style="color: var(--muted); font-size: 0.75rem;">Clusters:</span>
            <button type="button" class="we-preset-btn we-pca-preset" data-words="king, queen, prince, princess, man, woman, emperor, empress">Royalty</button>
            <button type="button" class="we-preset-btn we-pca-preset" data-words="paris, france, london, england, tokyo, japan, berlin, germany, rome, italy">Capitals</button>
            <button type="button" class="we-preset-btn we-pca-preset" data-words="dog, puppy, cat, kitten, wolf, lion, tiger, bear, elephant">Animals</button>
            <button type="button" class="we-preset-btn we-pca-preset" data-words="hot, cold, warm, freezing, fire, ice, summer, winter">Temperature</button>
            <button type="button" class="we-preset-btn we-pca-preset" data-words="doctor, nurse, hospital, medicine, patient, clinic, disease">Medicine</button>
          </div>

          <div>
            <label for="we-pca-words-input" style="font-size: 0.75rem; color: var(--muted); display: block; margin-bottom: 4px;">Words to project in 2D space (comma-separated):</label>
            <input id="we-pca-words-input" class="we-analogy-input" type="text" value="king, queen, man, woman, prince, princess, emperor, empress" autocomplete="off" />
          </div>

          <p style="font-size: 0.74rem; color: var(--muted); margin: 0;">
            Centered covariance eigen-decomposition (power iteration with Gram-Schmidt orthogonalization). Points scale to dynamic coordinate ranges.
          </p>

          <div class="we-pca-container">
            <svg id="we-pca-svg" class="we-pca-svg" viewBox="0 0 400 300"></svg>
          </div>
        </section>
      </div>
    </div>
  </article>`;
}

export async function bindWordEmbeddings(): Promise<void> {
  const loadingEl = document.getElementById("we-loading");
  const contentEl = document.getElementById("we-content");
  const statusEl = document.getElementById("we-status");
  const progressFill = document.getElementById("we-progress-fill");
  const loadingMsg = document.getElementById("we-loading-msg");

  if (!cachedIndex) {
    try {
      if (progressFill) progressFill.style.width = "20%";
      if (loadingMsg) loadingMsg.textContent = "Loading GloVe vocabulary index...";

      const vocabRes = await fetch("models/glove-20k-vocab.json");
      if (!vocabRes.ok) throw new Error(`Failed to fetch vocabulary: ${vocabRes.status}`);
      const vocab: string[] = await vocabRes.json();

      if (progressFill) progressFill.style.width = "50%";
      if (loadingMsg) loadingMsg.textContent = "Loading 100d binary vectors (8 MB)...";

      const binRes = await fetch("models/glove-20k-100d.bin");
      if (!binRes.ok) throw new Error(`Failed to fetch binary vectors: ${binRes.status}`);
      const buffer = await binRes.arrayBuffer();

      if (progressFill) progressFill.style.width = "100%";
      cachedIndex = createEmbeddingIndex(vocab, buffer, 100);
    } catch (err) {
      if (statusEl) {
        statusEl.textContent = "Error loading model";
        statusEl.style.color = "var(--red)";
      }
      if (loadingMsg) {
        loadingMsg.textContent = `Error loading GloVe vectors: ${String(err)}`;
      }
      return;
    }
  }

  const index = cachedIndex;
  if (statusEl) {
    statusEl.textContent = `Ready (${index.vocab.length.toLocaleString()} words, 100d)`;
  }
  if (loadingEl) loadingEl.style.display = "none";
  if (contentEl) contentEl.style.display = "flex";

  // Elements
  const exprInput = document.getElementById("we-expr-input") as HTMLInputElement | null;
  const exprEvalBtn = document.getElementById("we-expr-eval-btn");
  const termsBreakdown = document.getElementById("we-terms-breakdown");
  const winnerWord = document.getElementById("we-winner-word");
  const winnerScore = document.getElementById("we-winner-score");
  const candidatesList = document.getElementById("we-candidates-list");

  const neighborInput = document.getElementById("we-neighbor-input") as HTMLInputElement | null;
  const neighborBtn = document.getElementById("we-neighbor-btn");
  const neighborList = document.getElementById("we-neighbor-list");

  const matrixInput = document.getElementById("we-matrix-input") as HTMLInputElement | null;
  const heatmapTable = document.getElementById("we-heatmap-table");

  const pcaWordsInput = document.getElementById("we-pca-words-input") as HTMLInputElement | null;
  const pcaIncludeResult = document.getElementById("we-pca-include-result") as HTMLInputElement | null;
  const pcaSvg = document.getElementById("we-pca-svg") as unknown as SVGElement | null;

  let lastExpressionResult: ExpressionEvalResult | null = null;

  // Update Vector Arithmetic
  function updateArithmetic() {
    if (!exprInput || !winnerWord || !winnerScore || !candidatesList) return;

    const rawExpr = exprInput.value.trim();
    if (!rawExpr) {
      winnerWord.textContent = "—";
      winnerScore.textContent = "—";
      if (termsBreakdown) termsBreakdown.innerHTML = "";
      candidatesList.innerHTML = `<li style="color: var(--muted); font-size: 0.8rem;">Enter a vector linear combination (e.g. "king - man + woman" or "2 * paris - france + 0.5 * tokyo").</li>`;
      lastExpressionResult = null;
      updatePCA();
      return;
    }

    const res = evaluateExpression(index, rawExpr, 6, true);
    lastExpressionResult = res;

    if (!res || res.terms.length === 0) {
      winnerWord.textContent = "Invalid";
      winnerScore.textContent = "";
      if (termsBreakdown) termsBreakdown.innerHTML = "";
      candidatesList.innerHTML = `<li style="color: var(--red); font-size: 0.8rem;">Could not parse valid terms. Use format: wordA - wordB + wordC or scalar * word.</li>`;
      updatePCA();
      return;
    }

    // Render terms breakdown chips
    if (termsBreakdown) {
      termsBreakdown.innerHTML = res.terms
        .map((t) => {
          const hasVec = getWordVector(index, t.word) !== null;
          const signStr = t.sign === -1 ? "-" : "+";
          const weightStr = t.weight !== 1.0 ? `${t.weight} &times; ` : "";
          const badgeClass = hasVec ? "we-term-chip" : "we-term-chip we-term-missing";
          const title = hasVec ? `In vocabulary: ${t.word}` : `Missing from vocabulary: "${t.word}"`;
          return `<span class="${badgeClass}" title="${title}"><code>${signStr} ${weightStr}${escapeHtml(t.word)}</code></span>`;
        })
        .join("");
    }

    if (res.usedWords.length === 0) {
      winnerWord.textContent = "None found";
      winnerScore.textContent = "";
      candidatesList.innerHTML = `<li style="color: var(--red); font-size: 0.8rem;">None of the words in the expression exist in the 20,000 vocabulary: ${res.missingWords.map((w) => `"${w}"`).join(", ")}</li>`;
      updatePCA();
      return;
    }

    if (res.candidates.length === 0) {
      winnerWord.textContent = "No match";
      winnerScore.textContent = "";
      candidatesList.innerHTML = `<li style="color: var(--muted); font-size: 0.8rem;">No nearest neighbors found outside query terms.</li>`;
      updatePCA();
      return;
    }

    const top = res.candidates[0];
    winnerWord.textContent = top.word;
    winnerScore.textContent = `cos: ${top.similarity.toFixed(4)}`;

    let missingNotice = "";
    if (res.missingWords.length > 0) {
      missingNotice = `<li style="color: var(--amber, #d97706); font-size: 0.75rem; margin-bottom: 4px;">Note: skipped words not in 20k vocabulary: ${res.missingWords.map((w) => `"${escapeHtml(w)}"`).join(", ")}</li>`;
    }

    candidatesList.innerHTML =
      missingNotice +
      res.candidates
        .map(
          (c) => `
          <li class="we-candidate-item">
            <span class="we-candidate-name">${escapeHtml(c.word)}</span>
            <div class="we-candidate-bar-bg">
              <div class="we-candidate-bar-fill" style="width: ${Math.max(0, Math.min(100, c.similarity * 100))}%;"></div>
            </div>
            <span class="we-candidate-score">${c.similarity.toFixed(3)}</span>
          </li>
        `
        )
        .join("");

    updatePCA();
  }

  // Update Nearest Neighbors
  function updateNeighbors() {
    if (!neighborInput || !neighborList) return;
    const query = neighborInput.value.trim().toLowerCase();
    if (!query) {
      neighborList.innerHTML = "";
      return;
    }

    const vec = getWordVector(index, query);
    if (!vec) {
      neighborList.innerHTML = `<li style="color: var(--red); font-size: 0.8rem;">"${escapeHtml(query)}" is not in the 20k vocabulary.</li>`;
      return;
    }

    const neighbors = findNearestNeighbors(index, vec, 10, new Set([query]));
    neighborList.innerHTML = neighbors
      .map(
        (c) => `
        <li class="we-candidate-item">
          <span class="we-candidate-name">${escapeHtml(c.word)}</span>
          <div class="we-candidate-bar-bg">
            <div class="we-candidate-bar-fill" style="width: ${Math.max(0, Math.min(100, c.similarity * 100))}%;"></div>
          </div>
          <span class="we-candidate-score">${c.similarity.toFixed(3)}</span>
        </li>
      `
      )
      .join("");
  }

  // Update Heatmap
  function updateMatrix() {
    if (!matrixInput || !heatmapTable) return;
    const raw = matrixInput.value
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter((s) => s.length > 0);

    const { words, matrix } = computeSimilarityMatrix(index, raw);
    if (words.length === 0) {
      heatmapTable.innerHTML = `<tr><td style="color: var(--muted); padding: 12px;">No valid words found in vocabulary.</td></tr>`;
      return;
    }

    let tableHtml = "<thead><tr><th></th>";
    for (const w of words) {
      tableHtml += `<th>${escapeHtml(w)}</th>`;
    }
    tableHtml += "</tr></thead><tbody>";

    for (let i = 0; i < words.length; i++) {
      tableHtml += `<tr><th>${escapeHtml(words[i])}</th>`;
      for (let j = 0; j < words.length; j++) {
        const val = matrix[i][j];
        const intensity = Math.max(0, Math.min(1, val));
        const bg = `rgba(37, 99, 235, ${(intensity * 0.85 + 0.05).toFixed(2)})`;
        const textColor = intensity > 0.6 ? "#fff" : "var(--ink)";
        tableHtml += `<td class="we-heatmap-cell" style="background: ${bg}; color: ${textColor}; font-weight: 500;" title="${words[i]} &times; ${words[j]}: ${val.toFixed(4)}">${val.toFixed(2)}</td>`;
      }
      tableHtml += "</tr>";
    }
    tableHtml += "</tbody>";
    heatmapTable.innerHTML = tableHtml;
  }

  // Update PCA Projection
  function updatePCA() {
    if (!pcaWordsInput || !pcaSvg) return;

    const raw = pcaWordsInput.value
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter((s) => s.length > 0);

    const extraPoints: ExtraPCAPoint[] = [];
    if (
      pcaIncludeResult?.checked &&
      lastExpressionResult &&
      lastExpressionResult.usedWords.length > 0 &&
      lastExpressionResult.candidates.length > 0
    ) {
      const topCandidate = lastExpressionResult.candidates[0].word;
      extraPoints.push({
        label: `★ result (${topCandidate})`,
        vector: lastExpressionResult.resultVector,
        isResult: true,
      });
    }

    const points = project2DPCA(index, raw, extraPoints);
    renderPCA(pcaSvg, points);
  }

  function renderPCA(
    svg: SVGElement,
    points: { word: string; x: number; y: number; isResult?: boolean }[]
  ) {
    const width = 400;
    const height = 300;
    const pad = 42;

    if (points.length === 0) {
      svg.innerHTML = `
        <text x="${width / 2}" y="${height / 2}" text-anchor="middle" fill="var(--muted)" font-size="12">
          No valid vocabulary words to project
        </text>
      `;
      return;
    }

    let svgInner = `
      <!-- Grid & Axes -->
      <line x1="${pad}" y1="${height / 2}" x2="${width - pad}" y2="${height / 2}" class="we-pca-axis" />
      <line x1="${width / 2}" y1="${pad}" x2="${width / 2}" y2="${height - pad}" class="we-pca-axis" />
      <line x1="${pad}" y1="${pad}" x2="${width - pad}" y2="${pad}" class="we-pca-grid" />
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" class="we-pca-grid" />
      <text x="${width - pad - 6}" y="${height / 2 - 6}" class="we-pca-axis-label" text-anchor="end">PC1</text>
      <text x="${width / 2 + 6}" y="${pad + 12}" class="we-pca-axis-label">PC2</text>
    `;

    for (const pt of points) {
      const cx = width / 2 + pt.x * (width / 2 - pad);
      const cy = height / 2 - pt.y * (height / 2 - pad);

      if (pt.isResult) {
        svgInner += `
          <g class="we-pca-group-result">
            <circle class="we-pca-point-result-ring" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="8" />
            <circle class="we-pca-point-result" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="4.5" />
            <text class="we-pca-label we-pca-label-result" x="${(cx + 8).toFixed(1)}" y="${(cy + 4).toFixed(1)}">${escapeHtml(pt.word)}</text>
          </g>
        `;
      } else {
        svgInner += `
          <g class="we-pca-group">
            <circle class="we-pca-point" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="4" />
            <text class="we-pca-label" x="${(cx + 6).toFixed(1)}" y="${(cy + 3).toFixed(1)}">${escapeHtml(pt.word)}</text>
          </g>
        `;
      }
    }

    svg.innerHTML = svgInner;
  }

  // Event Listeners
  exprInput?.addEventListener("input", updateArithmetic);
  exprInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") updateArithmetic();
  });
  exprEvalBtn?.addEventListener("click", updateArithmetic);

  document.querySelectorAll<HTMLButtonElement>(".we-expr-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (exprInput && btn.dataset.expr) {
        exprInput.value = btn.dataset.expr;
        updateArithmetic();
      }
    });
  });

  neighborInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") updateNeighbors();
  });
  neighborBtn?.addEventListener("click", updateNeighbors);

  document.querySelectorAll<HTMLButtonElement>(".we-nn-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (neighborInput && btn.dataset.word) {
        neighborInput.value = btn.dataset.word;
        updateNeighbors();
      }
    });
  });

  matrixInput?.addEventListener("input", updateMatrix);

  document.querySelectorAll<HTMLButtonElement>(".we-matrix-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (matrixInput && btn.dataset.words) {
        matrixInput.value = btn.dataset.words;
        updateMatrix();
      }
    });
  });

  pcaWordsInput?.addEventListener("input", updatePCA);
  pcaIncludeResult?.addEventListener("change", updatePCA);

  document.querySelectorAll<HTMLButtonElement>(".we-pca-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (pcaWordsInput && btn.dataset.words) {
        pcaWordsInput.value = btn.dataset.words;
        updatePCA();
      }
    });
  });

  // Initial runs
  updateArithmetic();
  updateNeighbors();
  updateMatrix();
  updatePCA();
}
