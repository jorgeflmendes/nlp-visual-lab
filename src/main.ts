import "./styles.css";
import {
  allCategories,
  foundationsTopics,
  structuredTopics,
  neuralTopics,
  type Category,
  type DeepLearningTopic,
} from "./data/topics.ts";
import { getTopicIcon } from "./data/icons.ts";
import "./deepLearning.css";
import "./methodWorkspace.css";
import { getInitialTheme, applyTheme, toggleTheme } from "./theme.ts";

// Initialize theme on script load
const initialTheme = getInitialTheme();
applyTheme(initialTheme);

const appRoot = document.querySelector<HTMLDivElement>("#app");
if (!appRoot) throw new Error("App root not found");
const app = appRoot;

export async function typesetMath(root: ParentNode = document): Promise<void> {
  const elements = root.querySelectorAll<HTMLElement>("[data-latex]");
  if (elements.length === 0) return;
  const [{ default: katex }] = await Promise.all([
    import("katex"),
    import("katex/dist/katex.min.css"),
  ]);
  elements.forEach((element) => {
    katex.render(element.dataset.latex ?? "", element, {
      displayMode: element.dataset.display === "true",
      throwOnError: false,
      strict: false,
    });
  });
}

function route(): string {
  return location.hash.replace(/^#\/?/, "") || "home";
}

function themeToggleIcon(): string {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  // Sun icon for dark mode (click to switch to light), Moon icon for light mode (click to switch to dark)
  return isLight
    ? `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`
    : `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`;
}

function sidebar(active: string): string {
  const activeCategory = foundationsTopics.some((topic) => topic.slug === active)
    ? "foundations"
    : structuredTopics.some((topic) => topic.slug === active)
    ? "structured"
    : neuralTopics.some((topic) => topic.slug === active)
    ? "neural"
    : active;
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  return `
    <header class="site-header">
      <a class="brand" href="#/home"><img src="${import.meta.env.BASE_URL}logo.svg" alt="NLP Visual Lab" width="168" height="40" /></a>
      <div class="header-right">
        <nav aria-label="Categories">
          ${allCategories.map((category) => `
            <a href="#/${category.id}" class="${activeCategory === category.id ? "active" : ""}" ${activeCategory === category.id ? 'aria-current="page"' : ""}>
              <small>${category.index}</small><span>${category.title}</span>
            </a>
          `).join("")}
        </nav>
        <button id="theme-toggle-btn" class="theme-toggle" type="button" aria-label="Toggle theme (${isLight ? "light" : "dark"} mode active)" title="Switch to ${isLight ? "dark" : "light"} mode">
          ${themeToggleIcon()}
        </button>
      </div>
    </header>
  `;
}

function layout(active: string, content: string): void {
  app.innerHTML = `${sidebar(active)}<main id="content" class="content">${content}</main>`;
}

function home(): string {
  return `
    <header class="home-header">
      <p class="kicker">Natural Language Processing</p>
      <h1>Learn by following the calculation.</h1>
      <p>Visual, step-by-step execution of classical NLP algorithms and neural models. Run tokenizers, dynamic programming matrices, and local inference entirely in your browser.</p>
    </header>
    <section class="home-pillars" aria-label="Taxonomy Pillars">
      ${allCategories.map((category) => `
        <div class="pillar-column">
          <header class="pillar-header">
            <div class="pillar-eyebrow-row">
              <span class="pillar-badge">Pillar ${category.index}</span>
              <span class="pillar-count">${category.topics.length} laboratories</span>
            </div>
            <h2><a href="#/${category.id}">${category.subtitle} <span class="pillar-link-arrow" aria-hidden="true">→</span></a></h2>
            <p>${category.description}</p>
          </header>
          <div class="pillar-labs" aria-label="${category.subtitle} laboratories">
            ${category.topics.map((topic, index) => `
              <a class="lab-card" href="#/${topic.slug}">
                <div class="lab-card-top">
                  <div class="lab-card-icon-wrap">
                    ${getTopicIcon(topic.slug, "lab-card-icon")}
                    <span class="lab-card-index">${category.index}.${index + 1}</span>
                  </div>
                  ${topic.tag ? `<span class="lab-card-tag">${topic.tag}</span>` : ""}
                </div>
                <div class="lab-card-main">
                  <h3>${topic.title}</h3>
                  <p>${topic.summary}</p>
                </div>
                <div class="lab-card-footer">
                  ${topic.stages?.length ? `
                    <div class="lab-card-stages" aria-hidden="true">
                      ${topic.stages.map((stage) => `<span>${stage}</span>`).join('<b>→</b>')}
                    </div>
                  ` : ""}
                  <span class="lab-card-arrow" aria-hidden="true">→</span>
                </div>
              </a>
            `).join("")}
          </div>
        </div>
      `).join("")}
    </section>
  `;
}

function categoryView(category: Category): string {
  return `
    <section class="dl-category">
      <header>
        <p class="dl-eyebrow">${category.eyebrow}</p>
        <h1>${category.subtitle}</h1>
        <p>${category.description}</p>
      </header>
      <div class="dl-catalog" aria-label="${category.subtitle}">
        ${category.topics.map((topic, index) => `
          <a href="#/${topic.slug}">
            <span class="dl-catalog-index">0${index + 1}</span>
            <div class="dl-catalog-title">
              <h2>${topic.title}</h2>
              <p>${topic.summary}</p>
            </div>
            ${topic.stages?.length ? `
              <div class="dl-catalog-flow" aria-hidden="true">
                ${topic.stages.map((stage) => `<span>${stage}</span>`).join('<b>→</b>')}
              </div>
            ` : ""}
            <span class="dl-catalog-arrow" aria-hidden="true">↗</span>
          </a>
        `).join("")}
      </div>
      ${category.id === "neural" ? `<p class="dl-catalog-note">Published weights. Local inference. Every displayed activation comes from your run.</p>` : ""}
    </section>
  `;
}

async function render(): Promise<void> {
  const current = route();
  if (current === "home") {
    layout("", home());
    window.scrollTo(0, 0);
    return;
  }

  // Redirect legacy routes for backward compatibility
  if (current === "solvers") {
    location.hash = "#/structured";
    return;
  }
  if (current === "deep-learning") {
    location.hash = "#/neural";
    return;
  }

  // Category view routes
  const matchedCategory = allCategories.find((cat) => cat.id === current);
  if (matchedCategory) {
    layout(current, categoryView(matchedCategory));
    void typesetMath();
    window.scrollTo(0, 0);
    return;
  }

  // Neural topics
  const neuralTopic = neuralTopics.find((candidate) => candidate.slug === current);
  if (neuralTopic) {
    if (neuralTopic.kind === "sentence-embeddings") {
      const { sentenceEmbeddingsPage, bindSentenceEmbeddings } = await import("./sentenceEmbeddingsLab.ts");
      if (route() !== current) return;
      layout(current, sentenceEmbeddingsPage(neuralTopic));
      bindSentenceEmbeddings(neuralTopic, typesetMath);
      void typesetMath();
      window.scrollTo(0, 0);
      return;
    }
    const { deepLearningPage, bindDeepLearning } = await import("./deepLearning.ts");
    if (route() !== current) return;
    layout(current, deepLearningPage(neuralTopic));
    bindDeepLearning(neuralTopic, typesetMath);
    void typesetMath();
    window.scrollTo(0, 0);
    return;
  }

  // BPE (Foundations topic running via interactive BPE lab)
  if (current === "bpe") {
    const bpeTopic = foundationsTopics.find((candidate) => candidate.slug === "bpe") as DeepLearningTopic;
    const { deepLearningPage, bindDeepLearning } = await import("./deepLearning.ts");
    if (route() !== current) return;
    layout(current, deepLearningPage(bpeTopic));
    bindDeepLearning(bpeTopic, typesetMath);
    void typesetMath();
    window.scrollTo(0, 0);
    return;
  }

  // Word Embeddings (Foundations vector algebra lab)
  if (current === "word-embeddings") {
    const { wordEmbeddingsPage, bindWordEmbeddings } = await import("./wordEmbeddingsLab.ts");
    if (route() !== current) return;
    const topic = foundationsTopics.find((candidate) => candidate.slug === "word-embeddings") ?? {
      slug: "word-embeddings",
      title: "Word Embeddings",
      summary: "Explore 100-dimensional GloVe representations over a 20,000-word vocabulary.",
      tag: "GloVe & Vector Algebra",
      stages: ["Vectors", "Cosine", "Analogy", "PCA"],
    };
    layout(current, wordEmbeddingsPage(topic));
    void bindWordEmbeddings();
    void typesetMath();
    window.scrollTo(0, 0);
    return;
  }

  // Structured and Foundations algorithmic methods
  const topic = [...foundationsTopics, ...structuredTopics].find((candidate) => candidate.slug === current);
  if (!topic) {
    layout("", `<section class="not-found"><h1>Topic not found</h1><a href="#/home">Return to contents</a></section>`);
    return;
  }

  const method = topic.slug === "edit-distance" ? await import("./editDistanceLab.ts").then((module) => ({ page: module.editDistancePage, bind: module.bindEditDistance }))
    : topic.slug === "n-grams" ? await import("./bigramLab.ts").then((module) => ({ page: module.ngramsPage, bind: module.bindNgrams }))
    : topic.slug === "viterbi" ? await import("./viterbiLab.ts").then((module) => ({ page: module.viterbiPage, bind: module.bindViterbi }))
    : topic.slug === "cky" ? await import("./ckyLab.ts").then((module) => ({ page: module.ckyPage, bind: module.bindCky }))
    : await import("./tfidfLab.ts").then((module) => ({ page: module.tfidfPage, bind: module.bindTfidf }));

  if (route() !== current) return;
  layout(topic.slug, method.page(topic));
  method.bind();
  void typesetMath();
  window.scrollTo(0, 0);
}

// Global delegated listener for the theme toggle button
document.addEventListener("click", (event) => {
  const target = event.target as HTMLElement | null;
  const button = target?.closest<HTMLButtonElement>("#theme-toggle-btn");
  if (!button) return;
  const newTheme = toggleTheme();
  button.setAttribute("aria-label", `Toggle theme (${newTheme} mode active)`);
  button.title = `Switch to ${newTheme === "dark" ? "light" : "dark"} mode`;
  button.innerHTML = themeToggleIcon();
});

window.addEventListener("hashchange", render);
render();
