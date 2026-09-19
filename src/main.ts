import "./styles.css";
import { deepLearningTopics, topics } from "./data/topics.ts";
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
  const navigation = [{ slug: "solvers", title: "Solvers" }, { slug: "deep-learning", title: "Deep Learning" }];
  const activeCategory = deepLearningTopics.some((topic) => topic.slug === active) ? "deep-learning" : topics.some((topic) => topic.slug === active) ? "solvers" : active;
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  return `
    <header class="site-header">
      <a class="brand" href="#/home"><img src="${import.meta.env.BASE_URL}logo.svg" alt="NLP Visual Lab" width="168" height="40" /></a>
      <div class="header-right">
        <nav aria-label="Categories">
          ${navigation.map((topic, index) => `
            <a href="#/${topic.slug}" class="${activeCategory === topic.slug ? "active" : ""}" ${activeCategory === topic.slug ? 'aria-current="page"' : ""}>
              <small>0${index + 1}</small><span>${topic.title}</span>
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
      <p>Choose a method and inspect every decision made by the algorithm.</p>
    </header>
    <section class="topic-list" aria-label="Categories">
      <a href="#/solvers">
        <small>01 / Five solvers</small>
        <div><h2>Solvers</h2><p>Work through edit distance, bigram probabilities, Viterbi HMM, CKY parsing and TF-IDF. Follow the tables, formulas and decisions.</p></div>
        <span aria-hidden="true">→</span>
      </a>
      <a class="topic-category" href="#/deep-learning">
        <small>02 / Four models</small>
        <div><h2>Deep Learning</h2><p>Run LSTM, Seq2Seq, Attention and Transformer models locally. Inspect the values behind their predictions.</p></div>
        <span aria-hidden="true">→</span>
      </a>
    </section>
  `;
}

function solversCategory(): string {
  const stages = [
    ["Matrix", "Minimum paths"],
    ["Counts", "Sentence factors"],
    ["SS & BP tables", "Best sequence"],
    ["Grammar", "Chart", "Derivations"],
    ["Counts", "TF", "DF / IDF", "Vectors"],
  ];
  return `<section class="dl-category"><header><p class="dl-eyebrow">Solvers / classical methods</p><h1>Work through the solution.</h1><p>Edit an example and follow its complete calculation. Inspect the tables, compare candidates and trace the result.</p></header><div class="dl-catalog" aria-label="Solvers">${topics.map((topic, index) => `<a href="#/${topic.slug}"><span class="dl-catalog-index">0${index + 1}</span><div class="dl-catalog-title"><h2>${topic.title}</h2><p>${topic.summary}</p></div><div class="dl-catalog-flow">${stages[index].map((stage) => `<span>${stage}</span>`).join('<b aria-hidden="true">→</b>')}</div><span class="dl-catalog-arrow" aria-hidden="true">↗</span></a>`).join("")}</div></section>`;
}

async function render(): Promise<void> {
  const current = route();
  if (current === "home") {
    layout("", home());
    window.scrollTo(0, 0);
    return;
  }
  if (current === "solvers") {
    layout(current, solversCategory());
    window.scrollTo(0, 0);
    return;
  }
  if (current === "deep-learning") {
    const { deepLearningCategory } = await import("./deepLearning.ts");
    if (route() !== current) return;
    layout(current, deepLearningCategory(deepLearningTopics));
    void typesetMath();
    window.scrollTo(0, 0);
    return;
  }
  const deepTopic = deepLearningTopics.find((candidate) => candidate.slug === current);
  if (deepTopic) {
    const { deepLearningPage, bindDeepLearning } = await import("./deepLearning.ts");
    if (route() !== current) return;
    layout(current, deepLearningPage(deepTopic));
    bindDeepLearning(deepTopic, typesetMath);
    void typesetMath();
    window.scrollTo(0, 0);
    return;
  }
  const topic = topics.find((candidate) => candidate.slug === current);
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
