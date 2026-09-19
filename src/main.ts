import "./styles.css";
import { deepLearningTopics, topics } from "./data/topics.ts";
import "./deepLearning.css";
import "./methodWorkspace.css";

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

function sidebar(active: string): string {
  const navigation = [{ slug: "solvers", title: "Solvers" }, { slug: "deep-learning", title: "Deep Learning" }];
  const activeCategory = deepLearningTopics.some((topic) => topic.slug === active) ? "deep-learning" : topics.some((topic) => topic.slug === active) ? "solvers" : active;
  return `
    <header class="site-header">
      <a class="brand" href="#/home"><img src="${import.meta.env.BASE_URL}logo.svg" alt="NLP Visual Lab" width="168" height="40" /></a>
      <nav aria-label="Categories">
        ${navigation.map((topic, index) => `
          <a href="#/${topic.slug}" class="${activeCategory === topic.slug ? "active" : ""}" ${activeCategory === topic.slug ? 'aria-current="page"' : ""}>
            <small>0${index + 1}</small><span>${topic.title}</span>
          </a>
        `).join("")}
      </nav>
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

window.addEventListener("hashchange", render);
render();
