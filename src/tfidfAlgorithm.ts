export type TfMode = "relative" | "raw" | "binary" | "log";
export type IdfMode = "classic" | "smooth" | "log10" | "log2";
export type VocabOrder = "alpha" | "freq" | "idf";

export interface TfidfOptions {
  tfMode: TfMode;
  idfMode: IdfMode;
  lowercase: boolean;
  removePunctuation: boolean;
  removeStopwords: boolean;
  order: VocabOrder;
  precision: number;
}

export interface DocumentInput {
  name: string;
  text: string;
}

export interface TfidfState {
  docs: DocumentInput[];
  tokenDocs: string[][];
  counts: Map<string, number>[];
  N: number;
  vocab: string[];
  dfs: Map<string, number>;
  idfs: Map<string, number>;
  freq: Map<string, number>;
  tfs: Map<string, number>[];
  weights: Map<string, number>[];
}

export interface CosineResult {
  docAIndex: number;
  docBIndex: number;
  dotProduct: number;
  normA: number;
  normB: number;
  cosine: number;
  contributions: Array<{ term: string; value: number }>;
}

export interface RetrievalResult {
  tokens: string[];
  knownTokens: string[];
  unknownTokens: string[];
  ranking: Array<{ name: string; score: number; docIndex: number }>;
}

export interface ExerciseData {
  count: number;
  length: number;
  N: number;
  df: number;
  tf: number;
  idf: number;
  weight: number;
}

export const ENGLISH_STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
  "has", "have", "he", "her", "his", "i", "in", "is", "it", "its",
  "of", "on", "or", "our", "she", "that", "the", "their", "they",
  "this", "to", "was", "we", "were", "will", "with", "you", "your"
]);

export function tokenize(
  text: string,
  options: Pick<TfidfOptions, "lowercase" | "removePunctuation" | "removeStopwords">
): string[] {
  let s = String(text);
  if (options.lowercase) {
    s = s.toLocaleLowerCase();
  }
  if (options.removePunctuation) {
    try {
      s = s.replace(/[\p{P}\p{S}]+/gu, " ");
    } catch {
      s = s.replace(/[^\w\s]+/g, " ");
    }
  }
  let tokens = s.trim() ? s.trim().split(/\s+/u) : [];
  if (options.removeStopwords) {
    tokens = tokens.filter((t) => !ENGLISH_STOPWORDS.has(t));
  }
  return tokens;
}

export function computeTf(count: number, length: number, mode: TfMode): number {
  if (!count) return 0;
  if (mode === "raw") return count;
  if (mode === "binary") return 1;
  if (mode === "log") return 1 + Math.log(count);
  return length ? count / length : 0;
}

export function computeIdf(N: number, df: number, mode: IdfMode): number {
  if (!N || !df) return 0;
  if (mode === "smooth") return Math.log((N + 1) / (df + 1)) + 1;
  if (mode === "log10") return Math.log10(N / df);
  if (mode === "log2") return Math.log2(N / df);
  return Math.log(N / df);
}

export function computeTfidfState(
  docs: DocumentInput[],
  options: TfidfOptions
): TfidfState {
  const tokenDocs = docs.map((d) => tokenize(d.text, options));
  const counts = tokenDocs.map((tokens) => {
    const map = new Map<string, number>();
    tokens.forEach((t) => map.set(t, (map.get(t) || 0) + 1));
    return map;
  });

  const N = docs.length;
  const terms = new Set<string>();
  tokenDocs.forEach((arr) => arr.forEach((t) => terms.add(t)));
  const vocab = Array.from(terms);

  const dfs = new Map<string, number>(
    vocab.map((t) => [t, counts.reduce((sum, m) => sum + (m.has(t) ? 1 : 0), 0)])
  );

  const idfs = new Map<string, number>(
    vocab.map((t) => [t, computeIdf(N, dfs.get(t) || 0, options.idfMode)])
  );

  const freq = new Map<string, number>(
    vocab.map((t) => [t, counts.reduce((sum, m) => sum + (m.get(t) || 0), 0)])
  );

  if (options.order === "freq") {
    vocab.sort((a, b) => (freq.get(b) || 0) - (freq.get(a) || 0) || a.localeCompare(b));
  } else if (options.order === "idf") {
    vocab.sort((a, b) => (idfs.get(b) || 0) - (idfs.get(a) || 0) || a.localeCompare(b));
  } else {
    vocab.sort((a, b) => a.localeCompare(b));
  }

  const tfs = counts.map((m, j) =>
    new Map<string, number>(
      vocab.map((t) => [t, computeTf(m.get(t) || 0, tokenDocs[j].length, options.tfMode)])
    )
  );

  const weights = tfs.map((m) =>
    new Map<string, number>(
      vocab.map((t) => [t, (m.get(t) || 0) * (idfs.get(t) || 0)])
    )
  );

  return {
    docs,
    tokenDocs,
    counts,
    N,
    vocab,
    dfs,
    idfs,
    freq,
    tfs,
    weights,
  };
}

export function vectorDot(a: Map<string, number>, b: Map<string, number>, vocab: string[]): number {
  let sum = 0;
  for (const t of vocab) {
    sum += (a.get(t) || 0) * (b.get(t) || 0);
  }
  return sum;
}

export function vectorNorm(a: Map<string, number>, vocab: string[]): number {
  return Math.sqrt(vectorDot(a, a, vocab));
}

export function computeCosine(
  state: TfidfState,
  docAIndex: number,
  docBIndex: number
): CosineResult {
  const safeA = Math.max(0, Math.min(docAIndex, state.docs.length - 1));
  const safeB = Math.max(0, Math.min(docBIndex, state.docs.length - 1));
  const A = state.weights[safeA] || new Map<string, number>();
  const B = state.weights[safeB] || new Map<string, number>();

  const dp = vectorDot(A, B, state.vocab);
  const na = vectorNorm(A, state.vocab);
  const nb = vectorNorm(B, state.vocab);
  const cos = na && nb ? dp / (na * nb) : 0;

  const contributions = state.vocab
    .map((t) => ({ term: t, value: (A.get(t) || 0) * (B.get(t) || 0) }))
    .filter((x) => x.value > 0)
    .sort((x, y) => y.value - x.value);

  return {
    docAIndex: safeA,
    docBIndex: safeB,
    dotProduct: dp,
    normA: na,
    normB: nb,
    cosine: cos,
    contributions,
  };
}

export function performRetrieval(
  state: TfidfState,
  queryText: string,
  options: TfidfOptions
): RetrievalResult {
  const tokens = tokenize(queryText, options);
  const qCount = new Map<string, number>();
  tokens.forEach((t) => qCount.set(t, (qCount.get(t) || 0) + 1));

  const qVector = new Map<string, number>(
    state.vocab.map((t) => [
      t,
      computeTf(qCount.get(t) || 0, tokens.length, options.tfMode) * (state.idfs.get(t) || 0),
    ])
  );

  const qNorm = vectorNorm(qVector, state.vocab);
  const knownTokens = tokens.filter((t) => state.vocab.includes(t));
  const unknownTokens = tokens.filter((t) => !state.vocab.includes(t));

  const ranking = state.docs
    .map((d, i) => {
      const docVec = state.weights[i];
      const dp = vectorDot(qVector, docVec, state.vocab);
      const dNorm = vectorNorm(docVec, state.vocab);
      const score = qNorm && dNorm ? dp / (qNorm * dNorm) : 0;
      return { name: d.name, score, docIndex: i };
    })
    .sort((a, b) => b.score - a.score);

  return {
    tokens,
    knownTokens,
    unknownTokens,
    ranking,
  };
}

export function generateExercise(
  preset?: { count: number; length: number; N: number; df: number }
): ExerciseData {
  const count = preset ? preset.count : 1 + Math.floor(Math.random() * 8);
  const length = preset ? preset.length : 40 + Math.floor(Math.random() * 160);
  const N = preset ? preset.N : 1000 + Math.floor(Math.random() * 9000);
  const df = preset ? preset.df : 1 + Math.floor(Math.random() * Math.max(1, N * 0.25));

  const safeCount = Math.max(0, count);
  const safeLength = Math.max(1, length);
  const safeN = Math.max(1, N);
  const safeDf = Math.max(1, Math.min(safeN, df));

  const tf = safeCount / safeLength;
  const idf = Math.log(safeN / safeDf);
  const weight = tf * idf;

  return {
    count: safeCount,
    length: safeLength,
    N: safeN,
    df: safeDf,
    tf,
    idf,
    weight,
  };
}
