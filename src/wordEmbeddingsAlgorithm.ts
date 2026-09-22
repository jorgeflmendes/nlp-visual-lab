export interface WordEmbeddingIndex {
  vocab: string[];
  wordToId: Map<string, number>;
  vectors: Float32Array;
  dim: number;
}

export interface NearestNeighbor {
  word: string;
  similarity: number;
}

export interface ExpressionTerm {
  sign: 1 | -1;
  weight: number;
  word: string;
}

export interface ExpressionEvalResult {
  terms: ExpressionTerm[];
  resultVector: Float32Array;
  candidates: NearestNeighbor[];
  usedWords: string[];
  missingWords: string[];
  rawExpression: string;
}

export interface AnalogyResult {
  resultVector: Float32Array;
  candidates: NearestNeighbor[];
  usedWords?: string[];
}

export interface PCA2DPoint {
  word: string;
  x: number;
  y: number;
  isResult?: boolean;
}

export interface ExtraPCAPoint {
  label: string;
  vector: Float32Array;
  isResult?: boolean;
}

export function createEmbeddingIndex(
  vocab: string[],
  buffer: ArrayBuffer,
  dim = 100,
): WordEmbeddingIndex {
  const wordToId = new Map<string, number>();
  for (let i = 0; i < vocab.length; i++) {
    wordToId.set(vocab[i], i);
  }
  const vectors = new Float32Array(buffer);
  return {
    vocab,
    wordToId,
    vectors,
    dim,
  };
}

export function getWordVector(
  index: WordEmbeddingIndex,
  word: string,
): Float32Array | null {
  const normalized = word.toLowerCase().trim();
  const id = index.wordToId.get(normalized);
  if (id === undefined) return null;
  const start = id * index.dim;
  return index.vectors.subarray(start, start + index.dim);
}

export function dotProduct(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  const len = a.length;
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function vectorNorm(v: Float32Array): number {
  return Math.sqrt(dotProduct(v, v)) || 1e-12;
}

export function normalizeVector(v: Float32Array): Float32Array {
  const norm = vectorNorm(v);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) {
    out[i] = v[i] / norm;
  }
  return out;
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  const normA = vectorNorm(a);
  const normB = vectorNorm(b);
  return dotProduct(a, b) / (normA * normB);
}

export function findNearestNeighbors(
  index: WordEmbeddingIndex,
  vector: Float32Array,
  topK = 10,
  excludeWords = new Set<string>(),
): NearestNeighbor[] {
  const target = normalizeVector(vector);
  const dim = index.dim;
  const total = index.vocab.length;
  const vecs = index.vectors;

  // Track top K with an array sorted ascending by similarity
  const top: NearestNeighbor[] = [];

  for (let i = 0; i < total; i++) {
    const word = index.vocab[i];
    if (excludeWords.has(word)) continue;

    const offset = i * dim;
    let dot = 0;
    for (let d = 0; d < dim; d++) {
      dot += target[d] * vecs[offset + d];
    }

    if (top.length < topK) {
      top.push({ word, similarity: dot });
      top.sort((a, b) => a.similarity - b.similarity);
    } else if (dot > top[0].similarity) {
      top[0] = { word, similarity: dot };
      top.sort((a, b) => a.similarity - b.similarity);
    }
  }

  // Return descending
  return top.reverse();
}

/**
 * Parse an arbitrary arithmetic expression into linear combination terms.
 * Supports:
 * - "king - man + woman"
 * - "2 * paris - france + 0.5 * tokyo"
 * - "cat + dog - animal"
 * - "-man + woman + king"
 */
export function parseExpression(expr: string): ExpressionTerm[] {
  const clean = expr
    .replace(/[−–—]/g, "-")
    .replace(/\s+/g, " ")
    .trim();
  if (!clean) return [];

  const tokenRegex = /([+-])|(\d+(?:\.\d+)?)\s*\*?|([a-zA-Z][a-zA-Z0-9_\-]*)/g;
  let match: RegExpExecArray | null;
  let currentSign: 1 | -1 = 1;
  let currentWeight = 1.0;
  const terms: ExpressionTerm[] = [];

  while ((match = tokenRegex.exec(clean)) !== null) {
    if (match[1]) {
      currentSign = match[1] === "-" ? -1 : 1;
    } else if (match[2]) {
      currentWeight = parseFloat(match[2]);
    } else if (match[3]) {
      const word = match[3].toLowerCase();
      terms.push({ sign: currentSign, weight: currentWeight, word });
      currentSign = 1;
      currentWeight = 1.0;
    }
  }

  return terms;
}

/**
 * Evaluate an arbitrary vector arithmetic expression over the embedding vocabulary.
 * Computes: target = sum(sign_i * weight_i * v_i), normalizes to unit length,
 * and retrieves the closest candidate words.
 */
export function evaluateExpression(
  index: WordEmbeddingIndex,
  expr: string,
  topK = 6,
  excludeQueryWords = true,
): ExpressionEvalResult | null {
  const terms = parseExpression(expr);
  if (terms.length === 0) return null;

  const dim = index.dim;
  const target = new Float32Array(dim);
  const usedWords: string[] = [];
  const missingWords: string[] = [];

  for (const term of terms) {
    const vec = getWordVector(index, term.word);
    if (!vec) {
      if (!missingWords.includes(term.word)) {
        missingWords.push(term.word);
      }
      continue;
    }
    if (!usedWords.includes(term.word)) {
      usedWords.push(term.word);
    }
    const factor = term.sign * term.weight;
    for (let d = 0; d < dim; d++) {
      target[d] += factor * vec[d];
    }
  }

  if (usedWords.length === 0) {
    return {
      terms,
      resultVector: new Float32Array(dim),
      candidates: [],
      usedWords,
      missingWords,
      rawExpression: expr,
    };
  }

  const resultVector = normalizeVector(target);
  const exclude = excludeQueryWords ? new Set(usedWords) : new Set<string>();
  const candidates = findNearestNeighbors(index, resultVector, topK, exclude);

  return {
    terms,
    resultVector,
    candidates,
    usedWords,
    missingWords,
    rawExpression: expr,
  };
}

export function solveAnalogy(
  index: WordEmbeddingIndex,
  wordA: string,
  wordB: string,
  wordC: string,
  topK = 5,
  excludeQueryWords = true,
): AnalogyResult | null {
  const expr = `${wordA} - ${wordB} + ${wordC}`;
  const res = evaluateExpression(index, expr, topK, excludeQueryWords);
  if (!res || res.candidates.length === 0) return null;
  return {
    resultVector: res.resultVector,
    candidates: res.candidates,
    usedWords: res.usedWords,
  };
}

export function computeSimilarityMatrix(
  index: WordEmbeddingIndex,
  words: string[],
): { words: string[]; matrix: number[][] } {
  const validWords: string[] = [];
  const vectors: Float32Array[] = [];

  for (const w of words) {
    const v = getWordVector(index, w);
    if (v) {
      validWords.push(w);
      vectors.push(v);
    }
  }

  const n = validWords.length;
  const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    matrix[i][i] = 1.0;
    for (let j = i + 1; j < n; j++) {
      const sim = dotProduct(vectors[i], vectors[j]);
      matrix[i][j] = sim;
      matrix[j][i] = sim;
    }
  }

  return { words: validWords, matrix };
}

// 2D projection using Principal Component Analysis (power iteration)
export function project2DPCA(
  index: WordEmbeddingIndex,
  words: string[],
  extraPoints: ExtraPCAPoint[] = [],
): PCA2DPoint[] {
  const labels: string[] = [];
  const vectors: Float32Array[] = [];
  const isResultFlags: boolean[] = [];

  const seen = new Set<string>();
  for (const w of words) {
    const normalized = w.toLowerCase().trim();
    if (!normalized || seen.has(normalized)) continue;
    const v = getWordVector(index, normalized);
    if (v) {
      seen.add(normalized);
      labels.push(normalized);
      vectors.push(v);
      isResultFlags.push(false);
    }
  }

  for (const extra of extraPoints) {
    labels.push(extra.label);
    vectors.push(extra.vector);
    isResultFlags.push(extra.isResult ?? false);
  }

  const n = labels.length;
  if (n === 0) return [];
  if (n === 1) return [{ word: labels[0], x: 0, y: 0, isResult: isResultFlags[0] }];

  const dim = index.dim;

  // 1. Center the data
  const mean = new Float32Array(dim);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < dim; d++) {
      mean[d] += vectors[i][d];
    }
  }
  for (let d = 0; d < dim; d++) {
    mean[d] /= n;
  }

  const centered: Float32Array[] = [];
  for (let i = 0; i < n; i++) {
    const row = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      row[d] = vectors[i][d] - mean[d];
    }
    centered.push(row);
  }

  // 2. Power iteration to find PC1
  let pc1 = new Float32Array(dim);
  for (let d = 0; d < dim; d++) pc1[d] = Math.sin(d + 1);
  pc1 = new Float32Array(normalizeVector(pc1));

  for (let iter = 0; iter < 20; iter++) {
    const next = new Float32Array(dim);
    for (let i = 0; i < n; i++) {
      const proj = dotProduct(centered[i], pc1);
      for (let d = 0; d < dim; d++) {
        next[d] += proj * centered[i][d];
      }
    }
    pc1 = new Float32Array(normalizeVector(next));
  }

  // 3. Power iteration for PC2 (orthogonalized against PC1)
  let pc2 = new Float32Array(dim);
  for (let d = 0; d < dim; d++) pc2[d] = Math.cos(d + 1);
  const dot12 = dotProduct(pc2, pc1);
  for (let d = 0; d < dim; d++) pc2[d] -= dot12 * pc1[d];
  pc2 = new Float32Array(normalizeVector(pc2));

  for (let iter = 0; iter < 20; iter++) {
    const next = new Float32Array(dim);
    for (let i = 0; i < n; i++) {
      const proj = dotProduct(centered[i], pc2);
      for (let d = 0; d < dim; d++) {
        next[d] += proj * centered[i][d];
      }
    }
    // Orthogonalize against pc1
    const p1 = dotProduct(next, pc1);
    for (let d = 0; d < dim; d++) {
      next[d] -= p1 * pc1[d];
    }
    pc2 = new Float32Array(normalizeVector(next));
  }

  // 4. Project coordinates
  const coords: PCA2DPoint[] = [];
  let maxX = 0.001;
  let maxY = 0.001;

  for (let i = 0; i < n; i++) {
    const x = dotProduct(centered[i], pc1);
    const y = dotProduct(centered[i], pc2);
    if (Math.abs(x) > maxX) maxX = Math.abs(x);
    if (Math.abs(y) > maxY) maxY = Math.abs(y);
    coords.push({ word: labels[i], x, y, isResult: isResultFlags[i] });
  }

  // Normalize coordinates to [-1, 1]
  return coords.map((c) => ({
    word: c.word,
    x: c.x / maxX,
    y: c.y / maxY,
    isResult: c.isResult,
  }));
}
