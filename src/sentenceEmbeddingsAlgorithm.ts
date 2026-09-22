export interface SentenceSimilarityRanking {
  index: number;
  score: number;
}

export interface Sentence2DPoint {
  x: number;
  y: number;
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

export function l2Normalize(vector: Float32Array): Float32Array {
  const norm = vectorNorm(vector);
  const out = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i++) {
    out[i] = vector[i] / norm;
  }
  return out;
}

export function cosineSimilaritySentence(a: Float32Array, b: Float32Array): number {
  const normA = vectorNorm(a);
  const normB = vectorNorm(b);
  return dotProduct(a, b) / (normA * normB);
}

/**
 * Mean pooling with attention mask:
 * average token embeddings across all valid tokens (mask == 1) for each sequence in batch.
 *
 * @param hiddenState [batchSize, seqLen, hiddenDim]
 * @param attentionMask [batchSize, seqLen]
 */
export function meanPooling(
  hiddenState: number[][][] | Float32Array[][],
  attentionMask: number[][]
): Float32Array[] {
  const batchSize = hiddenState.length;
  const pooledBatch: Float32Array[] = [];

  for (let b = 0; b < batchSize; b++) {
    const seq = hiddenState[b];
    const mask = attentionMask[b];
    const seqLen = seq.length;
    if (seqLen === 0) {
      pooledBatch.push(new Float32Array(0));
      continue;
    }

    const hiddenDim = seq[0].length;
    const sum = new Float32Array(hiddenDim);
    let validCount = 0;

    for (let i = 0; i < seqLen; i++) {
      const isUnmasked = mask ? mask[i] > 0 : true;
      if (isUnmasked) {
        validCount++;
        const tokenVec = seq[i];
        for (let d = 0; d < hiddenDim; d++) {
          sum[d] += tokenVec[d];
        }
      }
    }

    const divisor = validCount > 0 ? validCount : 1;
    for (let d = 0; d < hiddenDim; d++) {
      sum[d] /= divisor;
    }

    pooledBatch.push(sum);
  }

  return pooledBatch;
}

export function buildSentenceMatrix(embeddings: Float32Array[]): number[][] {
  const n = embeddings.length;
  const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    matrix[i][i] = 1.0;
    for (let j = i + 1; j < n; j++) {
      const sim = cosineSimilaritySentence(embeddings[i], embeddings[j]);
      matrix[i][j] = sim;
      matrix[j][i] = sim;
    }
  }

  return matrix;
}

export function rankBySimilarity(
  queryEmb: Float32Array,
  docEmbs: Float32Array[]
): SentenceSimilarityRanking[] {
  const ranking: SentenceSimilarityRanking[] = docEmbs.map((emb, idx) => ({
    index: idx,
    score: cosineSimilaritySentence(queryEmb, emb),
  }));

  ranking.sort((a, b) => b.score - a.score);
  return ranking;
}

export function projectSentencePCA(embeddings: Float32Array[]): Sentence2DPoint[] {
  const n = embeddings.length;
  if (n === 0) return [];
  if (n === 1) return [{ x: 0, y: 0 }];

  const dim = embeddings[0].length;

  // 1. Mean center
  const mean = new Float32Array(dim);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < dim; d++) {
      mean[d] += embeddings[i][d];
    }
  }
  for (let d = 0; d < dim; d++) {
    mean[d] /= n;
  }

  const centered: Float32Array[] = [];
  for (let i = 0; i < n; i++) {
    const row = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      row[d] = embeddings[i][d] - mean[d];
    }
    centered.push(row);
  }

  // 2. Power iteration for PC1
  let pc1 = new Float32Array(dim);
  for (let d = 0; d < dim; d++) pc1[d] = Math.sin(d + 1);
  pc1 = new Float32Array(l2Normalize(pc1));

  for (let iter = 0; iter < 25; iter++) {
    const next = new Float32Array(dim);
    for (let i = 0; i < n; i++) {
      const proj = dotProduct(centered[i], pc1);
      for (let d = 0; d < dim; d++) {
        next[d] += proj * centered[i][d];
      }
    }
    pc1 = new Float32Array(l2Normalize(next));
  }

  // 3. Power iteration for PC2 (orthogonal to PC1)
  let pc2 = new Float32Array(dim);
  for (let d = 0; d < dim; d++) pc2[d] = Math.cos(d + 1);
  const dot12 = dotProduct(pc2, pc1);
  for (let d = 0; d < dim; d++) pc2[d] -= dot12 * pc1[d];
  pc2 = new Float32Array(l2Normalize(pc2));

  for (let iter = 0; iter < 25; iter++) {
    const next = new Float32Array(dim);
    for (let i = 0; i < n; i++) {
      const proj = dotProduct(centered[i], pc2);
      for (let d = 0; d < dim; d++) {
        next[d] += proj * centered[i][d];
      }
    }
    const p1 = dotProduct(next, pc1);
    for (let d = 0; d < dim; d++) {
      next[d] -= p1 * pc1[d];
    }
    pc2 = new Float32Array(l2Normalize(next));
  }

  // 4. Project coordinates
  const coords: Sentence2DPoint[] = [];
  let maxX = 0.001;
  let maxY = 0.001;

  for (let i = 0; i < n; i++) {
    const x = dotProduct(centered[i], pc1);
    const y = dotProduct(centered[i], pc2);
    if (Math.abs(x) > maxX) maxX = Math.abs(x);
    if (Math.abs(y) > maxY) maxY = Math.abs(y);
    coords.push({ x, y });
  }

  return coords.map((c) => ({
    x: c.x / maxX,
    y: c.y / maxY,
  }));
}
