export interface EditCell {
  value: number;
  from: Array<"diagonal" | "up" | "left">;
}

export function editMatrix(left: string, right: string): EditCell[][] {
  const a = Array.from(left);
  const b = Array.from(right);
  const matrix = Array.from({ length: b.length + 1 }, (_, row) =>
    Array.from({ length: a.length + 1 }, (_, column): EditCell => ({
      value: row === 0 ? column : column === 0 ? row : 0,
      from: row === 0 && column > 0 ? ["left"] : column === 0 && row > 0 ? ["up"] : [],
    })),
  );

  for (let row = 1; row <= b.length; row += 1) {
    for (let column = 1; column <= a.length; column += 1) {
      const candidates = {
        diagonal: matrix[row - 1][column - 1].value + (a[column - 1] === b[row - 1] ? 0 : 1),
        up: matrix[row - 1][column].value + 1,
        left: matrix[row][column - 1].value + 1,
      };
      const value = Math.min(candidates.diagonal, candidates.up, candidates.left);
      matrix[row][column] = {
        value,
        from: (Object.entries(candidates) as Array<[EditCell["from"][number], number]>)
          .filter(([, candidate]) => candidate === value)
          .map(([direction]) => direction),
      };
    }
  }

  return matrix;
}

export type EditCoordinate = [row: number, column: number];

export function optimalEditPaths(matrix: EditCell[][], limit = 64): EditCoordinate[][] {
  const paths: EditCoordinate[][] = [];
  const end: EditCoordinate = [matrix.length - 1, matrix[0].length - 1];

  function visit(row: number, column: number, reversePath: EditCoordinate[]): void {
    if (paths.length >= limit) return;
    const path = [...reversePath, [row, column] as EditCoordinate];
    if (row === 0 && column === 0) {
      paths.push(path.reverse());
      return;
    }
    for (const direction of matrix[row][column].from) {
      if (direction === "diagonal") visit(row - 1, column - 1, path);
      if (direction === "up") visit(row - 1, column, path);
      if (direction === "left") visit(row, column - 1, path);
    }
  }

  visit(end[0], end[1], []);
  return paths;
}

function corpusSentences(corpus: string): string[][] {
  return corpus
    .split(/\r?\n/)
    .map((line) => line.trim().toLocaleLowerCase("en"))
    .filter(Boolean)
    .map((line) => ["<s>", ...line.match(/[\p{L}\p{N}]+/gu) ?? [], "</s>"]);
}

export function bigramVocabulary(corpus: string, sentence: string): string[] {
  const vocabulary = new Set<string>(["<s>"]);
  corpusSentences(corpus).forEach((tokens) => tokens.slice(1, -1).forEach((token) => vocabulary.add(token)));
  const words = sentence.trim().toLocaleLowerCase("en").match(/[\p{L}\p{N}]+/gu) ?? [];
  words.forEach((word) => vocabulary.add(word));
  vocabulary.add("</s>");
  return [...vocabulary];
}

export function bigramStatistics(corpus: string): {
  bigrams: Map<string, number>;
  contexts: Map<string, number>;
} {
  const bigrams = new Map<string, number>();
  const contexts = new Map<string, number>();
  for (const tokens of corpusSentences(corpus)) {
    for (let index = 1; index < tokens.length; index += 1) {
      const previous = tokens[index - 1];
      const key = `${previous}\u0000${tokens[index]}`;
      bigrams.set(key, (bigrams.get(key) ?? 0) + 1);
      contexts.set(previous, (contexts.get(previous) ?? 0) + 1);
    }
  }
  return { bigrams, contexts };
}

export function bigramCount(corpus: string, previous: string, next: string): number {
  return bigramStatistics(corpus).bigrams.get(`${previous}\u0000${next}`) ?? 0;
}

export function contextCount(corpus: string, context: string): number {
  return bigramStatistics(corpus).contexts.get(context) ?? 0;
}

export interface BigramFactor {
  previous: string;
  next: string;
  bigramCount: number;
  previousCount: number;
  probability: number;
}

export function sentenceBigramProbability(corpus: string, sentence: string): {
  factors: BigramFactor[];
  probability: number;
} {
  const words = sentence.trim().toLocaleLowerCase("en").match(/[\p{L}\p{N}]+/gu) ?? [];
  const tokens = ["<s>", ...words, "</s>"];
  const factors: BigramFactor[] = [];
  const statistics = bigramStatistics(corpus);

  for (let index = 1; index < tokens.length; index += 1) {
    const previous = tokens[index - 1];
    const next = tokens[index];
    const numerator = statistics.bigrams.get(`${previous}\u0000${next}`) ?? 0;
    const denominator = statistics.contexts.get(previous) ?? 0;
    factors.push({
      previous,
      next,
      bigramCount: numerator,
      previousCount: denominator,
      probability: denominator === 0 ? 0 : numerator / denominator,
    });
  }

  return {
    factors,
    probability: factors.reduce((product, factor) => product * factor.probability, 1),
  };
}

export interface CkyRule {
  left: string;
  right: [string] | [string, string];
}

export interface CkyDerivation {
  parent: string;
  terminal?: string;
  split?: number;
  left?: string;
  right?: string;
}

export interface CkyCell {
  symbols: string[];
  derivations: CkyDerivation[];
}

export function parseCkyGrammar(source: string): { rules: CkyRule[]; errors: string[] } {
  const rules: CkyRule[] = [];
  const errors: string[] = [];
  source.split(/\r?\n/).forEach((rawLine, index) => {
    const line = rawLine.trim();
    if (!line) return;
    const arrow = line.indexOf("->");
    if (arrow < 1) {
      errors.push(`Line ${index + 1}: expected A -> B C or A -> word.`);
      return;
    }
    const left = line.slice(0, arrow).trim();
    const alternatives = line.slice(arrow + 2).split("|");
    for (const alternative of alternatives) {
      const parts = alternative.trim().split(/\s+/).filter(Boolean);
      if (parts.length < 1 || parts.length > 2) {
        errors.push(`Line ${index + 1}: each right-hand side must contain one or two symbols.`);
        continue;
      }
      const right = parts.map((part) => part.replace(/^(?:'([^']*)'|"([^"]*)")$/, "$1$2")) as [string] | [string, string];
      rules.push({ left, right });
    }
  });
  return { rules, errors };
}

export function ckyChart(words: string[], rules: CkyRule[]): CkyCell[][] {
  const size = words.length;
  const chart = Array.from({ length: size }, () =>
    Array.from({ length: size + 1 }, (): CkyCell => ({ symbols: [], derivations: [] })),
  );
  const lexical = rules.filter((rule): rule is CkyRule & { right: [string] } => rule.right.length === 1);
  const binary = rules.filter((rule): rule is CkyRule & { right: [string, string] } => rule.right.length === 2);

  for (let start = 0; start < size; start += 1) {
    const cell = chart[start][start + 1];
    for (const rule of lexical) {
      if (rule.right[0] !== words[start]) continue;
      cell.derivations.push({ parent: rule.left, terminal: words[start] });
      if (!cell.symbols.includes(rule.left)) cell.symbols.push(rule.left);
    }
  }

  for (let span = 2; span <= size; span += 1) {
    for (let start = 0; start <= size - span; start += 1) {
      const end = start + span;
      const cell = chart[start][end];
      for (let split = start + 1; split < end; split += 1) {
        for (const rule of binary) {
          if (!chart[start][split].symbols.includes(rule.right[0]) || !chart[split][end].symbols.includes(rule.right[1])) continue;
          cell.derivations.push({ parent: rule.left, split, left: rule.right[0], right: rule.right[1] });
          if (!cell.symbols.includes(rule.left)) cell.symbols.push(rule.left);
        }
      }
    }
  }

  return chart;
}

export interface ViterbiResult {
  scores: number[][];
  backpointers: number[][];
  path: number[];
}

export function viterbi(
  emissions: number[][],
  transitions: number[][],
  initial: number[],
): ViterbiResult {
  const steps = emissions.length;
  const states = initial.length;
  const scores = Array.from({ length: steps }, () => Array(states).fill(0) as number[]);
  const backpointers = Array.from({ length: steps }, () => Array(states).fill(-1) as number[]);

  for (let state = 0; state < states; state += 1) {
    scores[0][state] = initial[state] * emissions[0][state];
  }

  for (let step = 1; step < steps; step += 1) {
    for (let state = 0; state < states; state += 1) {
      let bestState = 0;
      let bestScore = -1;
      for (let previous = 0; previous < states; previous += 1) {
        const candidate = scores[step - 1][previous] * transitions[previous][state];
        if (candidate > bestScore) {
          bestScore = candidate;
          bestState = previous;
        }
      }
      scores[step][state] = bestScore * emissions[step][state];
      backpointers[step][state] = bestState;
    }
  }

  let state = scores.at(-1)?.reduce((best, score, index, row) => score > row[best] ? index : best, 0) ?? 0;
  const path = Array(steps).fill(0) as number[];
  path[steps - 1] = state;
  for (let step = steps - 1; step > 0; step -= 1) {
    state = backpointers[step][state];
    path[step - 1] = state;
  }

  return { scores, backpointers, path };
}
