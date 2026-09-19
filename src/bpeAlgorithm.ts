export interface BpeOptions {
  splitMode: "whitespace" | "lines" | "comma" | "custom" | "whole";
  customSeparator?: string;
  tieBreak: "first" | "lexical";
  normalizeNfc: boolean;
  lowercase: boolean;
  removePunctuation: boolean;
  targetVocabSize: number;
  maxMerges: number;
  minPairFrequency: number;
}

export interface BpePair {
  left: string;
  right: string;
  count: number;
  first: number;
}

export interface BpeRule {
  left: string;
  right: string;
  merged: string;
  count: number;
}

export interface BpeIterationState {
  seqs: string[][];
  vocab: string[];
  pairs: BpePair[];
  candidate: BpePair | null;
  rules: BpeRule[];
  reason: string;
}

export interface BpeModel {
  source: string[];
  initialVocab: string[];
  rules: BpeRule[];
  states: BpeIterationState[];
  stopReason: string;
  charCount: number;
}

export function normalizeText(
  text: string,
  options: Pick<BpeOptions, "normalizeNfc" | "lowercase" | "removePunctuation">
): string {
  let output = String(text);
  if (options.normalizeNfc && output.normalize) {
    output = output.normalize("NFC");
  }
  if (options.lowercase) {
    output = output.toLocaleLowerCase();
  }
  if (options.removePunctuation) {
    try {
      output = output.replace(/[\p{P}\p{S}]+/gu, "");
    } catch {
      output = output.replace(/[^\w\s]+/g, "");
    }
  }
  return output;
}

export function parseCorpus(
  text: string,
  options: Pick<BpeOptions, "splitMode" | "customSeparator" | "normalizeNfc" | "lowercase" | "removePunctuation">,
  isInfer: boolean = false
): string[] {
  const raw = normalizeText(text, options);
  if (!raw.trim()) return [];

  let parts: string[];
  const mode = options.splitMode;

  if (mode === "whole") {
    parts = [raw.replace(/\s+/g, " ").trim()];
  } else if (mode === "lines") {
    parts = raw.split(/\r?\n+/u);
  } else if (mode === "comma") {
    parts = raw.split(/\s*,\s*/u);
  } else if (mode === "custom") {
    const pattern = options.customSeparator || "\\s+";
    let regex: RegExp;
    try {
      regex = new RegExp(pattern, "u");
    } catch {
      throw new Error("Invalid custom separator regular expression.");
    }
    parts = raw.split(regex);
  } else {
    parts = raw.trim().split(/\s+/u);
  }

  const cleaned = parts.map((segment) => segment.trim()).filter(Boolean);
  if (!isInfer && cleaned.length > 5000) {
    throw new Error("Use at most 5,000 training sequences for interactive inspection.");
  }
  return cleaned;
}

export function buildInitialVocab(sequences: string[]): string[] {
  const seen = new Set<string>();
  const vocab: string[] = [];
  for (const seq of sequences) {
    for (const char of Array.from(seq)) {
      if (!seen.has(char)) {
        seen.add(char);
        vocab.push(char);
      }
    }
  }
  return vocab;
}

export function countAdjacentPairs(seqs: string[][], tieBreak: "first" | "lexical"): BpePair[] {
  const map = new Map<string, BpePair>();
  let order = 0;

  for (const seq of seqs) {
    for (let i = 0; i < seq.length - 1; i++) {
      const left = seq[i];
      const right = seq[i + 1];
      const key = `${JSON.stringify(left)}|${JSON.stringify(right)}`;
      const existing = map.get(key);
      if (!existing) {
        map.set(key, { left, right, count: 1, first: order++ });
      } else {
        existing.count++;
      }
    }
  }

  const list = Array.from(map.values());
  list.sort((a, b) => {
    if (b.count !== a.count) {
      return b.count - a.count;
    }
    if (tieBreak === "lexical") {
      const keyA = `${a.left}\0${a.right}`;
      const keyB = `${b.left}\0${b.right}`;
      return keyA.localeCompare(keyB);
    }
    return a.first - b.first;
  });

  return list;
}

export function applyMerge(seqs: string[][], rule: { left: string; right: string }): string[][] {
  return seqs.map((seq) => {
    const next: string[] = [];
    for (let i = 0; i < seq.length; i++) {
      if (i < seq.length - 1 && seq[i] === rule.left && seq[i + 1] === rule.right) {
        next.push(rule.left + rule.right);
        i++;
      } else {
        next.push(seq[i]);
      }
    }
    return next;
  });
}

export function trainBpe(corpusText: string, options: BpeOptions): BpeModel {
  const source = parseCorpus(corpusText, options);
  if (source.length === 0) {
    throw new Error("Add a non-empty corpus before training.");
  }
  if (corpusText.length > 60000) {
    throw new Error("Keep the corpus under 60,000 characters for this interactive demo.");
  }

  const target = Math.max(1, Math.min(4096, options.targetVocabSize || 100));
  const max = Math.max(0, Math.min(120, options.maxMerges ?? 25));
  const min = Math.max(1, options.minPairFrequency || 1);

  const initialVocab = buildInitialVocab(source);
  let seqs = source.map((s) => Array.from(s));
  const vocab = [...initialVocab];
  const rules: BpeRule[] = [];
  const states: BpeIterationState[] = [];
  let stopReason = "";
  let mergeCount = 0;

  while (true) {
    const pairList = countAdjacentPairs(seqs, options.tieBreak);
    let candidate: BpePair | null = pairList[0] || null;
    let reason = "";

    if (vocab.length >= target) {
      reason = "target vocabulary reached";
    } else if (mergeCount >= max) {
      reason = "maximum merges reached";
    } else if (!candidate) {
      reason = "no adjacent pairs remain";
    } else if (candidate.count < min) {
      reason = "pair frequency below minimum";
    }

    if (reason) {
      candidate = null;
    }

    states.push({
      seqs: seqs.map((seq) => [...seq]),
      vocab: [...vocab],
      pairs: pairList.slice(0, 20).map((pair) => ({ ...pair })),
      candidate: candidate ? { ...candidate } : null,
      rules: rules.map((r) => ({ ...r })),
      reason,
    });

    if (!candidate) {
      stopReason = reason;
      break;
    }

    const merged = candidate.left + candidate.right;
    seqs = applyMerge(seqs, candidate);
    if (!vocab.includes(merged)) {
      vocab.push(merged);
    }
    rules.push({
      left: candidate.left,
      right: candidate.right,
      merged,
      count: candidate.count,
    });
    mergeCount++;
  }

  const charCount = source.reduce((total, seq) => total + Array.from(seq).length, 0);

  return {
    source,
    initialVocab,
    rules,
    states,
    stopReason,
    charCount,
  };
}

export function tokenizeWithRules(
  tokens: string[],
  rules: BpeRule[]
): { tokens: string[]; trace: string[][] } {
  let current = [...tokens];
  const trace: string[][] = [[...current]];

  for (const rule of rules) {
    current = applyMerge([current], rule)[0];
    trace.push([...current]);
  }

  return { tokens: current, trace };
}
