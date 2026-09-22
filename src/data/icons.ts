/**
 * Algorithmic SVG icons for all 10 NLP Visual Lab laboratories.
 * Built with standard 24x24 geometry, responsive strokes and theme-aware colors.
 */

function svgWrap(content: string, className = "lab-icon"): string {
  const cls = className ? ` class="${className}"` : "";
  return `<svg${cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${content}</svg>`;
}

const icons: Record<string, string> = {
  // 01.1 Byte Pair Encoding: Two subword boxes merging downwards into a unified subword token
  bpe: svgWrap(`
    <rect x="3" y="4" width="7" height="6" rx="1.5" />
    <rect x="14" y="4" width="7" height="6" rx="1.5" />
    <path d="M6.5 10v2a2 2 0 0 0 2 2h7a2 2 0 0 0 2-2v-2" />
    <path d="M12 14v2.5" />
    <path d="M10 15l2 2 2-2" />
    <rect x="4" y="17.5" width="16" height="4.5" rx="1.5" />
  `),

  // 01.2 Minimum Edit Distance: Dynamic programming cost matrix with optimal diagonal traversal
  "edit-distance": svgWrap(`
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <path d="M3 9h18M3 15h18" />
    <path d="M9 3v18M15 3v18" />
    <path d="M5.5 6.5l4 4 6 6" stroke-width="2.4" />
    <path d="M13.5 16.5h2v-2" stroke-width="2.4" />
  `),

  // 01.3 TF-IDF Vector Space: Coordinate axes with document vectors and cosine similarity angle
  tfidf: svgWrap(`
    <path d="M3 4v16a1 1 0 0 0 1 1h16" />
    <path d="M4 20L13 7" stroke-width="2.2" />
    <path d="M10 7h3v3" stroke-width="2.2" />
    <path d="M4 20l14-7" stroke-width="2.2" />
    <path d="M15 13h3v3" stroke-width="2.2" />
    <path d="M9 17.5a6.5 6.5 0 0 1 4.5-3" stroke-dasharray="2 2" />
  `),

  // 01.4 Word Embeddings: Semantic vector parallelogram and linear analogy offset
  "word-embeddings": svgWrap(`
    <circle cx="5.5" cy="7.5" r="1.8" />
    <circle cx="6" cy="17" r="1.8" />
    <circle cx="18.5" cy="7.5" r="1.8" />
    <circle cx="19" cy="17" r="1.8" />
    <path d="M5.5 7.5L6 17" stroke-opacity="0.35" />
    <path d="M18.5 7.5L19 17" stroke-opacity="0.35" />
    <path d="M5.5 7.5h13" stroke-dasharray="2 2" stroke-width="2" />
    <path d="M15.5 5.5l3 2-3 2" stroke-width="2" />
    <path d="M6 17h13" stroke-dasharray="2 2" stroke-width="2" />
    <path d="M16 15l3 2-3 2" stroke-width="2" />
  `),

  // 02.1 Bigram Language Models: Markov chain token sequence with conditional transition arcs
  "n-grams": svgWrap(`
    <circle cx="5" cy="13" r="2.5" />
    <circle cx="12" cy="13" r="2.5" />
    <circle cx="19" cy="13" r="2.5" />
    <path d="M7.5 11.5c1.2-1.8 2.3-1.8 3.5 0" />
    <path d="M10.2 10.5l1.3 1-.3-1.5" />
    <path d="M14.5 11.5c1.2-1.8 2.3-1.8 3.5 0" />
    <path d="M17.2 10.5l1.3 1-.3-1.5" />
    <path d="M5.5 8.5C8 4 16 4 18.5 8.5" stroke-dasharray="2 2" />
  `),

  // 02.2 Viterbi HMM: State trellis with the maximum-likelihood decoded sequence path
  viterbi: svgWrap(`
    <circle cx="4.5" cy="7" r="1.8" />
    <circle cx="4.5" cy="17" r="1.8" />
    <circle cx="12" cy="7" r="1.8" />
    <circle cx="12" cy="17" r="1.8" />
    <circle cx="19.5" cy="7" r="1.8" />
    <circle cx="19.5" cy="17" r="1.8" />
    <path d="M6 7l4.5 9.5M6 17l4.5-9.5M13.5 7l4.5 9.5" stroke-opacity="0.35" />
    <path d="M4.5 17L12 7l7.5 10" stroke-width="2.3" />
  `),

  // 02.3 CKY Parsing: Triangular chart with binary syntactic tree derivation
  cky: svgWrap(`
    <path d="M12 3L2 20h20L12 3z" stroke-opacity="0.4" />
    <circle cx="12" cy="7" r="1.8" />
    <circle cx="8" cy="14" r="1.8" />
    <circle cx="16" cy="14" r="1.8" />
    <path d="M10.8 8.5L9 12.2" stroke-width="2" />
    <path d="M13.2 8.5L15 12.2" stroke-width="2" />
    <path d="M7 15.5L5 19.5" stroke-width="2" />
    <path d="M9 15.5L11 19.5" stroke-width="2" />
    <path d="M15 15.5L13 19.5" stroke-width="2" />
    <path d="M17 15.5L19 19.5" stroke-width="2" />
  `),

  // 03.1 LSTM: Recurrent cell with top cell state memory and gate loops
  lstm: svgWrap(`
    <rect x="3" y="5" width="18" height="14" rx="2.5" />
    <path d="M1 9h22" stroke-width="2.2" />
    <circle cx="8" cy="9" r="1.5" fill="currentColor" />
    <circle cx="16" cy="9" r="1.5" fill="currentColor" />
    <path d="M8 19v-4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v4" />
    <path d="M8 12.5V10.5M16 12.5V10.5" />
    <path d="M12 19v-2.5" />
  `),

  // 03.2 Seq2Seq: Coupled Encoder and Decoder blocks linked by latent context vector
  seq2seq: svgWrap(`
    <rect x="2" y="6" width="7.5" height="12" rx="2" />
    <rect x="14.5" y="6" width="7.5" height="12" rx="2" />
    <path d="M9.5 12h5" stroke-width="2.2" />
    <path d="M12.5 10l2 2-2 2" stroke-width="2.2" />
    <path d="M5.5 3v3M5.5 18v3" />
    <path d="M18.5 3v3M18.5 18v3" />
  `),

  // 03.3 Attention Mechanism: Alignment matrix with dynamic radiating weight distribution
  attention: svgWrap(`
    <circle cx="12" cy="4" r="2" />
    <circle cx="4" cy="19" r="1.8" />
    <circle cx="9.3" cy="19" r="1.8" />
    <circle cx="14.7" cy="19" r="1.8" />
    <circle cx="20" cy="19" r="1.8" />
    <path d="M12 6L4 17.5" stroke-opacity="0.3" />
    <path d="M12 6L9.3 17.5" stroke-width="2.4" />
    <path d="M12 6L14.7 17.5" stroke-opacity="0.6" stroke-width="1.8" />
    <path d="M12 6L20 17.5" stroke-opacity="0.3" />
  `),

  // 03.4 Sentence Embeddings: Multi-token sequence converging through a pooling lens into a dense vector
  "sentence-embeddings": svgWrap(`
    <rect x="3" y="4" width="10" height="3" rx="1" />
    <rect x="3" y="9" width="10" height="3" rx="1" />
    <rect x="3" y="14" width="10" height="3" rx="1" />
    <path d="M13 5.5l4 4.5M13 10.5h4M13 15.5l4-4.5" stroke-opacity="0.4" />
    <path d="M17 7.5v6.5" stroke-width="2" />
    <path d="M17 10.5h4" stroke-width="2.2" />
    <path d="M19 8.5l2 2-2 2" stroke-width="2.2" />
    <circle cx="17" cy="10.5" r="1.5" fill="currentColor" />
  `),

  // 03.5 Transformers: Stacked multi-head decoder blocks with causal projection & residual skip
  transformers: svgWrap(`
    <rect x="4" y="3.5" width="16" height="4" rx="1.2" />
    <rect x="4" y="10" width="16" height="4" rx="1.2" />
    <rect x="4" y="16.5" width="16" height="4" rx="1.2" />
    <path d="M12 16.5V14M12 10V7.5" stroke-width="2" />
    <path d="M10.8 9l1.2-1.5 1.2 1.5" stroke-width="1.8" />
    <path d="M10.8 15.5l1.2-1.5 1.2 1.5" stroke-width="1.8" />
    <path d="M20 18.5c1.8-2.5 1.8-8.5 0-11" stroke-dasharray="2 2" />
    <path d="M18.8 8.5l1.4-1.2 1.4 1.2" />
  `),

  // Fallback icon for any unspecified slug
  default: svgWrap(`
    <circle cx="12" cy="12" r="8" />
    <path d="M12 8v4l3 3" />
  `),
};

export function getTopicIcon(slug: string, className = "lab-icon"): string {
  const iconMarkup = icons[slug] ?? icons.default;
  if (!className || className === "lab-icon") {
    return iconMarkup;
  }
  return iconMarkup.replace('class="lab-icon"', `class="${className}"`);
}
