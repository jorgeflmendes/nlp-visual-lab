export interface Topic {
  slug: string;
  title: string;
  summary: string;
  tag?: string;
  stages?: string[];
  kind?: "bpe" | "lstm" | "seq2seq" | "attention" | "sentence-embeddings" | "transformer";
}

export interface DeepLearningTopic extends Topic {
  kind: "bpe" | "lstm" | "seq2seq" | "attention" | "sentence-embeddings" | "transformer";
}

export const foundationsTopics: (Topic | DeepLearningTopic)[] = [
  {
    slug: "bpe",
    kind: "bpe",
    title: "Byte Pair Encoding",
    summary: "Train subword tokenization on an arbitrary corpus and inspect every merge decision.",
    tag: "Subwords & Tokenization",
    stages: ["Corpus", "Pairs", "Merges", "Tokenizer"],
  },
  {
    slug: "edit-distance",
    title: "Minimum edit distance",
    summary: "Fill the dynamic programming matrix and trace its minimum cost paths.",
    tag: "Alignment & DP",
    stages: ["Matrix", "Minimum paths"],
  },
  {
    slug: "tfidf",
    title: "TF-IDF Vector Space",
    summary: "Build term-document representations from first principles, evaluate cosine similarities and query retrieval.",
    tag: "Vector Space & Cosine",
    stages: ["Counts", "TF", "DF / IDF", "Vectors"],
  },
  {
    slug: "word-embeddings",
    title: "Word Embeddings",
    summary: "Explore 100-dimensional GloVe representations over a 20,000-word vocabulary.",
    tag: "GloVe & Vector Algebra",
    stages: ["Vectors", "Cosine", "Analogy", "PCA"],
  },
];

export const structuredTopics: Topic[] = [
  {
    slug: "n-grams",
    title: "Bigram language models",
    summary: "Count bigrams and factor the probability of a complete sentence.",
    tag: "MLE & Probabilities",
    stages: ["Counts", "Sentence factors"],
  },
  {
    slug: "viterbi",
    title: "Viterbi HMM",
    summary: "Edit an HMM, build its sequence score and backpointer tables, and recover the best tag sequence.",
    tag: "Sequence Decoding",
    stages: ["SS & BP tables", "Best sequence"],
  },
  {
    slug: "cky",
    title: "CKY parsing",
    summary: "Fill a pyramidal chart from a grammar in Chomsky normal form.",
    tag: "CNF & Syntactic Trees",
    stages: ["Grammar", "Chart", "Derivations"],
  },
];

export const neuralTopics: DeepLearningTopic[] = [
  {
    slug: "lstm",
    kind: "lstm",
    title: "Long short-term memory",
    summary: "Classify arbitrary English reviews with a trained IMDB LSTM.",
    tag: "Gated Memory & Sentiment",
    stages: ["Words", "Embedding", "Gated memory", "Sentiment"],
  },
  {
    slug: "seq2seq",
    kind: "seq2seq",
    title: "Sequence-to-sequence models",
    summary: "Translate short English inputs using a trained LSTM encoder-decoder.",
    tag: "Encoder-Decoder",
    stages: ["Characters", "Encoder", "Decoder", "Translation"],
  },
  {
    slug: "attention",
    kind: "attention",
    title: "Attention mechanisms",
    summary: "Normalise a date and inspect the trained model's attention matrix.",
    tag: "Alignment Matrix",
    stages: ["Characters", "Encoder", "Attention", "ISO date"],
  },
  {
    slug: "sentence-embeddings",
    kind: "sentence-embeddings",
    title: "Sentence Embeddings",
    summary: "Extract 384-dimensional dense semantic vectors with all-MiniLM-L6-v2 and trace mean pooling.",
    tag: "MiniLM & Mean Pooling",
    stages: ["Tokens", "Hidden states", "Mean pooling", "Normalized vector"],
  },
  {
    slug: "transformers",
    kind: "transformer",
    title: "Transformers",
    summary: "Generate a continuation from real DistilGPT2 next-token logits.",
    tag: "DistilGPT2 Causal LM",
    stages: ["Tokens", "Decoder blocks", "Vocabulary scores", "Continuation"],
  },
];

export interface Category {
  id: "foundations" | "structured" | "neural";
  index: string;
  title: string;
  subtitle: string;
  eyebrow: string;
  description: string;
  topics: (Topic | DeepLearningTopic)[];
}

export const allCategories: Category[] = [
  {
    id: "foundations",
    index: "01",
    title: "Foundations",
    subtitle: "Foundations & Representations",
    eyebrow: "Foundations / text representations",
    description: "Subword tokenization, string alignment metrics, and vector space document representations.",
    topics: foundationsTopics,
  },
  {
    id: "structured",
    index: "02",
    title: "Structured",
    subtitle: "Structured & Classical NLP",
    eyebrow: "Structured NLP / exact inference",
    description: "Probabilistic n-gram models, sequence labeling with HMMs, and chart parsing with context-free grammars.",
    topics: structuredTopics,
  },
  {
    id: "neural",
    index: "03",
    title: "Neural",
    subtitle: "Neural Architectures",
    eyebrow: "Neural architectures / in-browser inference",
    description: "Recurrent networks, sequence-to-sequence translation, attention mechanisms, and causal transformers.",
    topics: neuralTopics,
  },
];

// Compatibility aliases
export const topics: Topic[] = [
  ...foundationsTopics.filter((t) => t.slug !== "bpe"),
  ...structuredTopics,
];

export const deepLearningTopics: DeepLearningTopic[] = [
  foundationsTopics[0] as DeepLearningTopic,
  ...neuralTopics,
];
