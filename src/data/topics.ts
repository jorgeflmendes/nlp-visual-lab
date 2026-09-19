export interface Topic {
  slug: string;
  title: string;
  summary: string;
}

export const topics: Topic[] = [
  {
    slug: "edit-distance",
    title: "Minimum edit distance",
    summary: "Fill the dynamic programming matrix and trace its minimum cost paths.",
  },
  {
    slug: "n-grams",
    title: "Bigram language models",
    summary: "Count bigrams and factor the probability of a complete sentence.",
  },
  {
    slug: "viterbi",
    title: "Viterbi HMM",
    summary: "Edit an HMM, build its sequence score and backpointer tables, and recover the best tag sequence.",
  },
  {
    slug: "cky",
    title: "CKY parsing",
    summary: "Fill a pyramidal chart from a grammar in Chomsky normal form.",
  },
  {
    slug: "tfidf",
    title: "TF-IDF Vector Space",
    summary: "Build term-document representations from first principles, evaluate cosine similarities and query retrieval.",
  },
];

export interface DeepLearningTopic extends Topic {
  kind: "bpe" | "lstm" | "seq2seq" | "attention" | "transformer";
}

export const deepLearningTopics: DeepLearningTopic[] = [
  { slug: "bpe", kind: "bpe", title: "Byte Pair Encoding", summary: "Train subword tokenization on an arbitrary corpus and inspect every merge decision." },
  { slug: "lstm", kind: "lstm", title: "Long short-term memory", summary: "Classify arbitrary English reviews with a trained IMDB LSTM." },
  { slug: "seq2seq", kind: "seq2seq", title: "Sequence-to-sequence models", summary: "Translate short English inputs using a trained LSTM encoder-decoder." },
  { slug: "attention", kind: "attention", title: "Attention mechanisms", summary: "Normalise a date and inspect the trained model's attention matrix." },
  { slug: "transformers", kind: "transformer", title: "Transformers", summary: "Generate a continuation from real DistilGPT2 next-token logits." },
];
