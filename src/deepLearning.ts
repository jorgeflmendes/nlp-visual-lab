import type { DeepLearningTopic } from "./data/topics.ts";
import { getTopicIcon } from "./data/icons.ts";
import { bindExecutionWorkspace } from "./executionWorkspace.ts";
import { bpePage, bindBpe } from "./bpeLab.ts";
import "./deepLearning.css";

type TypesetMath = (root: ParentNode) => Promise<void>;

interface PageCopy {
  label: string;
  task: string;
  description: string;
  inputLabel: string;
  inputValue: string;
  inputHint: string;
  presets: [string, string][];
  model: string;
  source: string;
  sourceUrl: string;
  inspect: string;
  stages: string[];
}

const copy: Record<DeepLearningTopic["kind"], PageCopy> = {
  bpe: {
    label: "BPE", task: "Train subwords. Inspect every merge.",
    description: "Follow how Byte Pair Encoding iteratively aggregates the most frequent adjacent character and subword pairs into a learned vocabulary.",
    inputLabel: "Training corpus", inputValue: "low lower lowest new newer newest wide wider widest",
    inputHint: "Define sequences, vocabulary limits and stopping criteria.",
    presets: [["Vocabulary expansion", "low lower lowest new newer newest wide wider widest"], ["Morphology & affixes", "play playing player plays replay walk walking walker walks rework teach teacher teaches reteach"]],
    model: "Byte Pair Encoding", source: "Subword Tokenization", sourceUrl: "https://en.wikipedia.org/wiki/Byte_pair_encoding",
    inspect: "Pair leaderboard, merge decisions, evolving vocabulary and inference trace.", stages: ["Corpus", "Pairs", "Merges", "Tokenizer"],
  },
  lstm: {
    label: "LSTM", task: "Read a review. Follow its memory.",
    description: "Trace how a trained recurrent network turns a sequence of words into a sentiment prediction.",
    inputLabel: "English review", inputValue: "This film starts slowly but becomes deeply moving.",
    inputHint: "Unseen words use the vocabulary’s unknown token.",
    presets: [["Mixed review", "This film starts slowly but becomes deeply moving."], ["Positive", "A wonderful film with excellent acting and a beautiful story."], ["Negative", "The movie was boring and the acting was terrible."]],
    model: "IMDB LSTM", source: "TensorFlow.js", sourceUrl: "https://github.com/tensorflow/tfjs-examples/tree/master/sentiment",
    inspect: "Word embeddings, recurrent gates, cell memory and the final sentiment score.", stages: ["Words", "Embedding", "Gated memory", "Sentiment"],
  },
  seq2seq: {
    label: "Seq2Seq", task: "Encode a phrase. Decode its translation.",
    description: "Follow a character-level LSTM encoder and decoder as they translate short English phrases into French.",
    inputLabel: "English source", inputValue: "Go.", inputHint: "Up to 16 characters. This compact model works best with short, simple phrases.",
    presets: [["Go.", "Go."], ["Hello!", "Hello!"], ["Thank you.", "Thank you."]],
    model: "English → French LSTM", source: "TensorFlow.js", sourceUrl: "https://github.com/tensorflow/tfjs-examples/tree/master/translation",
    inspect: "Character encodings, encoder memory, decoder states and each next-character decision.", stages: ["Characters", "Encoder", "Decoder", "Translation"],
  },
  attention: {
    label: "Attention", task: "Read a date. See where attention goes.",
    description: "Inspect how a trained network weights input characters when converting a date into ISO notation.",
    inputLabel: "Date", inputValue: "AUG 19, 2026", inputHint: "Up to 12 characters, using numbers or three-letter English month names.",
    presets: [["Month first", "AUG 19, 2026"], ["Day first", "19AUG2026"], ["Another date", "MAR 3, 2001"]],
    model: "Date conversion with attention", source: "TensorFlow.js", sourceUrl: "https://github.com/tensorflow/tfjs-examples/tree/master/date-conversion-attention",
    inspect: "Encoder states, attention weights, weighted context and output character probabilities.", stages: ["Characters", "Encoder", "Attention", "ISO date"],
  },
  transformer: {
    label: "Transformer", task: "Write a prompt. Inspect the next token.",
    description: "Explore DistilGPT2’s causal decoder blocks and the vocabulary scores behind its continuation.",
    inputLabel: "English prompt", inputValue: "Data visualization empowers users to", inputHint: "The first run downloads the quantized model. Short prompts make the computation easier to inspect.",
    presets: [["Visualization", "Data visualization empowers users to"], ["A story", "Once upon a time, in a small village"], ["A question", "The most interesting thing about science is"]],
    model: "DistilGPT2 · quantized", source: "Hugging Face", sourceUrl: "https://huggingface.co/Xenova/distilgpt2",
    inspect: "Token IDs, each block’s key/value cache, vocabulary logits and next-token probabilities.", stages: ["Tokens", "Decoder blocks", "Vocabulary scores", "Continuation"],
  },
};

function escapeHtml(value: string): string {
  return value.replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character] ?? character);
}

const routes: [DeepLearningTopic["kind"], string][] = [
  ["lstm", "lstm"],
  ["seq2seq", "seq2seq"],
  ["attention", "attention"],
  ["transformer", "transformers"],
];

export function deepLearningCategory(topics: DeepLearningTopic[]): string {
  return `<section class="dl-category"><header><p class="dl-eyebrow">Neural / architecture laboratory</p><h1>Follow the computation.</h1><p>Run a trained neural network in your browser. Move through its forward pass and inspect the values behind its prediction.</p></header><div class="dl-catalog" aria-label="Neural architecture experiments">${topics.map((topic, index) => {
    const page = copy[topic.kind];
    return `<a href="#/${topic.slug}"><span class="dl-catalog-index">0${index + 1}</span><div class="dl-catalog-title"><h2>${page.label}</h2><p>${page.task}</p></div><div class="dl-catalog-flow" aria-hidden="true">${page.stages.map(stage => `<span>${stage}</span>`).join('<b>→</b>')}</div><span class="dl-catalog-arrow" aria-hidden="true">↗</span></a>`;
  }).join("")}</div><p class="dl-catalog-note">Published weights. Local inference. Every displayed activation comes from your run.</p></section>`;
}

export function deepLearningPage(topic: DeepLearningTopic): string {
  if (topic.kind === "bpe") {
    return bpePage(topic);
  }
  const page = copy[topic.kind];
  return `<article class="dl-workspace" data-kind="${topic.kind}">
    <nav class="dl-model-nav" aria-label="Neural architecture experiments"><a class="dl-back" href="#/neural" aria-label="All neural architecture experiments">← <span>Neural Architectures</span></a><div>${routes.map(([kind, slug]) => `<a href="#/${slug}" ${kind === topic.kind ? 'aria-current="page"' : ""}>${copy[kind].label}</a>`).join("")}</div></nav>
    <header class="dl-page-heading"><div><p class="dl-eyebrow">${page.label} / ${page.model}</p><div class="dl-heading-title-row">${getTopicIcon(topic.slug, "dl-heading-icon")}<h1>${page.task}</h1></div><p>${page.description}</p></div><span id="model-state" class="dl-state">Not loaded</span></header>
    <section class="dl-experiment" aria-label="Model input and output">
      <div class="dl-input-panel"><div class="dl-section-label"><label for="neural-input">${page.inputLabel}</label><span>01 / Input</span></div><textarea id="neural-input" rows="2" aria-describedby="neural-input-hint" spellcheck="false">${escapeHtml(page.inputValue)}</textarea><div class="dl-presets" aria-label="Example inputs"><span>Try</span>${page.presets.map(([label, value]) => `<button type="button" data-preset="${escapeHtml(value)}">${escapeHtml(label)}</button>`).join("")}</div><p id="neural-input-hint" class="dl-input-hint">${page.inputHint}</p><div class="dl-run-controls"><button id="run-real-model" class="dl-primary" type="button">Load & run <span aria-hidden="true">→</span></button><button id="load-real-model" type="button">Load only</button><button id="reset-real-model" type="button">Reset</button></div></div>
      <div class="dl-output-panel"><div class="dl-section-label"><span>Model output</span><span>02 / Prediction</span></div><div id="neural-output" aria-live="polite"><p class="dl-output-empty">Your prediction appears here.</p><p class="dl-note">Run the model with an input to compute a result.</p></div><div class="dl-model-meta"><a href="${page.sourceUrl}" target="_blank" rel="noreferrer">${page.source} model source ↗</a><span>Published trained weights</span><span id="model-runtime">Inference runs in your browser.</span></div></div>
      <div id="model-status" role="status" aria-live="polite">Ready to load. The first run downloads the model weights.</div>
    </section>
    <div id="neural-result"><section class="dl-empty" aria-label="Execution inspector"><div><p class="dl-eyebrow">03 / Execution inspector</p><h2>From input to prediction.</h2><p>${page.inspect}</p></div><ol class="dl-empty-flow">${page.stages.map(stage => `<li>${stage}</li>`).join("")}</ol><p class="dl-note">Run the model to inspect this input’s actual activations.</p></section></div>
    <details class="dl-model-notes"><summary>About this model and its trace</summary><p>${topic.kind === "lstm" ? "The classifier reads padded word embeddings with an LSTM. Cell memory carries information across words; the hidden state exposes part of that memory through the output gate. The trace captures the actual input, forget and output gates." : topic.kind === "seq2seq" ? "The encoder reads one-hot character vectors. Its final hidden state and cell memory initialise the decoder. Greedy decoding feeds each selected character into the next step and stops at the end marker or the trained length limit." : topic.kind === "attention" ? "The model compares decoder and encoder hidden states with unscaled dot products. Softmax turns these scores into attention weights over source positions. The weighted context informs each output character." : "DistilGPT2 generates greedily from a quantized ONNX model. This export exposes vocabulary logits and key/value caches. Queries, attention-weight matrices and hidden activations inside each block are not exported and are not shown."}</p></details>
  </article>`;
}

export function bindDeepLearning(topic: DeepLearningTopic, typesetMath: TypesetMath): void {
  if (topic.kind === "bpe") {
    bindBpe();
    return;
  }
  bindExecutionWorkspace(topic.kind, typesetMath);
}
