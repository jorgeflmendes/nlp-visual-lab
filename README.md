<p align="center">
  <img src="public/logo.svg" alt="NLP Visual Lab" width="360">
</p>

<h1 align="center">NLP Visual Lab</h1>

<p align="center">
  <strong>Trace classical NLP algorithms and trained neural models step by step.</strong>
</p>

<p align="center">
  Inspect the tables, probabilities, paths and activations behind every result directly in the browser.
</p>

<p align="center">
  <a href="https://github.com/jorgeflmendes/nlp-visual-lab/actions/workflows/deploy.yml"><img src="https://github.com/jorgeflmendes/nlp-visual-lab/actions/workflows/deploy.yml/badge.svg" alt="Deploy status"></a>
  <img src="https://img.shields.io/badge/TypeScript-strict-172623" alt="TypeScript strict">
  <img src="https://img.shields.io/badge/inference-browser-172623" alt="Browser inference">
  <img src="https://img.shields.io/badge/core-WebAssembly-172623" alt="WebAssembly core">
</p>

<p align="center">
  <a href="https://jorgeflmendes.github.io/nlp-visual-lab/">Open NLP Visual Lab</a>
</p>

<p align="center">
  <a href="#foundations--representations">Foundations</a> ·
  <a href="#structured--classical-nlp">Structured NLP</a> ·
  <a href="#neural-architectures">Neural architectures</a> ·
  <a href="#technology">Technology</a> ·
  <a href="#getting-started">Getting started</a> ·
  <a href="#deployment">Deployment</a>
</p>

## Overview

NLP Visual Lab turns the intermediate calculations behind common natural language processing methods into explorable tables, diagrams and controls. Foundations and structured NLP algorithms run deterministically in the browser. Neural experiments download their model files on first use and then perform inference in the browser.

### Foundations & Representations

| Method | What the page shows |
| --- | --- |
| Byte Pair Encoding | Pair leaderboard, merge decisions, evolving vocabulary and subword tokenization trace |
| Minimum edit distance | Dynamic-programming matrix, operation costs and optimal edit sequence (WebAssembly core) |
| TF-IDF Vector Space | Term frequencies, inverse document frequencies, vector weights and cosine similarity ranking |
| Word Embeddings | 100-dimensional GloVe representations (20k vocabulary), arbitrary vector arithmetic ($\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}}$), $K$-NN cosine search, cross-similarity heatmaps, and 2D PCA projection |

### Structured & Classical NLP

| Method | What the page shows |
| --- | --- |
| Bigram language model | Token counts, conditional probabilities and the complete sentence probability product |
| Viterbi HMM | Editable model probabilities, score and backpointer tables, trellis and decoded path |
| CKY parser | Grammar rules in Chomsky Normal Form, pyramidal chart construction and recovered parse trees |

### Neural Architectures

| Model | Experiment |
| --- | --- |
| LSTM | Sentiment classification with token-level input inspection and gated recurrent memory |
| Sequence-to-sequence | Character-level English-to-French translation with encoder and decoder states |
| Attention | Date normalization with an attention matrix and dynamic context vectors |
| Sentence Embeddings | Real in-browser all-MiniLM-L6-v2 ONNX q8 inference via WebAssembly: 384-dimensional dense vectors, multi-sentence cosine similarity heatmaps, semantic search retrieval ranking, 2D PCA projection, and step-by-step unrolled encoder forward pass |
| DistilGPT2 | Autoregressive text continuation with vocabulary logits and generation controls |

## Technology

- TypeScript and Vite for the application
- KaTeX for mathematical notation
- Rust and WebAssembly for the edit-distance core
- ONNX Runtime Web & Transformers.js for in-browser sentence embeddings and causal language modeling
- TensorFlow.js for sentiment classification, sequence-to-sequence translation, and additive attention
- 2D PCA spatial projection via power iteration with Gram-Schmidt orthogonalization
- Node.js test runner for algorithm and trace regression tests
- GitHub Actions and GitHub Pages for deployment

## Getting started

Install [Node.js](https://nodejs.org/) 22.12 or newer and [Rust](https://www.rust-lang.org/tools/install), then add the WebAssembly compilation target:

```sh
rustup target add wasm32-unknown-unknown
```

Install the JavaScript dependencies and start the development server:

```sh
npm ci
npm run dev
```

Vite prints the local URL when the server is ready.

## Commands

| Command | Purpose |
| --- | --- |
| `npm run dev` | Build the WebAssembly module and start Vite |
| `npm test` | Run the algorithm and trace tests |
| `npm run build` | Build WebAssembly, type-check and create `dist` |
| `npm run preview` | Serve the production build locally |
| `npm run build:wasm` | Compile only the Rust WebAssembly module |

## Project structure

```text
src/                    Application, algorithms and interactive traces
src/data/               Topic definitions and examples
tests/                  Algorithm and trace regression tests
wasm-core/              Rust source for the WebAssembly core
scripts/build-wasm.mjs  WebAssembly build step
public/models/          Static models (GloVe 100d vectors and all-MiniLM-L6-v2 ONNX)
public/                 Static assets and WebAssembly binaries
.github/workflows/      Test, build and GitHub Pages deployment
```

## Model sources

The application runs 100% client-side in the browser with no backend servers:

### Local assets (bundled in `public/models/`)
- **GloVe 100d**: 20,000-word vocabulary stored as a compact binary float32 tensor (`glove-20k-100d.bin`) and JSON vocabulary index for instant vector arithmetic.
- **all-MiniLM-L6-v2**: 8-bit quantized ONNX model (`model_quantized.onnx`) and tokenizer assets (`tokenizer.json`, `vocab.txt`) served directly from the repository for serverless browser inference via WebAssembly.

### Remote runtime models
The remaining neural experiments download published model weights on first use and cache them locally in the browser:

- [TensorFlow.js sentiment model](https://github.com/tensorflow/tfjs-examples/tree/master/sentiment)
- [TensorFlow.js translation model](https://github.com/tensorflow/tfjs-examples/tree/master/translation)
- [TensorFlow.js date conversion with attention](https://github.com/tensorflow/tfjs-examples/tree/master/date-conversion-attention)
- [Xenova/distilgpt2](https://huggingface.co/Xenova/distilgpt2)

The classical solvers, vector spaces, and local embedding models work completely offline without external downloads.

## Deployment

Pushes to `main` or `master` run the test suite, create the production build and deploy it to GitHub Pages. Select **GitHub Actions** as the Pages source in the repository settings.
