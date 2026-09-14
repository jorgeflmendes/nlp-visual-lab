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
  <a href="#solvers">Solvers</a> ·
  <a href="#deep-learning">Deep learning</a> ·
  <a href="#technology">Technology</a> ·
  <a href="#getting-started">Getting started</a> ·
  <a href="#deployment">Deployment</a>
</p>

## Overview

NLP Visual Lab turns the intermediate calculations behind common natural language processing methods into explorable tables, diagrams and controls. Classical solvers run locally in the browser. Neural experiments download their model files on first use and then perform inference in the browser.

### Solvers

| Method | What the page shows |
| --- | --- |
| Minimum edit distance | Dynamic-programming matrix, operation costs and an optimal edit sequence |
| Bigram language model | Token counts, conditional probabilities and the complete sentence probability product |
| Viterbi HMM | Editable model probabilities, score and backpointer tables, trellis and decoded path |
| CKY parser | Grammar rules, chart construction and recovered parse trees |

### Deep learning

| Model | Experiment |
| --- | --- |
| LSTM | Sentiment classification with token-level input inspection |
| Sequence-to-sequence | Character-level English-to-French translation |
| Attention | Date normalization with an attention heatmap |
| DistilGPT2 | Autoregressive text continuation with generation controls |

## Technology

- TypeScript and Vite for the application
- KaTeX for mathematical notation
- Rust and WebAssembly for the edit-distance core
- TensorFlow.js and Transformers.js for neural model inference
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
public/                 Static assets
.github/workflows/      Test, build and GitHub Pages deployment
```

## Model sources

The repository does not contain model weights. The neural experiments load these published models at runtime:

- [TensorFlow.js sentiment model](https://github.com/tensorflow/tfjs-examples/tree/master/sentiment)
- [TensorFlow.js translation model](https://github.com/tensorflow/tfjs-examples/tree/master/translation)
- [TensorFlow.js date conversion with attention](https://github.com/tensorflow/tfjs-examples/tree/master/date-conversion-attention)
- [Xenova/distilgpt2](https://huggingface.co/Xenova/distilgpt2)

An internet connection is required the first time a neural model is loaded. The classical solvers work without external model downloads.

## Deployment

Pushes to `main` or `master` run the test suite, create the production build and deploy it to GitHub Pages. Select **GitHub Actions** as the Pages source in the repository settings.
