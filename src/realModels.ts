import type { DeepLearningTopic } from "./data/topics.ts";
import type { ExecutionStep, ExecutionTrace, TraceTensor, TraceToken } from "./modelTrace.ts";
import { assertEquivalent, candidates, validateTensor } from "./modelNumerics.ts";

export type RealModelKind = Extract<DeepLearningTopic["kind"], "lstm" | "seq2seq" | "attention" | "transformer">;
export type ModelProgress = (message: string, percent?: number) => void;
const TFJS_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
const WASM_PATH = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/";
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const INPUT_VOCAB = "\n0123456789/-., " + ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"].join("").split("").filter((character, index, all) => all.indexOf(character) === index).join("");
const OUTPUT_VOCAB = "\n\t0123456789-";
const scripts = new Map<string, Promise<void>>();
let tfPromise: Promise<any> | undefined;
const assets = new Map<RealModelKind, Promise<any>>();
const loaded = new Set<RealModelKind>();

export function isModelLoaded(kind: RealModelKind): boolean { return loaded.has(kind); }

const modelNames: Record<RealModelKind, string> = {
  lstm: "IMDB LSTM", seq2seq: "English to French LSTM", attention: "Date conversion attention model", transformer: "DistilGPT2 q8",
};

class ModelInputError extends Error {
  name = "ModelInputError";
}

function loadScript(source: string): Promise<void> {
  const cached = scripts.get(source);
  if (cached) return cached;
  const loading = new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = source;
    script.crossOrigin = "anonymous";
    script.onload = () => resolve();
    script.onerror = () => { script.remove(); reject(new Error("The inference runtime could not be downloaded. Check your connection and retry.")); };
    document.head.append(script);
  }).catch(error => { scripts.delete(source); throw error; });
  scripts.set(source, loading);
  return loading;
}

async function tensorflow(progress: ModelProgress): Promise<any> {
  if (!tfPromise) tfPromise = (async () => {
    progress("Loading TensorFlow.js");
    await loadScript(TFJS_URL);
    await loadScript(`${WASM_PATH}tf-backend-wasm.min.js`);
    const tf = (window as typeof window & { tf?: any }).tf;
    if (!tf) throw new Error("TensorFlow.js could not initialise in this browser.");
    tf.wasm.setWasmPaths(WASM_PATH);
    await tf.setBackend("wasm");
    await tf.ready();
    return tf;
  })().catch(error => { tfPromise = undefined; throw error; });
  return tfPromise;
}

async function getAssets(kind: RealModelKind, progress: ModelProgress): Promise<any> {
  if (!assets.has(kind)) {
    const loading = (async () => {
      if (kind === "transformer") {
        progress("Loading the ONNX inference runtime");
        const transformers = await import(/* @vite-ignore */ TRANSFORMERS_URL);
        transformers.env.useBrowserCache = true;
        progress("Downloading DistilGPT2 weights");
        return transformers.pipeline("text-generation", "Xenova/distilgpt2", {
          device: "wasm", dtype: "q8", progress_callback: (event: { status?: string; progress?: number }) => {
            if (event.status === "progress_total" && event.progress !== undefined) progress("Downloading DistilGPT2 weights", event.progress);
          },
        });
      }
      const tf = await tensorflow(progress);
      progress(`Downloading ${modelNames[kind]}`);
      if (kind === "attention") {
        class GetLastTimestepLayer extends tf.layers.Layer {
          static className = "GetLastTimestepLayer";
          constructor(config: Record<string, unknown> = {}) { super(config); this.supportsMasking = true; }
          computeOutputShape(inputShape: number[]) { const shape = inputShape.slice(); shape.splice(shape.length - 2, 1); return shape; }
          call(input: any) { return tf.tidy(() => { const tensor = Array.isArray(input) ? input[0] : input; return tensor.gather([tensor.shape[1] - 1], 1).squeeze([1]); }); }
        }
        tf.serialization.registerClass(GetLastTimestepLayer);
        const model = await tf.loadLayersModel("https://storage.googleapis.com/tfjs-examples/date-conversion-attention/dist/model/model.json", { onProgress: (fraction: number) => progress("Downloading date model weights", fraction * 100) });
        return { model, probe: tf.model({ inputs: model.inputs, outputs: model.layers.map((layer: any) => layer.output) }) };
      }
      const path = kind === "lstm" ? "sentiment_lstm_v1" : "translation_en_fr_v1";
      const [model, response] = await Promise.all([
        tf.loadLayersModel(`https://storage.googleapis.com/tfjs-models/tfjs/${path}/model.json`, { onProgress: (fraction: number) => progress(`Downloading ${modelNames[kind]}`, fraction * 100) }),
        fetch(`https://storage.googleapis.com/tfjs-models/tfjs/${path}/metadata.json`),
      ]);
      if (!response.ok) { model.dispose(); throw new Error("The model vocabulary could not be downloaded. Retry when your connection is available."); }
      const metadata = await response.json();
      if (kind === "lstm") return { model, metadata };
      const encoder = tf.model({ inputs: model.inputs[0], outputs: [model.layers[2].output[1], model.layers[2].output[2]] });
      const dimension = model.layers[2].output[1].shape.at(-1);
      const h = tf.input({ shape: [dimension], name: "decoder_state_h" });
      const c = tf.input({ shape: [dimension], name: "decoder_state_c" });
      const applied = model.layers[3].apply(model.inputs[1], { initialState: [h, c] });
      const decoder = tf.model({ inputs: [model.inputs[1], h, c], outputs: [model.layers[4].apply(applied[0]), applied[1], applied[2]] });
      return { model, metadata, encoder, decoder };
    })().catch(error => { assets.delete(kind); throw error; });
    assets.set(kind, loading);
  }
  return assets.get(kind)!;
}

export async function loadModel(kind: RealModelKind, progress: ModelProgress): Promise<string> {
  await getAssets(kind, progress);
  loaded.add(kind);
  return kind === "transformer" ? "ONNX · WASM · q8" : `TensorFlow.js · ${(await tensorflow(progress)).getBackend()}`;
}

function snapshot(name: string, tensor: any, axes?: string[]): TraceTensor {
  return { name, shape: tensor.shape.slice(), values: Float32Array.from(tensor.dataSync()), dtype: tensor.dtype, axes };
}

function dispose(values: any[]): void { new Set(values).forEach(value => value?.dispose()); }
const yieldToPage = () => new Promise<void>(resolve => setTimeout(resolve, 0));

// Capture the runtime's cell and gate outputs without changing its arithmetic.
function recordCells(layer: any, tokens: TraceToken[], stage: string, steps: ExecutionStep[]): () => void {
  const cell = layer.cell;
  const originalCall = cell.call;
  const recurrentApply = cell.recurrentActivation.apply;
  const activationApply = cell.activation.apply;
  let position = 0;
  cell.call = function (inputs: any[], kwargs: any) {
    const gates: TraceTensor[] = [];
    let candidate: TraceTensor | undefined;
    cell.recurrentActivation.apply = function (...args: any[]) {
      const value = recurrentApply.apply(this, args);
      gates.push(snapshot(["Input gate", "Forget gate", "Output gate"][gates.length], value, ["batch", "unit"]));
      return value;
    };
    cell.activation.apply = function (...args: any[]) {
      const value = activationApply.apply(this, args);
      if (!candidate) candidate = snapshot("Candidate memory", value, ["batch", "unit"]);
      return value;
    };
    try {
      const output = originalCall.call(this, inputs, kwargs);
      if (gates.length !== 3 || !candidate) throw new Error("The LSTM cell returned an unexpected gate sequence.");
      const previousCell = inputs[2].dataSync() as Float32Array;
      const currentCell = output[2].dataSync() as Float32Array;
      assertEquivalent(Float32Array.from(previousCell, (value, unit) => gates[1].values[unit] * value + gates[0].values[unit] * candidate!.values[unit]), currentCell);
      assertEquivalent(Float32Array.from(currentCell, (value, unit) => gates[2].values[unit] * Math.tanh(value)), output[1].dataSync());
      const token = tokens[position];
      steps.push({ name: `${stage === "Encoder" ? "Encode" : "Read"} ${position + 1} · ${token?.text ?? "padding"}`, stage, operation: "LSTM cell", description: "The learned gates update cell memory and expose a hidden state. Select a unit to compare its values across timesteps. This model uses hard sigmoid gates.", formula: "c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde c_t,\\quad h_t=o_t\\odot\\tanh(c_t)", tokens, selectedToken: position, tensors: [snapshot("Cell input", inputs[0], ["batch", "feature"]), snapshot("Previous hidden state", inputs[1], ["batch", "unit"]), snapshot("Previous cell memory", inputs[2], ["batch", "unit"]), ...gates, ...(candidate ? [candidate] : []), snapshot("Cell memory", output[2], ["batch", "unit"]), snapshot("Hidden state", output[1], ["batch", "unit"])] });
      position += 1;
      return output;
    } finally {
      cell.recurrentActivation.apply = recurrentApply;
      cell.activation.apply = activationApply;
    }
  };
  return () => { cell.call = originalCall; };
}

async function runSentiment(text: string, progress: ModelProgress): Promise<ExecutionTrace> {
  const tf = await tensorflow(progress);
  const { model, metadata } = await getAssets("lstm", progress);
  const words = text.trim().toLocaleLowerCase("en").replace(/[.,!]/g, "").split(/\s+/).filter(Boolean).slice(-metadata.max_len);
  if (!words.length) throw new ModelInputError("Enter at least one word after punctuation is removed.");
  const ids = words.map(word => { const raw = metadata.word_index[word]; return raw === undefined || raw + metadata.index_from >= metadata.vocabulary_size ? 2 : raw + metadata.index_from; });
  const padding = metadata.max_len - ids.length;
  const padded = Array(padding).fill(0).concat(ids);
  const tokens = padded.map((id, position) => ({ id, position, text: position < padding ? "[pad]" : words[position - padding] }));
  const input = tf.tensor2d([padded], [1, metadata.max_len], "int32");
  const steps: ExecutionStep[] = [{ name: "Review → word IDs", stage: "Input", operation: "Vocabulary lookup", description: `${padding} leading padding positions. ID 2 represents an unknown word. Padding is processed by this LSTM; it is not masked.`, tokens: tokens.slice(padding), tensors: [snapshot("Word IDs", input, ["batch", "position"])] }];
  const started = performance.now();
  let baseline: any; let prediction: any; let embeddings: any;
  let classifierLogit: TraceTensor | undefined;
  try {
    baseline = model.predict(input);
    const reference = Array.from(await baseline.data()) as number[];
    await yieldToPage();
    embeddings = model.layers[0].apply(input);
    steps.push({ name: "Learned word vectors", stage: "Embedding", operation: "Embedding", description: "Each vocabulary ID selects a row from the trained embedding table. Position includes leading padding.", tokens: tokens.slice(padding), tensors: [snapshot("Embeddings", embeddings, ["batch", "position", "coordinate"])] });
    const restore = recordCells(model.layers[1], tokens, "Memory", steps);
    const activation = model.layers.at(-1).activation;
    const apply = activation.apply;
    activation.apply = function (value: any) { classifierLogit = snapshot("Classifier logit", value, ["batch", "class"]); return apply.call(this, value); };
    try { prediction = model.predict(input); } finally { restore(); activation.apply = apply; }
    const memory = steps.splice(2);
    if (padding) {
      const paddedSteps = memory.splice(0, padding);
      steps.push({ name: 'Padding recurrence', stage: 'Memory', operation: 'LSTM cells', description: `The ${padding} leading padding positions are real recurrent operations. Select a padding position to inspect its states and gates.`, tensors: paddedSteps[0].tensors.map((tensor, index) => ({ ...tensor, shape: [padding, tensor.values.length], values: Float32Array.from(paddedSteps.flatMap(step => Array.from(step.tensors[index].values))), axes: ['padding position', 'unit'] })) });
    }
    memory.forEach(step => { step.tokens = tokens.slice(padding); step.selectedToken! -= padding; step.name = `Word ${step.selectedToken! + 1} · ${step.tokens[step.selectedToken!].text}`; });
    steps.push(...memory);
    const values = Array.from(await prediction.data()) as number[];
    assertEquivalent(values, reference);
    const score = values[0];
    steps.push({ name: "Sentiment decision", stage: "Output", operation: "Dense + sigmoid", description: "The final hidden state is projected to a binary sentiment score. The recorded pass matches a separate uninstrumented model run.", formula: "p(\\mathrm{positive})=\\sigma(Wh_T+b)", tensors: [...(classifierLogit ? [classifierLogit] : []), snapshot("Positive probability", prediction, ["batch", "class"])], probabilities: [{ token: "negative", probability: 1 - score, selected: score < .5 }, { token: "positive", probability: score, selected: score >= .5 }] });
    return { output: score >= .5 ? "positive" : "negative", outputLabel: "Review sentiment", backend: `TensorFlow.js · ${tf.getBackend()}`, elapsed: performance.now() - started, steps, note: "The gate, hidden-state and cell-memory tensors are captured directly from the TensorFlow.js LSTM cell. Timing includes the numerical verification pass." };
  } finally { dispose([input, baseline, prediction, embeddings]); }
}

async function runTranslation(text: string, progress: ModelProgress): Promise<ExecutionTrace> {
  const tf = await tensorflow(progress);
  const { model, encoder, decoder, metadata } = await getAssets("seq2seq", progress);
  if (text.length > metadata.max_encoder_seq_length) throw new ModelInputError(`This model accepts at most ${metadata.max_encoder_seq_length} characters.`);
  const buffer = tf.buffer([1, metadata.max_encoder_seq_length, encoder.inputs[0].shape[2]]);
  Array.from(text).forEach((character, index) => {
    const id = metadata.input_token_index[character];
    if (id === undefined) throw new ModelInputError(`The trained vocabulary does not contain “${character}”.`);
    buffer.set(1, 0, index, id);
  });
  const input = buffer.toTensor();
  const tokens: TraceToken[] = Array.from({ length: metadata.max_encoder_seq_length }, (_, position) => ({ text: text[position] ?? "[pad]", position, id: metadata.input_token_index[text[position]] }));
  const steps: ExecutionStep[] = [{ name: "Source characters", stage: "Input", operation: "One-hot encoding", description: "Each character activates one vocabulary coordinate. Remaining positions are zero vectors.", tokens, tensors: [snapshot("One-hot characters", input, ["batch", "position", "character"])] }];
  const reverse = Object.fromEntries(Object.entries(metadata.target_token_index).map(([character, id]) => [String(id), character]));
  const label = (id: number) => reverse[id] === "\n" ? "[end]" : reverse[id] === "\t" ? "[start]" : reverse[id] ?? "[unknown]";
  const started = performance.now();
  let states: any[] = [];
  let baseline: any[] = [];
  let output = "";
  try {
    baseline = encoder.predict(input);
    const restore = recordCells(model.layers[2], tokens, "Encoder", steps);
    try { states = encoder.predict(input); } finally { restore(); }
    states.forEach((state, index) => assertEquivalent(state.dataSync(), baseline[index].dataSync()));
    dispose(baseline); baseline = [];
    let targetIndex = metadata.target_token_index["\t"];
    const generated: TraceToken[] = [{ text: "[start]", id: targetIndex, position: 0 }];
    for (let index = 0; index < metadata.max_decoder_seq_length; index += 1) {
      const targetBuffer = tf.buffer([1, 1, decoder.inputs[0].shape[2]]);
      targetBuffer.set(1, 0, 0, targetIndex);
      const target = targetBuffer.toTensor();
      let predicted: any[] = []; let reference: any[] = [];
      try {
        reference = decoder.predict([target, ...states]);
        const cellSteps: ExecutionStep[] = [];
        const restoreDecoder = recordCells(model.layers[3], [{ text: label(targetIndex), id: targetIndex, position: index }], "Decoder", cellSteps);
        let logits: TraceTensor | undefined;
        const activation = model.layers[4].activation;
        const apply = activation.apply;
        activation.apply = function (value: any) { logits = snapshot("Character logits", value, ["batch", "position", "character"]); return apply.call(this, value); };
        try { predicted = decoder.predict([target, ...states]); } finally { restoreDecoder(); activation.apply = apply; }
        predicted.forEach((tensor, position) => assertEquivalent(tensor.dataSync(), reference[position].dataSync()));
        const distribution = predicted[0].dataSync();
        const choices = candidates(distribution, label).map(candidate => ({ ...candidate, logit: logits?.values[candidate.id!] }));
        const chosen = choices[0].id!;
        const cell = cellSteps[0];
        steps.push({ ...cell, name: `Decode ${index + 1} · ${label(targetIndex)}`, tokens: generated.slice(), selectedToken: generated.length - 1, description: "The decoder receives the preceding character and encoder or previous decoder states. Its real gates produce the next hidden state and memory." });
        steps.push({ name: `Choose character ${index + 1} · ${label(chosen)}`, stage: "Decoder", operation: "Dense + softmax", description: "Greedy decoding selects the most probable character. Its ID becomes the next decoder input.", formula: "p(y_t)=\\operatorname{softmax}(Wh_t+b)", tensors: [...(logits ? [logits] : []), snapshot("Character probabilities", predicted[0], ["batch", "position", "character"])], probabilities: choices, tokens: generated.concat({ text: label(chosen), id: chosen, position: index + 1 }), selectedToken: generated.length });
        dispose(states); states = [predicted[1], predicted[2]];
        predicted = [predicted[0]];
        targetIndex = chosen;
        generated.push({ text: label(chosen), id: chosen, position: index + 1 });
        if (reverse[chosen] === "\n" || reverse[chosen] === undefined) break;
        output += reverse[chosen];
      } finally { dispose([target, ...predicted, ...reference]); }
      if (index % 4 === 0) await yieldToPage();
    }
    steps.push({ name: "Translation complete", stage: "Output", operation: "Character decoding", description: "The generated character IDs form the translation. Every encoder and decoder tensor was checked against an uninstrumented call.", tokens: generated, tensors: [{ name: "Generated character IDs", shape: [generated.length - 1], values: generated.slice(1).map(token => token.id!), dtype: "int32", axes: ["position"] }] });
    return { output: output || "No characters emitted", outputLabel: "English → French", backend: `TensorFlow.js · ${tf.getBackend()}`, elapsed: performance.now() - started, steps, note: "This compact character model demonstrates encoder–decoder mechanics. Translation quality is limited. Timing includes numerical verification." };
  } finally { dispose([input, ...states, ...baseline]); }
}

const dateLayers: Record<string, { name: string; stage: string; description: string; formula?: string; axes: string[] }> = {
  input1: { name: "Source character IDs", stage: "Input", description: "The source vocabulary maps each character to an ID. ID 0 pads the remaining positions.", axes: ["batch", "source position"] },
  input2: { name: "Previous output IDs", stage: "Input", description: "A start marker followed by the characters already chosen by greedy decoding. This is the final causal decoder pass.", axes: ["batch", "target position"] },
  embedding_Embedding1: { name: "Source embeddings", stage: "Embedding", description: "Each input character ID selects a learned vector.", axes: ["batch", "source position", "coordinate"] },
  embedding_Embedding2: { name: "Decoder embeddings", stage: "Embedding", description: "Previous output characters select the decoder's learned vectors.", axes: ["batch", "target position", "coordinate"] },
  lstm_LSTM1: { name: "Encoder hidden states", stage: "Encoder", description: "The encoder reads the source and returns a hidden state for every input position.", axes: ["batch", "source position", "unit"] },
  encoderLast: { name: "Final encoder state", stage: "Encoder", description: "This trained architecture uses the final encoder hidden state to initialise both decoder hidden state and cell memory.", axes: ["batch", "unit"] },
  lstm_LSTM2: { name: "Decoder hidden states", stage: "Decoder", description: "The decoder processes preceding output characters from left to right.", axes: ["batch", "target position", "unit"] },
  dot_Dot1: { name: "Attention scores", stage: "Attention", description: "Unscaled dot products compare each decoder state with each encoder state. This model does not divide by the square root of the feature dimension.", formula: "s_{t,i}=h_t^{\\mathrm{decoder}}\\cdot h_i^{\\mathrm{encoder}}", axes: ["batch", "target position", "source position"] },
  attention: { name: "Attention weights", stage: "Attention", description: "Softmax normalises the compatibility scores across source positions. Select a target row and source cell to inspect its exact weight.", formula: "\\alpha_{t,i}=\\frac{\\exp(s_{t,i})}{\\sum_j\\exp(s_{t,j})}", axes: ["batch", "target position", "source position"] },
  context: { name: "Weighted context", stage: "Attention", description: "The attention weights form a weighted sum of encoder states for each output position.", formula: "c_t=\\sum_i\\alpha_{t,i}h_i^{\\mathrm{encoder}}", axes: ["batch", "target position", "coordinate"] },
  concatenate_Concatenate1: { name: "Context + decoder state", stage: "Decoder", description: "The context vector and decoder state are joined along their feature axis.", axes: ["batch", "target position", "coordinate"] },
  time_distributed_TimeDistributed1: { name: "Hidden projection", stage: "Decoder", description: "The same trained dense layer and tanh activation transform each output position.", axes: ["batch", "target position", "coordinate"] },
  time_distributed_TimeDistributed2: { name: "Character probabilities", stage: "Output", description: "The final projection and softmax produce a distribution over the output character vocabulary at every target position.", axes: ["batch", "target position", "character"] },
};

async function runDateAttention(text: string, progress: ModelProgress): Promise<ExecutionTrace> {
  const tf = await tensorflow(progress);
  const { model, probe } = await getAssets("attention", progress);
  const source = text.trim().toUpperCase();
  if (source.length < 6 || source.length > 12) throw new ModelInputError("Use a supported date format between 6 and 12 characters.");
  const encoderBuffer = tf.buffer([1, 12], "float32");
  Array.from(source).forEach((character, index) => {
    const id = INPUT_VOCAB.indexOf(character);
    if (id < 0) throw new ModelInputError(`The trained vocabulary does not contain “${character}”.`);
    encoderBuffer.set(id, 0, index);
  });
  const encoder = encoderBuffer.toTensor();
  const decoderBuffer = tf.buffer([1, 10]);
  decoderBuffer.set(1, 0, 0);
  const started = performance.now();
  const chosen: number[] = [];
  const decisions: ExecutionStep[] = [];
  let decoderInput: any; let tensors: any[] = [];
  try {
    for (let position = 0; position < 10; position += 1) {
      const target = decoderBuffer.toTensor();
      let output: any;
      try {
        output = model.predict([encoder, target]);
        const row = output.dataSync().slice(position * OUTPUT_VOCAB.length, (position + 1) * OUTPUT_VOCAB.length);
        const choices = candidates(row, id => OUTPUT_VOCAB[id] === "\n" ? "[pad]" : OUTPUT_VOCAB[id] === "\t" ? "[start]" : OUTPUT_VOCAB[id]);
        chosen.push(choices[0].id!);
        decisions.push({ name: `Output ${position + 1} · ${choices[0].token}`, stage: "Decoder", operation: "Greedy character choice", description: "This distribution comes from the actual autoregressive pass for this output position.", tensors: [{ name: "Character probabilities", shape: [OUTPUT_VOCAB.length], values: Float32Array.from(row), dtype: "float32", axes: ["character"] }], probabilities: choices });
        if (position < 9) decoderBuffer.set(chosen[position], 0, position + 1);
      } finally { dispose([target, output]); }
      await yieldToPage();
    }
    decoderInput = decoderBuffer.toTensor();
    tensors = probe.predict([encoder, decoderInput]);
    const outputIndex = model.layers.findIndex((layer: any) => layer.output === model.outputs[0]);
    const outputTensor = tensors[outputIndex];
    for (let row = 0; row < 10; row += 1) assertEquivalent(outputTensor.dataSync().slice(row * OUTPUT_VOCAB.length, (row + 1) * OUTPUT_VOCAB.length), decisions[row].tensors[0].values);
    const output = chosen.map(id => OUTPUT_VOCAB[id]).join("");
    const sourceTokens = Array.from({ length: 12 }, (_, position) => ({ text: source[position] ?? "[pad]", position, id: encoderBuffer.get(0, position) }));
    const targetTokens = chosen.map((id, position) => ({ text: OUTPUT_VOCAB[id], id, position }));
    const attentionIndex = model.layers.findIndex((layer: any) => layer.name === "attention");
    const weights = tensors[attentionIndex].dataSync();
    const matrix = Array.from({ length: 10 }, (_, row) => Array.from(weights.slice(row * 12, row * 12 + 12)) as number[]);
    matrix.forEach(row => candidates(row, id => sourceTokens[id].text));
    const steps: ExecutionStep[] = model.layers.map((layer: any, index: number) => {
      const detail = dateLayers[layer.name];
      return { name: detail?.name ?? layer.name, stage: detail?.stage ?? "Decoder", operation: layer.getClassName(), description: detail?.description ?? "An output tensor from the trained graph.", formula: detail?.formula, tensors: [snapshot(detail?.name ?? layer.name, tensors[index], detail?.axes)], ...(layer.name === "input1" || layer.name === "embedding_Embedding1" || layer.name === "lstm_LSTM1" ? { tokens: sourceTokens } : {}), ...(layer.name === "attention" ? { attention: { source: sourceTokens.map(token => token.text), target: targetTokens.map(token => token.text), weights: matrix, row: 0 } } : {}) };
    });
    decisions.forEach((step, position) => {
      step.tokens = targetTokens; step.selectedToken = position;
      step.description = "The preceding character enters the decoder. Its hidden state scores the source, attention forms a weighted context, and the output distribution selects the next character.";
      for (const [layerName, name, axis] of [["lstm_LSTM2", "Decoder hidden state", "unit"], ["dot_Dot1", "Attention scores", "source position"], ["attention", "Attention weights", "source position"], ["context", "Weighted context", "coordinate"]]) {
        const tensor = tensors[model.layers.findIndex((layer: any) => layer.name === layerName)];
        const width = tensor.shape.at(-1);
        step.tensors.push({ name, shape: [width], values: Float32Array.from(tensor.dataSync().slice(position * width, (position + 1) * width)), dtype: tensor.dtype, axes: [axis] });
      }
      step.attention = { source: sourceTokens.map(token => token.text), target: targetTokens.map(token => token.text), weights: matrix, row: position };
    });
    steps.push(...decisions, { name: "Normalised date", stage: "Output", operation: "Character decoding", description: "Greedy character IDs form the final date. The layer probe matches the probabilities used at every decoding position.", tokens: targetTokens, tensors: [{ name: "Output character IDs", shape: [10], values: chosen, dtype: "int32", axes: ["position"] }] });
    return { output, outputLabel: "Normalised date", backend: `TensorFlow.js · ${tf.getBackend()}`, elapsed: performance.now() - started, steps, note: "Attention values are exposed by the trained graph. The layer tensors record the final causal decoding pass; the character decisions retain their original per-position distributions." };
  } finally { dispose([encoder, decoderInput, ...tensors]); }
}

async function runTransformer(text: string, progress: ModelProgress): Promise<ExecutionTrace> {
  const generator = await getAssets("transformer", progress);
  const encoded = generator.tokenizer(text);
  const ids = Array.from(encoded.input_ids.data, Number);
  if (!ids.length) throw new ModelInputError("Enter a prompt that contains at least one token.");
  if (ids.length > 1012) throw new ModelInputError("Use at most 1,012 prompt tokens, leaving 12 positions in the model context for generation.");
  const decode = (id: number) => generator.tokenizer.decode([id]) || "[empty]";
  const tokens: TraceToken[] = ids.map((id, position) => ({ id, position, text: decode(id) }));
  const steps: ExecutionStep[] = [{ name: "Prompt → token IDs", stage: "Input", operation: "GPT-2 tokenizer", description: "The tokenizer splits text into subword tokens, preserving spaces. Each ID indexes the model vocabulary.", tokens, tensors: [{ name: "Token IDs", shape: [1, ids.length], values: ids, dtype: "int64", axes: ["batch", "position"] }] }];
  const generation: ExecutionStep[] = [];
  const forward = generator.model.forward;
  const started = performance.now();
  progress("Running greedy decoding and checking the recorded trace");
  const baseline = await generator(text, { max_new_tokens: 12, do_sample: false, return_full_text: false });
  let call = 0;
  const selected: TraceToken[] = [];
  generator.model.forward = async function (...args: any[]) {
    const result = await forward.apply(this, args);
    const logits = result.logits;
    const vocabulary = logits.dims.at(-1);
    const positions = logits.dims.at(-2);
    if (call === 0) {
      for (const [name, tensor] of Object.entries(result) as Array<[string, any]>) {
        const match = name.match(/(?:present|past_key_values)\.(\d+)\.(key|value)$/);
        if (!match || !tensor?.dims || !tensor?.data) continue;
        const block = Number(match[1]) + 1;
        let step = steps.find(step => step.name === `Block ${block} · key/value cache`);
        if (!step) {
          step = { name: `Block ${block} · key/value cache`, stage: "Attention", operation: "Exported key/value cache", description: "These are the real projected keys and values for each head and prompt position. This ONNX graph does not export queries, attention weights, residuals or feedforward activations. A cache is not an attention matrix.", tokens, tensors: [] };
          steps.push(step);
        }
        step.tensors.push({ name: match[2] === "key" ? "Key cache" : "Value cache", shape: tensor.dims.slice(), values: Float32Array.from(tensor.data), dtype: tensor.type, axes: ["batch", "head", "position", "coordinate"] });
      }
      for (let position = 0; position < positions; position += 1) {
        const row = logits.data.subarray(position * vocabulary, (position + 1) * vocabulary);
        const probabilities = candidates(row, decode, true).map(candidate => ({ ...candidate, selected: position === positions - 1 && candidate.selected }));
        steps.push({ name: `Prompt ${position + 1} · next-token distribution`, stage: "Decoder", operation: "LM head + softmax", description: "The causal model uses the prompt through the selected position to predict the next token. The full vocabulary supplies the softmax denominator; the eight strongest candidates are shown.", formula: "p(x_{t+1}=i)=\\frac{\\exp(z_i)}{\\sum_j\\exp(z_j)}", tokens, selectedToken: position, probabilities, tensors: position === positions - 1 ? [{ name: "Vocabulary logits", shape: [vocabulary], values: Float32Array.from(row), dtype: logits.type, axes: ["vocabulary ID"] }] : [{ name: "Top candidate logits", shape: [probabilities.length], values: probabilities.map(candidate => candidate.logit!), dtype: logits.type, axes: ["candidate rank"], description: "An exact subset of the vocabulary logits, ordered by descending probability." }] });
      }
    }
    if (call > 0) {
      const updates: TraceTensor[] = [];
      for (const [name, tensor] of Object.entries(result) as Array<[string, any]>) {
        const match = name.match(/(?:present|past_key_values)\.(\d+)\.(key|value)$/);
        if (!match || !tensor?.dims || tensor.dims.length !== 4) continue;
        const [batch, heads, length, dimensions] = tensor.dims;
        const last = new Float32Array(batch * heads * dimensions);
        for (let head = 0; head < batch * heads; head++) last.set(tensor.data.subarray((head * length + length - 1) * dimensions, (head * length + length) * dimensions), head * dimensions);
        updates.push({ name: `Block ${Number(match[1]) + 1} ${match[2]} cache`, shape: [batch, heads, 1, dimensions], values: last, dtype: tensor.type, axes: ["batch", "head", "new position", "coordinate"], description: `Last position of the returned cache [${tensor.dims.join(" × ")}], at position ${length - 1}.` });
      }
      if (updates.length) generation.push({ name: `Cache update ${call} · position ${ids.length + call - 1}`, stage: "Decoder", operation: "Key/value cache update", description: "The latest input token adds one key and one value per head. Earlier positions remain in the cache; only the new position is shown here.", tokens: [selected.at(-1)!], selectedToken: 0, tensors: updates });
    }
    const row = logits.data.subarray((positions - 1) * vocabulary, positions * vocabulary);
    const probabilities = candidates(row, decode, true);
    selected.push({ text: decode(probabilities[0].id!), id: probabilities[0].id, position: ids.length + call });
    generation.push({ name: `Generate ${call + 1} · ${probabilities[0].token}`, stage: "Decoder", operation: "Greedy next-token choice", description: "The highest-probability vocabulary token is appended to the context. The next forward pass reuses the model's real key/value cache.", formula: "x_{t+1}=\\operatorname{argmax}_i z_i", tokens: tokens.concat(selected), selectedToken: tokens.length + selected.length - 1, probabilities, tensors: [{ name: "Vocabulary logits", shape: [vocabulary], values: Float32Array.from(row), dtype: logits.type, axes: ["vocabulary ID"] }] });
    call += 1;
    return result;
  };
  let generated: any;
  try { generated = await generator(text, { max_new_tokens: 12, do_sample: false, return_full_text: false }); }
  finally { generator.model.forward = forward; }
  const continuation = generated[0].generated_text;
  if (continuation !== baseline[0].generated_text) throw new Error("The recorded generation did not match the model's original output.");
  steps.push(...generation, { name: "Continuation complete", stage: "Output", operation: "Token decoding", description: "The generated token IDs are decoded into text. The continuation exactly matches a separate uninstrumented greedy run.", tokens: selected, tensors: [{ name: "Generated token IDs", shape: [selected.length], values: selected.map(token => token.id!), dtype: "int64", axes: ["generated position"] }] });
  return { output: continuation || "No text emitted", outputLabel: "Greedy continuation", backend: "ONNX · WASM · q8", elapsed: performance.now() - started, steps, note: "Only exported key/value caches and logits are shown. Queries, attention weights and other internal block tensors are not exposed by this ONNX export. Timing includes the verification run." };
}

let execution: Promise<unknown> = Promise.resolve();
export async function runModel(kind: RealModelKind, text: string, progress: ModelProgress): Promise<ExecutionTrace> {
  if (!text.trim()) throw new ModelInputError("Enter an input first.");
  // Shared model methods are instrumented only while their own run holds the queue.
  const run = execution.catch(() => {}).then(async () => {
    const runInference = kind === "lstm" ? runSentiment : kind === "seq2seq" ? runTranslation : kind === "attention" ? runDateAttention : runTransformer;
    progress("Running the model");
    await yieldToPage();
    const trace = await runInference(text, progress);
    trace.steps.forEach(step => step.tensors.forEach(validateTensor));
    return trace;
  });
  execution = run.then(() => undefined, () => undefined);
  return run;
}
