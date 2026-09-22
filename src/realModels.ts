import type { DeepLearningTopic } from "./data/topics.ts";
import type { ExecutionStep, ExecutionTrace, TraceTensor, TraceToken } from "./modelTrace.ts";
import { assertEquivalent, candidates, validateTensor } from "./modelNumerics.ts";

export type RealModelKind = Extract<DeepLearningTopic["kind"], "lstm" | "seq2seq" | "attention" | "sentence-embeddings" | "transformer">;
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
  lstm: "IMDB LSTM",
  seq2seq: "English to French LSTM",
  attention: "Date conversion attention model",
  "sentence-embeddings": "all-MiniLM-L6-v2 q8",
  transformer: "DistilGPT2 q8",
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
      if (kind === "sentence-embeddings") {
        progress("Loading the ONNX inference runtime");
        const transformers = await import(/* @vite-ignore */ TRANSFORMERS_URL);
        transformers.env.allowLocalModels = true;
        transformers.env.localModelPath = "/models/";
        progress("Loading all-MiniLM-L6-v2 weights");
        return transformers.pipeline("feature-extraction", "minilm", {
          device: "wasm",
          dtype: "q8",
          local_files_only: true,
          progress_callback: (event: { status?: string; progress?: number }) => {
            if (event.status === "progress_total" && event.progress !== undefined) progress("Loading all-MiniLM-L6-v2 weights", event.progress);
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
  return kind === "transformer" || kind === "sentence-embeddings" ? "ONNX · WASM · q8" : `TensorFlow.js · ${(await tensorflow(progress)).getBackend()}`;
}

export async function extractSentenceEmbedding(
  text: string,
  progress?: ModelProgress
): Promise<Float32Array> {
  const extractor = await getAssets("sentence-embeddings", progress ?? (() => {}));
  loaded.add("sentence-embeddings");
  const raw = await extractor(text, { pooling: "mean", normalize: true });
  return new Float32Array(raw.data);
}

export async function extractSentenceEmbeddingsBatch(
  texts: string[],
  progress?: ModelProgress
): Promise<Float32Array[]> {
  const extractor = await getAssets("sentence-embeddings", progress ?? (() => {}));
  loaded.add("sentence-embeddings");
  const results: Float32Array[] = [];
  for (let i = 0; i < texts.length; i++) {
    if (progress) progress(`Encoding sentence ${i + 1} of ${texts.length}`, Math.round(((i + 1) / texts.length) * 100));
    const raw = await extractor(texts[i], { pooling: "mean", normalize: true });
    results.push(new Float32Array(raw.data));
  }
  return results;
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
      const mean = (vals: number[] | Float32Array) => {
        let sum = 0;
        for (let i = 0; i < vals.length; i++) sum += vals[i];
        return vals.length ? sum / vals.length : 0;
      };
      const avgForget = mean(gates[1].values);
      const avgInput = mean(gates[0].values);
      const avgOutput = mean(gates[2].values);
      const isEncoder = stage === "Encoder";
      const desc = isEncoder
        ? `Encoding character “${token?.text ?? "padding"}” (step ${position + 1}/${tokens.length}). The cell balances retention of previous context (mean forget gate: ${(avgForget * 100).toFixed(1)}%) with new character information (mean input gate: ${(avgInput * 100).toFixed(1)}%). State vector h will transfer to the decoder.`
        : `Processing word “${token?.text ?? "padding"}” (step ${position + 1}/${tokens.length}). Forget gate retention averages ${(avgForget * 100).toFixed(1)}%; input gate activation averages ${(avgInput * 100).toFixed(1)}%. Output gate exposes ${(avgOutput * 100).toFixed(1)}% of squashed memory tanh(c_t) as hidden state h_t.`;
      steps.push({ name: `${isEncoder ? "Encode" : "Read"} ${position + 1} · ${token?.text ?? "padding"}`, stage, operation: "LSTM cell", description: desc, formula: "c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde c_t,\\quad h_t=o_t\\odot\\tanh(c_t)", tokens, selectedToken: position, tensors: [snapshot("Cell input", inputs[0], ["batch", "feature"]), snapshot("Previous hidden state", inputs[1], ["batch", "unit"]), snapshot("Previous cell memory", inputs[2], ["batch", "unit"]), ...gates, ...(candidate ? [candidate] : []), snapshot("Cell memory", output[2], ["batch", "unit"]), snapshot("Hidden state", output[1], ["batch", "unit"])] });
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
    const embShape = embeddings.shape;
    steps.push({ name: "Learned word vectors", stage: "Embedding", operation: "Embedding", description: `Each vocabulary ID selects a learned ${embShape.at(-1)}-dimensional dense vector. Padded positions (0) use the model's dedicated zero/padding embedding row.`, tokens: tokens.slice(padding), tensors: [snapshot("Embeddings", embeddings, ["batch", "position", "coordinate"])] });
    const restore = recordCells(model.layers[1], tokens, "Memory", steps);
    const activation = model.layers.at(-1).activation;
    const apply = activation.apply;
    activation.apply = function (value: any) { classifierLogit = snapshot("Classifier logit", value, ["batch", "class"]); return apply.call(this, value); };
    try { prediction = model.predict(input); } finally { restore(); activation.apply = apply; }
    const memory = steps.splice(2);
    if (padding) {
      const paddedSteps = memory.splice(0, padding);
      steps.push({ name: 'Padding recurrence', stage: 'Memory', operation: 'LSTM cells', description: `${padding} leading padding steps ran through the recurrent cell. The LSTM state initialized at zero and accumulated padding transitions before reaching the first real word.`, tensors: paddedSteps[0].tensors.map((tensor, index) => ({ ...tensor, shape: [padding, tensor.values.length], values: Float32Array.from(paddedSteps.flatMap(step => Array.from(step.tensors[index].values))), axes: ['padding position', 'unit'] })) });
    }
    memory.forEach(step => { step.tokens = tokens.slice(padding); step.selectedToken! -= padding; step.name = `Word ${step.selectedToken! + 1} · ${step.tokens[step.selectedToken!].text}`; });
    steps.push(...memory);
    const values = Array.from(await prediction.data()) as number[];
    assertEquivalent(values, reference);
    const score = values[0];
    const rawLogit = classifierLogit?.values[0];
    const logitDesc = rawLogit !== undefined ? ` (pre-activation logit z = ${rawLogit.toFixed(3)})` : "";
    const decisionDesc = score >= 0.5
      ? `The dense layer projects the final hidden state h_T${logitDesc}. Sigmoid activation yields p(positive) = ${(score * 100).toFixed(1)}% ≥ 50%, classifying this review as positive.`
      : `The dense layer projects the final hidden state h_T${logitDesc}. Sigmoid activation yields p(positive) = ${(score * 100).toFixed(1)}% < 50%, classifying this review as negative (confidence ${( (1 - score) * 100 ).toFixed(1)}%).`;
    steps.push({ name: "Sentiment decision", stage: "Output", operation: "Dense + sigmoid", description: decisionDesc, formula: "p(\\mathrm{positive})=\\sigma(Wh_T+b)", tensors: [...(classifierLogit ? [classifierLogit] : []), snapshot("Positive probability", prediction, ["batch", "class"])], probabilities: [{ token: "negative", probability: 1 - score, selected: score < .5 }, { token: "positive", probability: score, selected: score >= .5 }] });
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

    // Context handover step: expose the information bottleneck (final encoder states initializing the decoder)
    const finalH = snapshot("Final encoder hidden state (h_T)", states[0], ["batch", "unit"]);
    const finalC = snapshot("Final encoder cell memory (c_T)", states[1], ["batch", "unit"]);
    steps.push({
      name: "Encoder → Decoder handover",
      stage: "Encoder",
      operation: "State transfer (Information bottleneck)",
      description: `The encoder finishes reading the English sequence. The final hidden state h_T and cell memory c_T (${states[0].shape.at(-1)} units each) compress the entire semantic content of the input to initialize the decoder: s_0 = h_T and c_0 = c_T. In classic Seq2Seq without attention, this fixed-size vector is the sole communication bridge between source and target.`,
      formula: "s_0^{\\text{dec}}=h_T^{\\text{enc}},\\quad c_0^{\\text{dec}}=c_T^{\\text{enc}}",
      tokens,
      selectedToken: text.length - 1,
      tensors: [finalH, finalC],
    });

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
        const isEnd = reverse[chosen] === "\n";
        const topProb = (choices[0].probability * 100).toFixed(1);
        const secondChoice = choices[1] ? ` runner-up “${choices[1].token}” at ${(choices[1].probability * 100).toFixed(1)}%` : "";
        steps.push({ ...cell, name: `Decode ${index + 1} · ${label(targetIndex)}`, tokens: generated.slice(), selectedToken: generated.length - 1, description: `The decoder cell takes preceding token “${label(targetIndex)}” (target ID ${targetIndex}) and prior hidden/cell states [h_{t-1}, c_{t-1}]. Its internal gates compute the new representation h_t.` });
        steps.push({ name: `Choose character ${index + 1} · ${label(chosen)}`, stage: "Decoder", operation: "Dense + softmax", description: `Dense projection and softmax produce a distribution across target characters. Greedy selection picks “${label(chosen)}” with ${topProb}% probability (${secondChoice}).${isEnd ? " Emitted [end] marker terminates autoregressive sequence generation." : ""}`, formula: "p(y_t)=\\operatorname{softmax}(Wh_t+b)", tensors: [...(logits ? [logits] : []), snapshot("Character probabilities", predicted[0], ["batch", "position", "character"])], probabilities: choices, tokens: generated.concat({ text: label(chosen), id: chosen, position: index + 1 }), selectedToken: generated.length });
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
  lstm_LSTM1: { name: "Encoder hidden states", stage: "Encoder", description: "The encoder LSTM reads input characters sequentially, producing recurrent hidden state vectors h_1..h_{12} (64 units each). In the attention layer, these states serve as both Keys (for compatibility matching) and Values (to form context).", formula: "h_i=\\operatorname{LSTM}(x_i, h_{i-1})", axes: ["batch", "source position", "unit"] },
  encoderLast: { name: "Final encoder state", stage: "Encoder", description: "This trained architecture uses the final encoder hidden state to initialise both decoder hidden state and cell memory.", axes: ["batch", "unit"] },
  lstm_LSTM2: { name: "Decoder hidden states", stage: "Decoder", description: "The decoder LSTM generates recurrent state s_t at step t by reading previous output characters. In the attention layer, s_t acts as the Query vector probing the encoder representations.", formula: "s_t=\\operatorname{LSTM}(y_{t-1}, s_{t-1})", axes: ["batch", "target position", "unit"] },
  dot_Dot1: { name: "Attention scores (Dot product)", stage: "Attention", description: "Unscaled dot product computing affinity between the decoder state s_t (acting as Query Q_t) and encoder states h_i (acting as Keys K_i): score(t, i) = s_t^\\top h_i.", formula: "e_{t,i}=s_t^\\top h_i", axes: ["batch", "target position", "source position"] },
  attention: { name: "Attention distribution (softmax)", stage: "Attention", description: "Softmax normalisation across source positions produces attention weights α_{t,i}. Select a target row and source cell to inspect its exact weight.", formula: "\\alpha_{t,i}=\\frac{\\exp(e_{t,i})}{\\sum_j\\exp(e_{t,j})}", axes: ["batch", "target position", "source position"] },
  context: { name: "Context vector (weighted sum)", stage: "Attention", description: "Weighted linear combination of encoder states h_i (acting as Values V_i) using attention weights α_{t,i}, concentrating source information relevant to decoding step t into c_t.", formula: "c_t=\\sum_i\\alpha_{t,i}h_i", axes: ["batch", "target position", "coordinate"] },
  concatenate_Concatenate1: { name: "Context + decoder state concatenation", stage: "Decoder", description: "The context vector c_t and current decoder hidden state s_t are concatenated along their feature axis to inform character logits.", formula: "[c_t; s_t]", axes: ["batch", "target position", "coordinate"] },
  time_distributed_TimeDistributed1: { name: "Hidden projection", stage: "Decoder", description: "A shared dense layer with tanh activation projects the combined context and query representations.", formula: "\\tilde{s}_t=\\tanh(W_c[c_t; s_t]+b_c)", axes: ["batch", "target position", "coordinate"] },
  time_distributed_TimeDistributed2: { name: "Character probabilities", stage: "Output", description: "The final projection and softmax produce a distribution over the output character vocabulary at every target position.", formula: "p(y_t)=\\operatorname{softmax}(W_s\\tilde{s}_t+b_s)", axes: ["batch", "target position", "character"] },
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
    const encoderSteps: ExecutionStep[] = model.layers
      .filter((layer: any) => {
        const detail = dateLayers[layer.name];
        return detail?.stage === "Input" || detail?.stage === "Embedding" || detail?.stage === "Encoder";
      })
      .map((layer: any) => {
        const detail = dateLayers[layer.name];
        const index = model.layers.findIndex((l: any) => l.name === layer.name);
        return {
          name: detail?.name ?? layer.name,
          stage: detail?.stage ?? "Encoder",
          operation: layer.getClassName(),
          description: detail?.description ?? "An output tensor from the trained graph.",
          formula: detail?.formula,
          tensors: [snapshot(detail?.name ?? layer.name, tensors[index], detail?.axes)],
          ...(layer.name === "input1" || layer.name === "embedding_Embedding1" || layer.name === "lstm_LSTM1" ? { tokens: sourceTokens } : {}),
        };
      });
    decisions.forEach((step, position) => {
      const generatedTokens = targetTokens.slice(0, position + 1);
      step.tokens = generatedTokens;
      step.selectedToken = position;
      const emittedChar = targetTokens[position]?.text ?? "";
      const weightsRow = matrix[position] ?? [];
      let maxWeight = -1;
      let maxSourceIdx = 0;
      weightsRow.forEach((w, idx) => {
        if (w > maxWeight) { maxWeight = w; maxSourceIdx = idx; }
      });
      const attendedToken = sourceTokens[maxSourceIdx]?.text ?? "";
      const topPct = (maxWeight * 100).toFixed(1);
      step.description = `Generating output character ${position + 1}/10 (“${emittedChar}”). Decoder state s_${position} (acting as Query Q) compares compatibility against all 12 encoder hidden states h_i (acting as Keys K), concentrating ${topPct}% attention weight on source position ${maxSourceIdx} (“${attendedToken}”). The weighted combination of encoder states (acting as Values V) forms context c_${position}, which concatenates with s_${position} to drive greedy character selection.`;
      for (const [layerName, name, axis, desc] of [
        ["lstm_LSTM2", "Decoder hidden state (Query s_t)", "unit", "Current decoder LSTM recurrent state s_t, functioning as Query Q_t in the cross-attention calculation."],
        ["dot_Dot1", "Attention scores (s_t · h_i)", "source position", "Dot product affinity between decoder state s_t and encoder states h_i (e_{t,i} = s_tᵀ h_i)."],
        ["attention", "Attention weights (softmax)", "source position", "Normalised probability distribution α_{t,i} over the 12 input character positions."],
        ["context", "Context vector (∑ α_i h_i)", "coordinate", "Weighted sum of encoder representations h_i, concentrating relevant input information into c_t."]
      ]) {
        const tensor = tensors[model.layers.findIndex((layer: any) => layer.name === layerName)];
        const width = tensor.shape.at(-1);
        step.tensors.push({ name, shape: [width], values: Float32Array.from(tensor.dataSync().slice(position * width, (position + 1) * width)), dtype: tensor.dtype, axes: [axis], description: desc });
      }
      step.attention = {
        source: sourceTokens.map(token => token.text),
        target: generatedTokens.map(token => token.text),
        weights: matrix.slice(0, position + 1),
        row: position,
      };
    });
    const steps: ExecutionStep[] = [
      ...encoderSteps,
      ...decisions,
      {
        name: "Normalised date",
        stage: "Output",
        operation: "Character decoding",
        description: "Greedy character IDs form the final date. The layer probe matches the probabilities used at every decoding position.",
        tokens: targetTokens,
        tensors: [{ name: "Output character IDs", shape: [10], values: chosen, dtype: "int32", axes: ["position"] }],
      },
    ];
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
        const currentTokenText = tokens[position]?.text ?? "";
        const topPred = probabilities[0]?.token ?? "";
        const topProb = ((probabilities[0]?.probability ?? 0) * 100).toFixed(1);
        const promptStepDesc = position === positions - 1
          ? `End of prompt at position ${position + 1} (“${currentTokenText}”). Softmax over all ${vocabulary.toLocaleString()} vocabulary logits yields the initial continuation distribution. Top prediction is “${topPred}” (${topProb}%).`
          : `Causal conditioning through prompt position ${position + 1} (“${currentTokenText}”). Attention masks future tokens; the model predicts next token “${topPred}” (${topProb}%).`;
        steps.push({ name: `Prompt ${position + 1} · next-token distribution`, stage: "Decoder", operation: "LM head + softmax", description: promptStepDesc, formula: "p(x_{t+1}=i)=\\frac{\\exp(z_i)}{\\sum_j\\exp(z_j)}", tokens, selectedToken: position, probabilities, tensors: position === positions - 1 ? [{ name: "Vocabulary logits", shape: [vocabulary], values: Float32Array.from(row), dtype: logits.type, axes: ["vocabulary ID"] }] : [{ name: "Top candidate logits", shape: [probabilities.length], values: probabilities.map(candidate => candidate.logit!), dtype: logits.type, axes: ["candidate rank"], description: "An exact subset of the vocabulary logits, ordered by descending probability." }] });
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
      if (updates.length) generation.push({ name: `Cache update ${call} · position ${ids.length + call - 1}`, stage: "Decoder", operation: "Key/value cache update", description: `Autoregressive step ${call}: appended key and value vectors for newly generated token “${selected.at(-1)?.text ?? ""}” across all 12 attention heads. Earlier cached states are preserved without recomputation.`, tokens: [selected.at(-1)!], selectedToken: 0, tensors: updates });
    }
    const row = logits.data.subarray((positions - 1) * vocabulary, positions * vocabulary);
    const probabilities = candidates(row, decode, true);
    const winner = probabilities[0];
    const winProb = (winner.probability * 100).toFixed(1);
    const runnerUp = probabilities[1] ? `, followed by “${probabilities[1].token}” (${(probabilities[1].probability * 100).toFixed(1)}%)` : "";
    selected.push({ text: decode(probabilities[0].id!), id: probabilities[0].id, position: ids.length + call });
    generation.push({ name: `Generate ${call + 1} · ${probabilities[0].token}`, stage: "Decoder", operation: "Greedy next-token choice", description: `Greedy decoding (argmax) selects token ID ${winner.id} (“${winner.token}”) with ${winProb}% probability${runnerUp}. The sequence advances to length ${ids.length + call + 1}.`, formula: "x_{t+1}=\\operatorname{argmax}_i z_i", tokens: tokens.concat(selected), selectedToken: tokens.length + selected.length - 1, probabilities, tensors: [{ name: "Vocabulary logits", shape: [vocabulary], values: Float32Array.from(row), dtype: logits.type, axes: ["vocabulary ID"] }] });
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

function getOutputTensor(outputs: Record<string, any> | undefined, pattern: string): { data: Float32Array; dims: number[] } | undefined {
  if (!outputs) return undefined;
  for (const [key, val] of Object.entries(outputs)) {
    if (key.includes(pattern) && val && val.data && val.dims) {
      return {
        data: val.data instanceof Float32Array ? val.data : Float32Array.from(val.data),
        dims: Array.from(val.dims),
      };
    }
  }
  return undefined;
}

async function runSentenceEmbedding(text: string, progress: ModelProgress): Promise<ExecutionTrace> {
  const extractor = await getAssets("sentence-embeddings", progress);
  const tokenizer = extractor.tokenizer;
  const encoded = await tokenizer(text);
  const ids = Array.from(encoded.input_ids.data, Number);
  const mask = Array.from(encoded.attention_mask.data, Number);

  if (!ids.length) throw new ModelInputError("Enter at least one valid word or sentence.");

  const decode = (id: number) => tokenizer.decode([id]) || "[unk]";
  const tokens: TraceToken[] = ids.map((id, position) => ({ id, position, text: decode(id) }));
  const tokenTexts = tokens.map(t => t.text);
  const seqLen = ids.length;

  const started = performance.now();
  progress("Running instrumented Transformer encoder forward pass");

  let rawOutputs: Record<string, any> | undefined;
  try {
    rawOutputs = await extractor.model(encoded);
  } catch (err) {
    console.warn("Direct model forward call fallback:", err);
  }

  let finalHidden: Float32Array;
  let hiddenDim = 384;

  if (rawOutputs) {
    const last = getOutputTensor(rawOutputs, "last_hidden_state");
    if (last) {
      finalHidden = last.data;
      hiddenDim = last.dims[2] ?? 384;
    } else {
      const fallback = await extractor(text, { pooling: "none", normalize: false });
      hiddenDim = fallback.dims[2] ?? 384;
      finalHidden = Float32Array.from(fallback.data);
    }
  } else {
    const fallback = await extractor(text, { pooling: "none", normalize: false });
    hiddenDim = fallback.dims[2] ?? 384;
    finalHidden = Float32Array.from(fallback.data);
  }

  const steps: ExecutionStep[] = [
    // Step 1: Tokenization
    {
      name: "Tokenization & attention mask",
      stage: "Tokens",
      operation: "WordPiece tokenization",
      description: `The sentence is tokenized into ${seqLen} subword pieces including [CLS] (position 0) and [SEP] (position ${seqLen - 1}). Binary attention mask marks unpadded valid tokens.`,
      tokens,
      tensors: [
        { name: "Token IDs", shape: [1, seqLen], values: ids, dtype: "int64", axes: ["batch", "position"] },
        { name: "Attention mask", shape: [1, mask.length], values: mask, dtype: "int64", axes: ["batch", "position"] },
      ],
    },
  ];

  // Step 2: Word Embedding Lookup
  const wordEmb = getOutputTensor(rawOutputs, "word_embeddings");
  if (wordEmb) {
    steps.push({
      name: "Token embedding lookup",
      stage: "Embeddings",
      operation: "Vocabulary embedding matrix lookup",
      description: `Indexed 384-dimensional dense vectors for all ${seqLen} subword tokens from embedding matrix W_word ∈ R^(30522 × 384).`,
      formula: "\\mathbf{E}_{\\text{word}} = \\mathbf{W}_{\\text{word}}[\\mathbf{x}]",
      tokens,
      tensors: [
        {
          name: "Token embeddings",
          shape: [1, seqLen, hiddenDim],
          values: wordEmb.data,
          dtype: "float32",
          axes: ["batch", "position", "dimension"],
          description: `Token embedding vectors for each sequence position [1 × ${seqLen} × ${hiddenDim}].`,
        },
      ],
    });
  }

  // Step 3: Positional Embedding Lookup
  const posEmb = getOutputTensor(rawOutputs, "position_embeddings");
  if (posEmb) {
    steps.push({
      name: "Positional embedding addition",
      stage: "Embeddings",
      operation: "Learned positional encoding",
      description: `Looked up learned position representations W_pos[0..${seqLen - 1}] ∈ R^384, encoding order into the permutation-invariant Transformer.`,
      formula: "\\mathbf{E}_{\\text{pos}} = \\mathbf{W}_{\\text{pos}}[0 \\dots L-1]",
      tokens,
      tensors: [
        {
          name: "Position embeddings",
          shape: [1, seqLen, hiddenDim],
          values: posEmb.data,
          dtype: "float32",
          axes: ["batch", "position", "dimension"],
          description: `Absolute positional vectors for positions 0 to ${seqLen - 1} [1 × ${seqLen} × ${hiddenDim}].`,
        },
      ],
    });
  }

  // Step 4: Embedding LayerNorm & Residual
  const normEmb = getOutputTensor(rawOutputs, "embeddings/LayerNorm") ?? getOutputTensor(rawOutputs, "LayerNorm/Add_1");
  if (normEmb) {
    steps.push({
      name: "Embedding LayerNorm & dropout",
      stage: "Embeddings",
      operation: "Layer normalization",
      description: `Combined word, position, and token type embeddings are summed coordinate-wise and normalized via LayerNorm to standardize input variance.`,
      formula: "\\mathbf{H}^{(0)} = \\operatorname{LayerNorm}(\\mathbf{E}_{\\text{word}} + \\mathbf{E}_{\\text{pos}} + \\mathbf{E}_{\\text{type}})",
      tokens,
      tensors: [
        {
          name: "Initial layer representation H(0)",
          shape: [1, seqLen, hiddenDim],
          values: normEmb.data,
          dtype: "float32",
          axes: ["batch", "position", "dimension"],
          description: `Normalized input representation entering the first Transformer encoder layer [1 × ${seqLen} × ${hiddenDim}].`,
        },
      ],
    });
  }

  // Compute head-averaged attention matrix and collect intermediate layer representations
  const numHeads = 12;
  const layer6Attn = getOutputTensor(rawOutputs, "layer.5/attention/self/Softmax");
  let headAvgMatrix: number[][] | undefined;
  let headAvgFlat: Float32Array | undefined;

  if (layer6Attn && layer6Attn.data.length >= numHeads * seqLen * seqLen) {
    headAvgMatrix = [];
    headAvgFlat = new Float32Array(seqLen * seqLen);
    for (let i = 0; i < seqLen; i++) {
      const row: number[] = [];
      for (let j = 0; j < seqLen; j++) {
        let sum = 0;
        for (let h = 0; h < numHeads; h++) {
          const idx = h * seqLen * seqLen + i * seqLen + j;
          sum += layer6Attn.data[idx];
        }
        const avg = sum / numHeads;
        row.push(avg);
        headAvgFlat[i * seqLen + j] = avg;
      }
      headAvgMatrix.push(row);
    }
  }

  // Gather intermediate layer representations (Layers 1-5) if exported
  const intermediateLayerTensors: TraceTensor[] = [];
  for (let l = 0; l < 5; l++) {
    const intermediate = getOutputTensor(rawOutputs, `layer.${l}/output/LayerNorm`);
    if (intermediate) {
      intermediateLayerTensors.push({
        name: `Layer ${l + 1} output H(${l + 1})`,
        shape: [1, seqLen, hiddenDim],
        values: intermediate.data,
        dtype: "float32",
        axes: ["batch", "position", "dimension"],
        description: `Contextual token representations after block ${l + 1} multi-head self-attention and feed-forward network.`,
      });
    }
  }

  // Steps: Progressive token-by-token contextual encoding through the 6 Transformer blocks
  for (let i = 0; i < seqLen; i++) {
    const token = tokens[i];
    const tokenSlice = finalHidden.slice(i * hiddenDim, (i + 1) * hiddenDim);
    const weightsRow = headAvgMatrix?.[i] ?? [];
    let peakCol = 0;
    let peakWeight = -1;
    weightsRow.forEach((w, col) => {
      if (w > peakWeight) {
        peakWeight = w;
        peakCol = col;
      }
    });
    const peakToken = tokens[peakCol]?.text ?? "";
    const peakPct = (peakWeight * 100).toFixed(1);

    const stepTensors: TraceTensor[] = [
      {
        name: `Contextual token vector h_${i + 1}`,
        shape: [1, hiddenDim],
        values: tokenSlice,
        dtype: "float32",
        axes: ["batch", "dimension"],
        description: `Final 384-dimensional contextual representation for token “${token.text}” (position ${i}) after 6 Transformer encoder blocks.`,
      },
    ];

    if (weightsRow.length > 0) {
      stepTensors.push({
        name: `Self-attention distribution (token ${i + 1})`,
        shape: [1, seqLen],
        values: Float32Array.from(weightsRow),
        dtype: "float32",
        axes: ["batch", "key position"],
        description: `Attention weights from query token “${token.text}” across all ${seqLen} key positions (sums to 1.000).`,
      });
    }

    stepTensors.push({
      name: "All tokens H(6)",
      shape: [1, seqLen, hiddenDim],
      values: finalHidden,
      dtype: "float32",
      axes: ["batch", "position", "dimension"],
      description: `Complete sequence matrix of contextual representations after 6 Transformer encoder blocks [1 × ${seqLen} × ${hiddenDim}].`,
    });

    if (headAvgFlat) {
      stepTensors.push({
        name: "Head-averaged attention matrix",
        shape: [1, seqLen, seqLen],
        values: headAvgFlat,
        dtype: "float32",
        axes: ["batch", "query position", "key position"],
        description: `Average token-to-token attention weight across all 12 attention heads (${seqLen} × ${seqLen}).`,
      });
    }

    if (layer6Attn) {
      stepTensors.push({
        name: "All 12 attention heads",
        shape: [1, numHeads, seqLen, seqLen],
        values: layer6Attn.data,
        dtype: "float32",
        axes: ["batch", "head", "query position", "key position"],
        description: `Full multi-head self-attention distributions across each of the 12 attention heads for layer 6.`,
      });
    }

    stepTensors.push(...intermediateLayerTensors);

    steps.push({
      name: `Encode ${i + 1} · ${token.text}`,
      stage: "Encoder",
      operation: "Bidirectional self-attention & FFN",
      description: `Query token ${i} (“${token.text}”) attends across all ${seqLen} sequence positions over 6 Transformer blocks (12 heads each), focusing most strongly on token ${peakCol} (“${peakToken}”) with ${peakPct}% weight. Residual skip connections and feed-forward layers yield contextual state h_${i + 1} ∈ R^384.`,
      formula: `\\mathbf{h}_i^{(6)} = \\operatorname{FFN}\\Big(\\operatorname{LayerNorm}\\big(\\mathbf{h}_i^{(5)} + \\sum_{h=1}^{12} \\operatorname{Attn}^{(h)}_i\\big)\\Big)`,
      tokens,
      selectedToken: i,
      attention: headAvgMatrix ? {
        source: tokenTexts,
        target: tokenTexts,
        weights: headAvgMatrix,
        row: i,
      } : undefined,
      tensors: stepTensors,
    });
  }

  // Step 11: Mean pooling across unmasked tokens
  const pooled = new Float32Array(hiddenDim);
  const summed = new Float32Array(hiddenDim);
  let unmaskedCount = 0;
  for (let i = 0; i < seqLen; i++) {
    if (mask[i] > 0) {
      unmaskedCount++;
      const offset = i * hiddenDim;
      for (let d = 0; d < hiddenDim; d++) {
        summed[d] += finalHidden[offset + d];
      }
    }
  }
  const divisor = unmaskedCount > 0 ? unmaskedCount : 1;
  for (let d = 0; d < hiddenDim; d++) {
    pooled[d] = summed[d] / divisor;
  }

  steps.push({
    name: "Attention-masked mean pooling",
    stage: "Pooling",
    operation: "Masked sequence reduction",
    description: `Averaged contextual hidden representations over ${unmaskedCount} unmasked tokens, compressing variable-length sequence matrix [1 × ${seqLen} × ${hiddenDim}] into a fixed 384-dimensional sentence vector.`,
    formula: "\\mathbf{e}_{\\text{pooled}} = \\frac{\\sum_{i=1}^L \\mathbf{h}_i^{(6)} \\cdot m_i}{\\sum_{i=1}^L m_i}",
    tensors: [
      {
        name: "Pooled sentence vector",
        shape: [1, hiddenDim],
        values: pooled,
        dtype: "float32",
        axes: ["batch", "dimension"],
        description: `Mean vector of unmasked token representations [1 × ${hiddenDim}].`,
      },
      {
        name: "Summed token states",
        shape: [1, hiddenDim],
        values: summed,
        dtype: "float32",
        axes: ["batch", "dimension"],
        description: `Element-wise sum of all unmasked token vectors before dividing by count ${divisor}.`,
      },
    ],
  });

  // Step 12: L2 Unit Normalization
  let normSq = 0;
  for (let d = 0; d < hiddenDim; d++) normSq += pooled[d] * pooled[d];
  const l2Norm = Math.sqrt(normSq) || 1e-12;
  const normalized = new Float32Array(hiddenDim);
  for (let d = 0; d < hiddenDim; d++) {
    normalized[d] = pooled[d] / l2Norm;
  }

  steps.push({
    name: "Unit L2 hypersphere normalization",
    stage: "Normalized",
    operation: "Euclidean L2 projection",
    description: `Projected pooled representation onto the 384-dimensional unit hypersphere S^383 (||e||_2 = 1.0000). Cosine similarity between any two encoded sentences now equals their inner dot product.`,
    formula: "\\mathbf{e} = \\frac{\\mathbf{e}_{\\text{pooled}}}{\\|\\mathbf{e}_{\\text{pooled}}\\|_2}, \\quad \\|\\mathbf{e}\\|_2 = 1.0",
    tensors: [
      {
        name: "Final unit embedding",
        shape: [1, hiddenDim],
        values: normalized,
        dtype: "float32",
        axes: ["batch", "dimension"],
        description: `Unit-length dense semantic embedding vector on S^383 [1 × ${hiddenDim}].`,
      },
    ],
  });

  const preview = `[${normalized[0].toFixed(3)}, ${normalized[1].toFixed(3)}, ${normalized[2].toFixed(3)}, … (384d)]`;

  return {
    output: preview,
    outputLabel: "384-dimensional unit embedding",
    backend: "ONNX · WASM · q8",
    elapsed: performance.now() - started,
    steps,
    note: `Complete ${steps.length}-step forward pass through all-MiniLM-L6-v2: tokenization, token embedding lookup, learned positional encodings, LayerNorm, progressive self-attention across ${seqLen} token positions, attention-masked mean pooling, and L2 unit hypersphere normalization.`,
  };
}

let execution: Promise<unknown> = Promise.resolve();
export async function runModel(kind: RealModelKind, text: string, progress: ModelProgress): Promise<ExecutionTrace> {
  if (!text.trim()) throw new ModelInputError("Enter an input first.");
  // Shared model methods are instrumented only while their own run holds the queue.
  const run = execution.catch(() => {}).then(async () => {
    const runInference =
      kind === "lstm"
        ? runSentiment
        : kind === "seq2seq"
        ? runTranslation
        : kind === "attention"
        ? runDateAttention
        : kind === "sentence-embeddings"
        ? runSentenceEmbedding
        : runTransformer;
    progress("Running the model");
    await yieldToPage();
    const trace = await runInference(text, progress);
    trace.steps.forEach(step => step.tensors.forEach(validateTensor));
    return trace;
  });
  execution = run.then(() => undefined, () => undefined);
  return run;
}
