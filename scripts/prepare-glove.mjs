import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import readline from "node:readline";
import https from "node:https";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const outDir = path.join(root, "public", "models");
fs.mkdirSync(outDir, { recursive: true });

const GLOVE_URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz";
const TARGET_WORDS = 20000;
const DIM = 100;

console.log("Streaming GloVe 100d from release archive...");

function fetchUrl(url) {
  https.get(url, { headers: { "User-Agent": "Mozilla/5.0" } }, (res) => {
    if (res.statusCode === 301 || res.statusCode === 302) {
      console.log("Following redirect to:", res.headers.location);
      fetchUrl(res.headers.location);
      return;
    }
    if (res.statusCode !== 200) {
      console.error(`Download failed with status ${res.statusCode}`);
      process.exit(1);
    }
    handleStream(res);
  }).on("error", (err) => {
    console.error("Network error:", err);
    process.exit(1);
  });
}

function handleStream(stream) {
  const gunzip = zlib.createGunzip();
  const rl = readline.createInterface({ input: stream.pipe(gunzip), crlfDelay: Infinity });

  const vocab = [];
  const buffer = new Float32Array(TARGET_WORDS * DIM);
  let count = 0;
  let isFirstLine = true;

  rl.on("line", (line) => {
    if (count >= TARGET_WORDS) {
      rl.close();
      stream.destroy();
      return;
    }

    // gensim files might have header line "400000 100"
    if (isFirstLine) {
      isFirstLine = false;
      const tokens = line.trim().split(/\s+/);
      if (tokens.length === 2 && !isNaN(Number(tokens[0])) && !isNaN(Number(tokens[1]))) {
        return;
      }
    }

    const parts = line.trim().split(/\s+/);
    if (parts.length !== DIM + 1) return;

    const word = parts[0];
    let normSq = 0;
    const vec = new Float32Array(DIM);
    for (let i = 0; i < DIM; i++) {
      const val = parseFloat(parts[i + 1]);
      vec[i] = val;
      normSq += val * val;
    }

    const norm = Math.sqrt(normSq) || 1e-12;
    const offset = count * DIM;
    for (let i = 0; i < DIM; i++) {
      buffer[offset + i] = vec[i] / norm;
    }
    vocab.push(word);
    count++;

    if (count % 5000 === 0) {
      console.log(`Processed ${count}/${TARGET_WORDS} words...`);
    }
  });

  rl.on("close", () => {
    console.log(`Extracted ${count} words.`);
    const vocabPath = path.join(outDir, "glove-20k-vocab.json");
    const binPath = path.join(outDir, "glove-20k-100d.bin");

    fs.writeFileSync(vocabPath, JSON.stringify(vocab));
    fs.writeFileSync(binPath, Buffer.from(buffer.buffer));

    console.log(`Wrote vocab to ${vocabPath} (${fs.statSync(vocabPath).size} bytes)`);
    console.log(`Wrote vectors to ${binPath} (${fs.statSync(binPath).size} bytes)`);
  });
}

fetchUrl(GLOVE_URL);
