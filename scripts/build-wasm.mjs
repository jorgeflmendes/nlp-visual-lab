import { cp, mkdir } from "node:fs/promises";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const cargo = process.platform === "win32" ? "cargo.exe" : "cargo";

await new Promise((resolve, reject) => {
  const child = spawn(cargo, ["build", "--manifest-path", "wasm-core/Cargo.toml", "--target", "wasm32-unknown-unknown", "--release"], {
    cwd: root,
    stdio: "inherit",
  });
  child.on("error", reject);
  child.on("exit", (code) => code === 0 ? resolve() : reject(new Error(`cargo exited with ${code}`)));
});

await mkdir(path.join(root, "public", "wasm"), { recursive: true });
await cp(
  path.join(root, "wasm-core", "target", "wasm32-unknown-unknown", "release", "nlp_core.wasm"),
  path.join(root, "public", "wasm", "nlp_core.wasm"),
);
