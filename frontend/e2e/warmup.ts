// Loads the corpus and the ONNX embedding model once before the parallel workers start, so no test pays the cold start.
export default async function warmup() {
  const base = "http://127.0.0.1:8766";
  for (const path of ["/api/meta", "/api/evals", "/api/copilot/recorded", "/api/search?q=stolen%20card&retriever_name=dense_bge"]) {
    const r = await fetch(base + path);
    if (!r.ok) throw new Error(`warm-up ${path}: ${r.status}`);
  }
}
