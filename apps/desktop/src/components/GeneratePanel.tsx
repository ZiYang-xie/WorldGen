import { useState } from "react";
import type { DaemonHook } from "../hooks/useDaemon";
import { invoke } from "@tauri-apps/api/core";

interface Props {
  daemon: DaemonHook;
  onResult: (msg: string) => void;
}

export function GeneratePanel({ daemon, onResult }: Props) {
  const [prompt, setPrompt] = useState("");
  const [mode, setMode]     = useState<"t2s" | "i2s">("t2s");
  const [mesh, setMesh]     = useState(false);
  const [busy, setBusy]     = useState(false);

  async function generate() {
    if (!prompt.trim() || busy) return;
    setBusy(true);
    onResult("Sending generate_scene command…");
    try {
      const result = await daemon.command("generate_scene", {
        prompt, mode, return_mesh: mesh,
      }) as Record<string, unknown> | null;
      const count = result?.object_count ?? "?";
      onResult(`Done — scene generated (${count} objects). Open viewer to inspect.`);
    } catch (e) {
      onResult(`Error: ${e}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card generate-panel">
      <div className="card-title">Generate Scene</div>

      {/* Mode toggle */}
      <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
        {(["t2s", "i2s"] as const).map((m) => (
          <button key={m} className="btn secondary"
            style={{ flex: 1, fontSize: 12,
              background: mode === m ? "var(--syna-accent-dim)" : undefined,
              borderColor: mode === m ? "var(--syna-accent)" : undefined }}
            onClick={() => setMode(m)}>
            {m === "t2s" ? "Text → Scene" : "Image → Scene"}
          </button>
        ))}
      </div>

      <label>Prompt</label>
      <textarea
        rows={4}
        placeholder="A sun-lit Japanese garden with stone lanterns…"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && e.metaKey) generate(); }}
      />

      <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
        <input type="checkbox" checked={mesh} onChange={(e) => setMesh(e.target.checked)} />
        Output mesh (GLB) instead of splats
      </label>

      <button className="btn primary" onClick={generate}
        disabled={daemon.status !== "running" || busy || !prompt.trim()}>
        {busy ? "Generating…" : "Generate  ⌘↵"}
      </button>

      <button className="btn secondary" style={{ marginTop: 2 }}
        onClick={() => invoke("open_viewer")}
        disabled={daemon.status !== "running"}>
        Open 3D Viewer ↗
      </button>
    </div>
  );
}
