import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";

const VISER_URL = "http://localhost:8080";

interface Props {
  daemonRunning: boolean;
}

export function SceneViewer({ daemonRunning }: Props) {
  const [loaded, setLoaded] = useState(false);

  if (!daemonRunning) {
    return (
      <div className="scene-viewer">
        <div className="scene-placeholder">
          <div className="orb" />
          <div style={{ fontSize: 13 }}>Start the engine to view scenes</div>
          <div style={{ fontSize: 11 }}>Text → Scene or Image → Scene</div>
        </div>
      </div>
    );
  }

  return (
    <div className="scene-viewer">
      {!loaded && (
        <div className="scene-placeholder" style={{ position: "absolute" }}>
          <div className="orb" />
          <div style={{ fontSize: 13 }}>Loading viewer…</div>
        </div>
      )}
      <iframe
        src={VISER_URL}
        onLoad={() => setLoaded(true)}
        style={{ opacity: loaded ? 1 : 0, transition: "opacity 0.3s" }}
        allow="fullscreen"
        title="3D Scene Viewer"
      />
      <button
        className="btn secondary"
        style={{
          position: "absolute", bottom: 12, right: 12,
          fontSize: 11, padding: "5px 10px", opacity: 0.7,
        }}
        onClick={() => invoke("open_viewer")}
      >
        Open in window ↗
      </button>
    </div>
  );
}
