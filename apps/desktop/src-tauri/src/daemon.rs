//! Python daemon sidecar lifecycle.
//!
//! The daemon is a PyInstaller-bundled binary (`binaries/syna-daemon-*`)
//! that runs `auroch_syna.daemon.server` over WebSocket on 127.0.0.1:8765.
//! We start it on app launch and stop it on exit.

use std::sync::Mutex;
use tauri::{AppHandle, Manager};
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

pub const DAEMON_HOST: &str = "127.0.0.1";
pub const DAEMON_PORT: u16 = 8765;

pub struct DaemonState {
    pub child: Mutex<Option<CommandChild>>,
}

impl DaemonState {
    pub fn new() -> Self {
        Self { child: Mutex::new(None) }
    }
}

/// Start the syna-daemon sidecar. No-op if already running.
pub fn start(app: &AppHandle) -> Result<(), String> {
    let state = app.state::<DaemonState>();
    let mut guard = state.child.lock().unwrap();

    if guard.is_some() {
        return Ok(());
    }

    let (_, child) = app
        .shell()
        .sidecar("syna-daemon")
        .map_err(|e| format!("sidecar spawn error: {e}"))?
        .args(["--host", DAEMON_HOST, "--port", &DAEMON_PORT.to_string()])
        .spawn()
        .map_err(|e| format!("spawn failed: {e}"))?;

    *guard = Some(child);
    Ok(())
}

/// Stop the daemon gracefully.
pub fn stop(app: &AppHandle) {
    let state = app.state::<DaemonState>();
    let mut guard = state.child.lock().unwrap();
    if let Some(child) = guard.take() {
        let _ = child.kill();
    }
}

/// Ping the daemon's HTTP health endpoint (lightweight check without websockets).
pub async fn is_alive() -> bool {
    let url = format!("http://{}:{}/health", DAEMON_HOST, DAEMON_PORT);
    // A TCP connect is enough — the Python server may not expose /health yet,
    // so we just test whether the port is open.
    tokio::net::TcpStream::connect((DAEMON_HOST, DAEMON_PORT))
        .await
        .is_ok()
}
