mod daemon;

use daemon::DaemonState;
use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Emitter, Manager, WindowEvent,
};
use tauri_plugin_shell::ShellExt;

// ---------------------------------------------------------------------------
// Tauri commands — called from the frontend via invoke()
// ---------------------------------------------------------------------------

#[tauri::command]
async fn daemon_start(app: tauri::AppHandle) -> Result<(), String> {
    daemon::start(&app)
}

#[tauri::command]
async fn daemon_stop(app: tauri::AppHandle) {
    daemon::stop(&app);
}

#[tauri::command]
async fn daemon_alive() -> bool {
    daemon::is_alive().await
}

/// Forward a JSON command to the daemon over WebSocket and return the result.
/// We open a fresh WS connection per call — good enough for infrequent ops.
#[tauri::command]
async fn daemon_command(kind: String, params: serde_json::Value) -> Result<serde_json::Value, String> {
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::Message;
    use futures_util::{SinkExt, StreamExt};

    let url = format!("ws://{}:{}", daemon::DAEMON_HOST, daemon::DAEMON_PORT);
    let (mut ws, _) = connect_async(&url)
        .await
        .map_err(|e| format!("ws connect: {e}"))?;

    let op_id = uuid::Uuid::new_v4().to_string().replace('-', "");
    let cmd = serde_json::json!({
        "type": "command",
        "command": {
            "kind": kind,
            "params": params,
            "op_id": op_id,
            "client_id": "desktop"
        }
    });

    ws.send(Message::Text(cmd.to_string()))
        .await
        .map_err(|e| format!("ws send: {e}"))?;

    // Wait for the ack matching our op_id (ignore events in between).
    while let Some(msg) = ws.next().await {
        let msg = msg.map_err(|e| format!("ws recv: {e}"))?;
        if let Message::Text(raw) = msg {
            let val: serde_json::Value =
                serde_json::from_str(&raw).map_err(|e| format!("json: {e}"))?;
            let t = val.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let resp_id = val.get("op_id").and_then(|v| v.as_str()).unwrap_or("");
            if resp_id == op_id {
                if t == "ack" {
                    return Ok(val.get("result").cloned().unwrap_or(serde_json::Value::Null));
                } else if t == "error" {
                    let msg = val
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or("daemon error");
                    return Err(msg.to_string());
                }
            }
        }
    }
    Err("connection closed before ack".into())
}

/// Open the viser 3D viewer URL in the default browser.
#[tauri::command]
async fn open_viewer(app: tauri::AppHandle) -> Result<(), String> {
    let url = "http://127.0.0.1:8080";  // viser default
    app.shell()
        .open(url, None)
        .map_err(|e| format!("open_viewer: {e}"))
}

/// Minimise / maximise / close — called from the custom titlebar.
#[tauri::command]
fn window_minimize(window: tauri::Window) { let _ = window.minimize(); }
#[tauri::command]
fn window_maximize(window: tauri::Window) {
    if window.is_maximized().unwrap_or(false) {
        let _ = window.unmaximize();
    } else {
        let _ = window.maximize();
    }
}
#[tauri::command]
fn window_close(window: tauri::Window) { let _ = window.close(); }

// ---------------------------------------------------------------------------
// App setup
// ---------------------------------------------------------------------------

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_http::init())
        .manage(DaemonState::new())
        .invoke_handler(tauri::generate_handler![
            daemon_start,
            daemon_stop,
            daemon_alive,
            daemon_command,
            open_viewer,
            window_minimize,
            window_maximize,
            window_close,
        ])
        .setup(|app| {
            // Build system tray
            let show = MenuItem::with_id(app, "show", "Show Syna", true, None::<&str>)?;
            let sep  = tauri::menu::PredefinedMenuItem::separator(app)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &sep, &quit])?;

            TrayIconBuilder::with_id("syna-tray")
                .menu(&menu)
                .tooltip("Auroch Syna")
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                    "quit" => {
                        daemon::stop(app);
                        app.exit(0);
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        if let Some(w) = app.get_webview_window("main") {
                            if w.is_visible().unwrap_or(false) {
                                let _ = w.hide();
                            } else {
                                let _ = w.show();
                                let _ = w.set_focus();
                            }
                        }
                    }
                })
                .build(app)?;

            // Auto-start daemon
            if let Err(e) = daemon::start(app.handle()) {
                eprintln!("[syna] daemon start failed: {e}");
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            // Hide to tray instead of quitting on close.
            if let WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "main" {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error running Auroch Syna");
}
