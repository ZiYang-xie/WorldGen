import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

export type DaemonStatus = "starting" | "running" | "stopped" | "error";

export interface DaemonHook {
  status: DaemonStatus;
  start: () => Promise<void>;
  stop: () => Promise<void>;
  command: (kind: string, params?: Record<string, unknown>) => Promise<unknown>;
  lastError: string | null;
}

export function useDaemon(): DaemonHook {
  const [status, setStatus] = useState<DaemonStatus>("starting");
  const [lastError, setLastError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const poll = useCallback(async () => {
    try {
      const alive = await invoke<boolean>("daemon_alive");
      setStatus(alive ? "running" : "stopped");
    } catch {
      setStatus("error");
    }
  }, []);

  const start = useCallback(async () => {
    setStatus("starting");
    setLastError(null);
    try {
      await invoke("daemon_start");
      // Give the process 1.5 s to bind the port before polling
      await new Promise((r) => setTimeout(r, 1500));
      await poll();
    } catch (e) {
      setLastError(String(e));
      setStatus("error");
    }
  }, [poll]);

  const stop = useCallback(async () => {
    await invoke("daemon_stop");
    setStatus("stopped");
  }, []);

  const command = useCallback(
    async (kind: string, params: Record<string, unknown> = {}) => {
      if (status !== "running") throw new Error("Daemon is not running");
      return invoke("daemon_command", { kind, params });
    },
    [status]
  );

  useEffect(() => {
    // Initial start attempt
    start();

    // Poll every 5 s
    pollRef.current = setInterval(poll, 5000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return { status, start, stop, command, lastError };
}
