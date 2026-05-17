/**
 * Exits immediately with the given `exitCode`.
 * @example
 * ```typescript
 * import { exit } from '@tauri-apps/plugin-process';
 * await exit(1);
 * ```
 *
 * @param code The exit code to use.
 * @returns A promise indicating the success or failure of the operation.
 *
 * @since 2.0.0
 */
declare function exit(code?: number): Promise<void>;
/**
 * Exits the current instance of the app then relaunches it.
 * @example
 * ```typescript
 * import { relaunch } from '@tauri-apps/plugin-process';
 * await relaunch();
 * ```
 *
 * @returns A promise indicating the success or failure of the operation.
 *
 * @since 2.0.0
 */
declare function relaunch(): Promise<void>;
export { exit, relaunch };
