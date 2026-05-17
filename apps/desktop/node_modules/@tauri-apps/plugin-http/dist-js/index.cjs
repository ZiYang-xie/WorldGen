'use strict';

var core = require('@tauri-apps/api/core');

// Copyright 2019-2023 Tauri Programme within The Commons Conservancy
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: MIT
/**
 * Make HTTP requests with the Rust backend.
 *
 * ## Security
 *
 * This API has a scope configuration that forces you to restrict the URLs that can be accessed using glob patterns.
 *
 * For instance, this scope configuration only allows making HTTP requests to all subdomains for `tauri.app` except for `https://private.tauri.app`:
 * ```json
 * {
 *   "permissions": [
 *     {
 *       "identifier": "http:default",
 *       "allow": [{ "url": "https://*.tauri.app" }],
 *       "deny": [{ "url": "https://private.tauri.app" }]
 *     }
 *   ]
 * }
 * ```
 * Trying to execute any API with a URL not configured on the scope results in a promise rejection due to denied access.
 *
 * @module
 */
const ERROR_REQUEST_CANCELLED = 'Request cancelled';
/**
 * Fetch a resource from the network. It returns a `Promise` that resolves to the
 * `Response` to that `Request`, whether it is successful or not.
 *
 * @example
 * ```typescript
 * const response = await fetch("http://my.json.host/data.json");
 * console.log(response.status);  // e.g. 200
 * console.log(response.statusText); // e.g. "OK"
 * const jsonData = await response.json();
 * ```
 *
 * @since 2.0.0
 */
async function fetch(input, init) {
    // Optimistically check for abort signal and avoid doing any work
    const signal = init?.signal;
    if (signal?.aborted) {
        throw new Error(ERROR_REQUEST_CANCELLED);
    }
    const maxRedirections = init?.maxRedirections;
    const connectTimeout = init?.connectTimeout;
    const proxy = init?.proxy;
    const danger = init?.danger;
    // Remove these fields before creating the request
    if (init) {
        delete init.maxRedirections;
        delete init.connectTimeout;
        delete init.proxy;
        delete init.danger;
    }
    const headers = init?.headers
        ? init.headers instanceof Headers
            ? init.headers
            : new Headers(init.headers)
        : new Headers();
    const req = new Request(input, init);
    const buffer = await req.arrayBuffer();
    const data = buffer.byteLength !== 0 ? Array.from(new Uint8Array(buffer)) : null;
    // append new headers created by the browser `Request` implementation,
    // if not already declared by the caller of this function
    for (const [key, value] of req.headers) {
        if (!headers.get(key)) {
            headers.set(key, value);
        }
    }
    const headersArray = headers instanceof Headers
        ? Array.from(headers.entries())
        : Array.isArray(headers)
            ? headers
            : Object.entries(headers);
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    const mappedHeaders = headersArray.map(([name, val]) => [
        name,
        // we need to ensure we have all header values as strings
        // eslint-disable-next-line
        typeof val === 'string' ? val : val.toString()
    ]);
    // Optimistically check for abort signal and avoid doing any work on the Rust side
    if (signal?.aborted) {
        throw new Error(ERROR_REQUEST_CANCELLED);
    }
    const rid = await core.invoke('plugin:http|fetch', {
        clientConfig: {
            method: req.method,
            url: req.url,
            headers: mappedHeaders,
            data,
            maxRedirections,
            connectTimeout,
            proxy,
            danger
        }
    });
    const abort = () => core.invoke('plugin:http|fetch_cancel', { rid });
    // Optimistically check for abort signal
    // and avoid doing any work after doing intial work on the Rust side
    if (signal?.aborted) {
        // we don't care about the result of this proimse
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        abort();
        throw new Error(ERROR_REQUEST_CANCELLED);
    }
    signal?.addEventListener('abort', () => void abort());
    const { status, statusText, url, headers: responseHeaders, rid: responseRid } = await core.invoke('plugin:http|fetch_send', {
        rid
    });
    const dropBody = () => {
        return core.invoke('plugin:http|fetch_cancel_body', { rid: responseRid });
    };
    const readChunk = async (controller) => {
        let data;
        try {
            data = await core.invoke('plugin:http|fetch_read_body', {
                rid: responseRid
            });
        }
        catch (e) {
            // close the stream if an error occurs
            // and drop the body on Rust side
            controller.error(e);
            void dropBody();
            return;
        }
        const dataUint8 = new Uint8Array(data);
        const lastByte = dataUint8[dataUint8.byteLength - 1];
        const actualData = dataUint8.slice(0, dataUint8.byteLength - 1);
        // close when the signal to close (last byte is 1) is sent from the IPC.
        if (lastByte === 1) {
            controller.close();
            return;
        }
        controller.enqueue(actualData);
    };
    // no body for 101, 103, 204, 205 and 304
    // see https://fetch.spec.whatwg.org/#null-body-status
    const body = [101, 103, 204, 205, 304].includes(status)
        ? null
        : new ReadableStream({
            start: (controller) => {
                // listen for abort events to cancel reading
                signal?.addEventListener('abort', () => {
                    controller.error(ERROR_REQUEST_CANCELLED);
                    void dropBody();
                });
            },
            pull: (controller) => readChunk(controller),
            cancel: () => {
                // Ensure body resources are released on stream cancellation
                void dropBody();
            }
        });
    const res = new Response(body, {
        status,
        statusText
    });
    // `Response.url` cannot be set via the constructor, so we define it manually
    Object.defineProperty(res, 'url', { value: url, writable: false });
    // Expose `set-cookie` via `response.headers` (and `getSetCookie()` where
    // supported). This is not Fetch-spec compliant for network responses in
    // browsers, where `set-cookie` is treated as a forbidden response
    // header and is generally not readable from JavaScript.
    Object.defineProperty(res, 'headers', {
        value: new Headers(responseHeaders),
        writable: false
    });
    // Patch clone() per-instance so cloning preserves the overridden properties
    const originalClone = res.clone.bind(res);
    Object.defineProperty(res, 'clone', {
        value: () => {
            const cloned = originalClone();
            Object.defineProperty(cloned, 'url', { value: url, writable: false });
            Object.defineProperty(cloned, 'headers', {
                value: new Headers(responseHeaders),
                writable: false
            });
            return cloned;
        }
    });
    return res;
}

exports.fetch = fetch;
