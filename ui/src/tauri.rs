//! Thin bridge to the Tauri JS API (requires `withGlobalTauri`).

use leptos::task::spawn_local;
use serde::de::DeserializeOwned;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(catch, js_namespace = ["window", "__TAURI__", "core"], js_name = invoke)]
    async fn invoke_raw(cmd: &str, args: JsValue) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"], js_name = convertFileSrc)]
    fn convert_file_src_raw(path: &str) -> String;

    #[wasm_bindgen(catch, js_namespace = ["window", "__TAURI__", "event"], js_name = listen)]
    async fn listen_raw(event: &str, handler: &JsValue) -> Result<JsValue, JsValue>;
}

pub fn in_tauri() -> bool {
    web_sys::window()
        .and_then(|w| js_sys::Reflect::get(&w, &"__TAURI__".into()).ok())
        .map(|v| !v.is_undefined())
        .unwrap_or(false)
}

fn js_to_string(v: &JsValue) -> String {
    if let Some(s) = v.as_string() {
        return s;
    }
    js_sys::JSON::stringify(v)
        .map(String::from)
        .unwrap_or_else(|_| "unknown error".into())
}

/// Invoke a backend command. Arguments are top-level camelCase keys.
pub async fn call<R: DeserializeOwned>(cmd: &str, args: serde_json::Value) -> Result<R, String> {
    if !in_tauri() {
        return Err("Run Trackwright through the desktop app (cargo tauri dev) to use this".into());
    }
    let js_args = js_sys::JSON::parse(&args.to_string()).map_err(|e| js_to_string(&e))?;
    match invoke_raw(cmd, js_args).await {
        Ok(v) => {
            let s = if v.is_undefined() || v.is_null() {
                "null".to_string()
            } else {
                js_sys::JSON::stringify(&v)
                    .map(String::from)
                    .unwrap_or_else(|_| "null".into())
            };
            serde_json::from_str(&s).map_err(|e| format!("Unexpected reply from {cmd}: {e}"))
        }
        Err(e) => Err(js_to_string(&e)),
    }
}

pub fn convert_file_src(path: &str) -> String {
    if in_tauri() {
        convert_file_src_raw(path)
    } else {
        path.to_string()
    }
}

/// Subscribe to a backend event for the lifetime of the app.
pub fn listen<T: DeserializeOwned + 'static>(event: &'static str, mut f: impl FnMut(T) + 'static) {
    if !in_tauri() {
        return;
    }
    let cb = Closure::<dyn FnMut(JsValue)>::new(move |ev: JsValue| {
        let payload = js_sys::Reflect::get(&ev, &"payload".into()).unwrap_or(JsValue::NULL);
        let s = js_sys::JSON::stringify(&payload)
            .map(String::from)
            .unwrap_or_default();
        match serde_json::from_str::<T>(&s) {
            Ok(v) => f(v),
            Err(e) => web_sys::console::error_1(&format!("bad {event} payload: {e}").into()),
        }
    });
    let js = cb.into_js_value();
    spawn_local(async move {
        let _ = listen_raw(event, &js).await;
    });
}
