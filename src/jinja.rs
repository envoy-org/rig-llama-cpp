//! Jinja chat-template rendering, used as a fallback for models llama.cpp
//! cannot template itself.
//!
//! `llama_chat_apply_template` is not a jinja engine. It sniffs the template
//! string for a handful of well-known shapes (ChatML, Llama-3, Mistral, …) and
//! returns `-1` for anything it does not recognise. It also accepts only
//! `(role, content)` pairs, so template variables like `enable_thinking` and
//! structured `tools` have nowhere to go — `llama-cpp-2` 0.1.147 removed the
//! `chat_template_kwargs` plumbing that used to forward them.
//!
//! Models shipping a bespoke template therefore fail that path outright and,
//! before this module existed, were silently prompted with ChatML instead —
//! a format they were never trained on. Gemma-4 is one: its template renders
//! `<|turn>role` turns and gates reasoning behind `enable_thinking`.
//!
//! Rendering the template ourselves recovers both the correct prompt format
//! and the variables the C API cannot forward. This is strictly a *fallback*:
//! llama.cpp's applier remains the primary path, so every model it already
//! handles is byte-for-byte unchanged.

use minijinja::value::Value as JValue;
use minijinja::{Environment, Error, ErrorKind, State};
use serde_json::Value;

/// Python-style methods that real chat templates call but minijinja does not
/// implement natively. Upstream ships a broad set in `minijinja-contrib`'s
/// `pycompat` module; templates in practice need only these two, so we
/// implement them here rather than take a second dependency.
fn unknown_method(
    _state: &State,
    value: &JValue,
    method: &str,
    args: &[JValue],
) -> Result<JValue, Error> {
    match method {
        // `message.get('tool_calls')` / `.get(key, default)`. Jinja2 returns
        // the default (None when omitted) for a missing key rather than
        // raising, which templates rely on to probe optional fields.
        "get" => {
            let key = args
                .first()
                .ok_or_else(|| Error::new(ErrorKind::MissingArgument, "get() requires a key"))?;
            let default = args.get(1).cloned().unwrap_or_else(|| JValue::from(()));
            match value.get_item(key) {
                Ok(found) if !found.is_undefined() => Ok(found),
                _ => Ok(default),
            }
        }
        // `text.split('<channel|>')`, and the no-argument whitespace form.
        "split" => {
            let text = value.as_str().ok_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, "split() called on a non-string")
            })?;
            let parts: Vec<JValue> = match args.first().and_then(|a| a.as_str()) {
                Some(sep) => text.split(sep).map(JValue::from).collect(),
                None => text.split_whitespace().map(JValue::from).collect(),
            };
            Ok(JValue::from(parts))
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

/// Variables a chat template expects alongside the message list.
pub(crate) struct TemplateVars<'a> {
    /// Forwarded from the request's `additional_params` (`{"thinking": bool}`).
    pub enable_thinking: bool,
    /// The model's BOS piece. Templates emit it explicitly via `bos_token`;
    /// tokenization adds BOS separately, so an empty string here avoids
    /// doubling it.
    pub bos_token: &'a str,
}

/// Render `template` with minijinja.
///
/// `messages` are the same flattened `{role, content}` pairs the llama.cpp and
/// ChatML paths use, deliberately *not* the raw structured messages. Feeding a
/// template its native structures sounds better and is worse in practice:
///
/// - **Tools.** Handing over a `tools` array makes the template emit its own
///   tool dialect, and the model answers in kind — Gemma-4's is
///   `<|tool_call>call:name{k:v}`, a bespoke DSL, not JSON. `parse_tool_calls`
///   reads one portable `<tool_call>` JSON protocol, so a native dialect
///   silently yields no tool calls at all. The directive that `build_prompt`
///   injects into the system message keeps every model on that one protocol,
///   exactly as the other two paths do. `tools` is therefore passed as none.
/// - **Media.** Image content arrives as a `media_marker` part whose text is
///   llama.cpp's marker; flattening inlines it so markers and bitmaps stay
///   balanced. A template iterating raw content parts drops the type it does
///   not recognise, and mtmd then rejects the prompt for having fewer markers
///   than bitmaps.
///
/// So this path changes exactly two things versus ChatML: the turn format, and
/// the template variables llama.cpp's C API cannot forward.
pub(crate) fn render_chat_template(
    template: &str,
    messages: &[Value],
    vars: &TemplateVars<'_>,
) -> Result<String, Error> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(unknown_method);
    // Chat templates are whitespace-sensitive: they control spacing with
    // explicit `{%-` / `-%}` markers and rely on everything else surviving
    // verbatim. Keep minijinja from trimming on its own.
    env.set_keep_trailing_newline(true);
    env.add_template("chat", template)?;
    let tmpl = env.get_template("chat")?;

    tmpl.render(minijinja::context! {
        messages => JValue::from_serialize(messages),
        // Explicitly none rather than undefined: templates branch on `tools`
        // in boolean context, and none is unambiguously falsy. See above for
        // why tools are not passed natively.
        tools => JValue::from(()),
        enable_thinking => vars.enable_thinking,
        // Templates that render reasoning back into history gate it on this;
        // leave it off so prior turns' thoughts are not replayed.
        preserve_thinking => false,
        add_generation_prompt => true,
        bos_token => vars.bos_token,
        eos_token => "",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msgs() -> Vec<Value> {
        vec![
            json!({"role": "system", "content": "Be brief."}),
            json!({"role": "user", "content": "Hi"}),
        ]
    }

    fn vars<'a>(thinking: bool) -> TemplateVars<'a> {
        TemplateVars {
            enable_thinking: thinking,
            bos_token: "",
        }
    }

    #[test]
    fn renders_a_chatml_style_template() {
        // Non-trimming `{% endfor %}` / `{% endif %}` so the newlines the
        // template emits survive — chat templates are whitespace-sensitive.
        let tmpl = "{%- for m in messages -%}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{%- if add_generation_prompt -%}<|im_start|>assistant\n{% endif %}";
        let out = render_chat_template(tmpl, &msgs(), &vars(false)).unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn enable_thinking_reaches_the_template() {
        let tmpl = "{%- set enable_thinking = enable_thinking | default(false) -%}\
                    {%- if enable_thinking -%}<|think|>{%- endif -%}done";
        let on = render_chat_template(tmpl, &msgs(), &vars(true)).unwrap();
        let off = render_chat_template(tmpl, &msgs(), &vars(false)).unwrap();
        assert_eq!(on, "<|think|>done");
        assert_eq!(off, "done");
    }

    #[test]
    fn tools_are_none_so_templates_do_not_emit_a_native_tool_dialect() {
        // Tools reach the model through the portable `<tool_call>` directive
        // injected into the system message, not the template's own dialect.
        // `tools` must be falsy so the template's tool block stays closed.
        let tmpl = "{%- if tools -%}native{%- else -%}absent{%- endif -%}";
        let out = render_chat_template(tmpl, &msgs(), &vars(false)).unwrap();
        assert_eq!(out, "absent");
    }

    #[test]
    fn get_method_returns_default_for_missing_key() {
        // The `.get()` pycompat shim: present key, missing key, explicit default.
        let tmpl = "{{ messages[0].get('role') }}|{{ messages[0].get('nope') }}|{{ messages[0].get('nope', 'fb') }}";
        let out = render_chat_template(tmpl, &msgs(), &vars(false)).unwrap();
        assert_eq!(out, "system|none|fb");
    }

    #[test]
    fn split_method_splits_on_a_separator() {
        let tmpl = "{%- for p in 'a<channel|>b'.split('<channel|>') -%}[{{ p }}]{%- endfor -%}";
        let out = render_chat_template(tmpl, &msgs(), &vars(false)).unwrap();
        assert_eq!(out, "[a][b]");
    }

    #[test]
    fn supports_macros_namespaces_and_dictsort() {
        // The constructs Gemma-4's template leans on hardest.
        let tmpl = "{%- macro emit(k, v) -%}{{ k }}={{ v }};{%- endmacro -%}\
                    {%- set ns = namespace(n=0) -%}\
                    {%- for k, v in {'b': 2, 'a': 1} | dictsort -%}\
                    {%- set ns.n = ns.n + 1 -%}{{ emit(k, v) }}\
                    {%- endfor -%}count={{ ns.n }}";
        let out = render_chat_template(tmpl, &msgs(), &vars(false)).unwrap();
        assert_eq!(out, "a=1;b=2;count=2");
    }

    #[test]
    fn reports_an_error_for_a_broken_template() {
        assert!(render_chat_template("{% for %}", &msgs(), &vars(false)).is_err());
    }
}
