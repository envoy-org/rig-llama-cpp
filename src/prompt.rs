use serde_json::Value;

use crate::types::{PreparedRequest, PromptBuildResult};

/// Render a [`PreparedRequest`] into a prompt string.
///
/// Three paths, in order:
///
/// 1. **llama.cpp's applier** — the fast path, used whenever it recognises the
///    template. Every model it already handles renders exactly as before.
/// 2. **minijinja** ([`crate::jinja`]) — for models whose bespoke jinja
///    template llama.cpp cannot apply. This path additionally forwards
///    `enable_thinking`, which the C API has no way to pass.
/// 3. **ChatML** — last resort, for a model with no usable template at all.
///
/// All three paths receive the same flattened messages, so they differ only in
/// the turn format they wrap them in.
///
/// `llama-cpp-2` 0.1.147 removed the `apply_chat_template_oaicompat` /
/// `OpenAIChatTemplateParams` path that previously let llama.cpp's jinja
/// engine ingest `tools` / `tool_choice` / `json_schema` directly. The
/// remaining `apply_chat_template` only takes `(role, content)` messages, so
/// tool schemas are injected into the system message here on every path — see
/// [`crate::jinja::render_chat_template`] for why path 2 does not hand the
/// template its native `tools` instead. JSON-schema constraints are applied as
/// a sampler (see `src/sampling.rs`) rather than baked into the prompt.
pub(crate) fn build_prompt(
    model: &llama_cpp_2::model::LlamaModel,
    request: &PreparedRequest,
) -> Result<PromptBuildResult, String> {
    use llama_cpp_2::model::LlamaChatMessage;

    let raw_messages: Vec<Value> = serde_json::from_str(&request.messages_json)
        .map_err(|e| format!("Message deserialization failed: {e}"))?;

    let parsed_messages: Vec<(String, String)> = raw_messages
        .iter()
        .filter_map(|msg| {
            Some((
                msg.get("role")?.as_str()?.to_string(),
                message_content_as_text(msg),
            ))
        })
        .collect();

    // Fold tool schemas into the leading system message when present. The
    // oaicompat path used to hand these to the jinja engine out-of-band; that
    // API is gone, so system-prompt injection is the portable replacement.
    let rendered_messages: Vec<(String, String)> = request
        .tools_json
        .as_deref()
        .map(|tools_json| {
            inject_tools_into_system(&parsed_messages, tools_json, request.tool_choice.as_deref())
        })
        .unwrap_or(parsed_messages.clone());

    let template = model.chat_template(None).ok();

    // Path 1: llama.cpp's built-in applier.
    let applied = template.as_ref().and_then(|tmpl| {
        let chat_msgs: Vec<LlamaChatMessage> = rendered_messages
            .iter()
            .map(|(role, content)| LlamaChatMessage::new(role.clone(), content.clone()))
            .collect::<Result<_, _>>()
            .ok()?;
        match model.apply_chat_template(tmpl, &chat_msgs, true) {
            Ok(prompt) => Some(prompt),
            Err(e) => {
                // `apply_chat_template` is llama.cpp's *non-jinja* applier: it
                // recognises a fixed set of known template shapes and returns
                // -1 for anything else. Models with a bespoke jinja template
                // (Gemma-4's `<|turn>role` format, for one) land here and fall
                // through to minijinja below.
                log::debug!("llama.cpp could not apply the chat template ({e}); trying minijinja");
                None
            }
        }
    });

    if let Some(prompt) = applied {
        log::debug!("messages_json: {}", request.messages_json);
        log::debug!("rendered prompt:\n{prompt}");
        return Ok(PromptBuildResult { prompt });
    }

    // Path 2: render the model's real jinja template ourselves, using the same
    // flattened messages as path 1 so only the turn format and the template
    // variables change.
    if let Some(tmpl) = template.as_ref() {
        match render_with_jinja(tmpl, &rendered_messages, request) {
            Ok(prompt) => {
                log::debug!("rendered prompt (minijinja):\n{prompt}");
                return Ok(PromptBuildResult { prompt });
            }
            Err(e) => log::warn!(
                "model ships a chat template neither llama.cpp nor minijinja could render \
                 ({e}); falling back to ChatML. Output quality will suffer if the model does \
                 not use a ChatML-compatible format."
            ),
        }
    } else {
        log::warn!("model ships no chat template; falling back to ChatML.");
    }

    // Fallback to ChatML.
    let mut prompt = String::new();
    for (role, content) in &rendered_messages {
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    Ok(PromptBuildResult { prompt })
}

/// Render the model's own jinja chat template, supplying the variables the
/// llama.cpp C API has no way to forward.
///
/// `messages` are the same flattened, tool-directive-injected pairs the other
/// two paths use — see [`crate::jinja::render_chat_template`] for why the raw
/// structured messages are deliberately not handed to the template.
///
/// The BOS piece is deliberately passed as an empty string: templates emit
/// `bos_token` explicitly, but tokenization already prepends BOS
/// (`AddBos::Always` in `worker.rs`), and emitting it here too would duplicate
/// it at position 0.
fn render_with_jinja(
    template: &llama_cpp_2::model::LlamaChatTemplate,
    messages: &[(String, String)],
    request: &PreparedRequest,
) -> Result<String, String> {
    let template = template
        .to_str()
        .map_err(|e| format!("chat template is not valid UTF-8: {e}"))?;

    let messages: Vec<Value> = messages
        .iter()
        .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
        .collect();

    crate::jinja::render_chat_template(
        template,
        &messages,
        &crate::jinja::TemplateVars {
            enable_thinking: request.enable_thinking,
            bos_token: "",
        },
    )
    .map_err(|e| format!("{e:#}"))
}

/// Merge a tool-schema description into the first system message (creating one
/// if none exists), returning a new `(role, content)` vec.
fn inject_tools_into_system(
    messages: &[(String, String)],
    tools_json: &str,
    tool_choice: Option<&str>,
) -> Vec<(String, String)> {
    let tool_directive = build_tool_directive(tools_json, tool_choice);
    let mut out: Vec<(String, String)> = Vec::with_capacity(messages.len());
    let mut injected = false;

    for (role, content) in messages {
        if role == "system" && !injected {
            let merged = if content.trim().is_empty() {
                tool_directive.clone()
            } else {
                format!("{content}\n\n{tool_directive}")
            };
            out.push((role.clone(), merged));
            injected = true;
        } else {
            out.push((role.clone(), content.clone()));
        }
    }

    if !injected {
        out.insert(0, ("system".to_string(), tool_directive));
    }
    out
}

/// Render a portable tool-calling directive describing the available tools and
/// how the model should emit calls. Models trained for OpenAI-style function
/// calling (Qwen, Llama, etc.) generally honour `<tool_call>` JSON blocks
/// described this way.
fn build_tool_directive(tools_json: &str, tool_choice: Option<&str>) -> String {
    let mut directive = String::new();
    directive.push_str("You have access to the following tools:\n");
    directive.push_str(tools_json);
    directive.push_str(
        "\n\nTo call a tool, respond with a JSON object of the form \
         `{\"name\": \"<tool_name>\", \"arguments\": {<key>: <value>, ...}}` \
         wrapped in <tool_call></tool_call> tags. You may emit multiple \
         <tool_call> blocks. Do not place any other text inside the tags.",
    );
    if let Some(choice) = tool_choice
        && choice != "auto"
    {
        let how = match choice {
            "none" => "Do not call any tools for this turn.",
            "required" => "You must call at least one tool in your response.",
            other => other,
        };
        directive.push_str(&format!("\nTool choice: {how}"));
    }
    directive
}

fn message_content_as_text(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}
