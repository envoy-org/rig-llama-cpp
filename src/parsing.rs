use rig_core::message::{AssistantContent, Reasoning, ToolCall, ToolFunction};
use rig_core::one_or_many::OneOrMany;
use serde_json::Value;

/// Extract a JSON object from grammar-constrained output that may be wrapped in
/// markdown code fences, ChatML role tokens, or extra prose.
///
/// Scans for the first `{` and returns the substring up to the matching `}`,
/// tracking brace depth and JSON string escaping so that braces inside strings
/// don't confuse the balance. Returns `None` if no balanced object is found.
pub(crate) fn extract_structured_json(raw_text: &str) -> Option<String> {
    let bytes = raw_text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;

    let mut depth: usize = 0;
    let mut in_string = false;
    let mut escaped = false;

    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }

        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(raw_text[start..=i].to_string());
                }
            }
            _ => {}
        }
    }

    None
}

pub(crate) fn parse_completion_output(
    raw_text: &str,
    has_json_schema: bool,
) -> Result<OneOrMany<AssistantContent>, String> {
    log::debug!("raw output:\n{raw_text}");

    // When the caller set an output schema, grammar-constrained generation produces
    // a valid JSON object — but chat templates often wrap it in role tokens
    // (e.g. `<|im_start|>assistant\n`) or markdown fences (```json ... ```).
    // Strip those before any other parsing so Rig's typed prompt can deserialize.
    if has_json_schema && let Some(json) = extract_structured_json(raw_text) {
        return Ok(OneOrMany::one(AssistantContent::text(json)));
    }

    // Try to surface tool calls. The OpenAI-compatible
    // `parse_response_oaicompat` path from llama-cpp-2 0.1.146 was removed in
    // 0.1.147, so we rely on our own parsers here (XML `<tool_call>` blocks,
    // bare / markdown-fenced JSON).
    if let Some(tool_calls) = parse_tool_calls(raw_text) {
        let mut content: Vec<AssistantContent> = Vec::new();
        for (i, (name, arguments)) in tool_calls.into_iter().enumerate() {
            content.push(AssistantContent::ToolCall(ToolCall::new(
                format!("tool-call-{i}"),
                ToolFunction::new(name, arguments),
            )));
        }
        if let Ok(result) = OneOrMany::many(content) {
            return Ok(result);
        }
    }

    // Try to surface reasoning wrapped in <think>...</think> before giving up
    // on structured content.
    if let Some((reasoning, text)) = split_thinking(raw_text) {
        let mut content = Vec::new();
        if !reasoning.is_empty() {
            content.push(AssistantContent::Reasoning(Reasoning::new(&reasoning)));
        }
        content.push(AssistantContent::text(text));
        if let Ok(result) = OneOrMany::many(content) {
            return Ok(result);
        }
    }

    Ok(OneOrMany::one(AssistantContent::text(raw_text.to_string())))
}

/// Unified tool-call parser. Tries, in order:
///
/// 1. `<tool_call>` blocks (Qwen-style XML parameter form or JSON form).
/// 2. Gemma-4's native `<|tool_call>call:NAME{...}<tool_call|>` DSL.
/// 3. Bare / markdown-fenced JSON objects shaped like
///    `{"name": ..., "arguments": ...}`.
///
/// Returns the first format that yields at least one tool call, so a response
/// that mixes prose and a single JSON tool call still parses.
pub(crate) fn parse_tool_calls(output: &str) -> Option<Vec<(String, Value)>> {
    if let Some(xml) = parse_xml_tool_calls(output) {
        return Some(xml);
    }
    if let Some(gemma) = parse_gemma_tool_calls(output) {
        return Some(gemma);
    }
    // Bare / markdown-fenced JSON. Models that ignore the `<tool_call>`
    // directive often emit a fenced ```json block instead.
    if let Some(json) = extract_structured_json(output)
        && let Ok(value) = serde_json::from_str::<Value>(&json)
        && let Some(tc) = json_value_to_tool_call(&value)
    {
        return Some(vec![tc]);
    }
    None
}

/// Parse XML-style tool calls emitted by some models (e.g. Qwen).
///
/// Two shapes are recognised:
///
/// 1. Parameter form:
/// ```text
/// <tool_call>
/// <function=write_file>
/// <parameter=path>
/// output.txt
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// 2. JSON form (emitted by the system-prompt directive in `prompt.rs`):
/// ```text
/// <tool_call>
/// {"name": "write_file", "arguments": {"path": "output.txt"}}
/// </tool_call>
/// ```
pub(crate) fn parse_xml_tool_calls(output: &str) -> Option<Vec<(String, Value)>> {
    let mut results = Vec::new();

    for block in output.split("<tool_call>").skip(1) {
        let block = block.split("</tool_call>").next().unwrap_or(block);
        let trimmed = block.trim();

        // JSON form first.
        if trimmed.starts_with('{')
            && let Ok(value) = serde_json::from_str::<Value>(trimmed)
            && let Some(tc) = json_value_to_tool_call(&value)
        {
            results.push(tc);
            continue;
        }

        // Parameter form fallback.
        if let Some(tc) = parse_parameter_form(block) {
            results.push(tc);
        }
    }

    if results.is_empty() {
        None
    } else {
        Some(results)
    }
}

/// Gemma-4 opens and closes strings with this same token, quote-style.
const GEMMA_STRING_DELIM: &str = "<|\"|>";

/// Tool-call markers, longest-first so `<|tool_call>` is never mistaken for
/// `<tool_call>`. Used to locate prose preceding the first call.
pub(crate) const TOOL_CALL_MARKERS: &[&str] = &["<|tool_call>", "<tool_call>"];

/// Parse Gemma-4's native tool-call blocks.
///
/// Gemma reverts to this format — rather than the portable `<tool_call>` JSON
/// directive injected into the system prompt — whenever it is prompted in its
/// own turn format, because that is what it was trained to emit:
///
/// ```text
/// <|tool_call>call:write_file{path:<|"|>a.txt<|"|>,retries:2}<tool_call|>
/// ```
///
/// Arguments are a bespoke DSL, not JSON: strings are wrapped in
/// `<|"|>` on both sides, `null` / `true` / `false` and bare numbers are
/// literals, and objects/arrays nest with `{k:v,...}` / `[a,b]`. Top-level
/// argument keys are unquoted; nested keys are string-wrapped.
pub(crate) fn parse_gemma_tool_calls(output: &str) -> Option<Vec<(String, Value)>> {
    let mut results = Vec::new();

    for block in output.split("<|tool_call>").skip(1) {
        let block = block.split("<tool_call|>").next().unwrap_or(block).trim();
        // The template writes `call:` before the function name.
        let block = block.strip_prefix("call:").unwrap_or(block);

        let Some(brace) = block.find('{') else {
            continue;
        };
        let name = block[..brace].trim();
        if name.is_empty() {
            continue;
        }

        let mut at = brace;
        let arguments = parse_gemma_value(block, &mut at).unwrap_or(Value::Null);
        results.push((name.to_string(), arguments));
    }

    if results.is_empty() {
        None
    } else {
        Some(results)
    }
}

fn skip_gemma_ws(s: &str, at: &mut usize) {
    while let Some(c) = s[*at..].chars().next() {
        if c.is_whitespace() {
            *at += c.len_utf8();
        } else {
            break;
        }
    }
}

/// Read one value of Gemma's argument DSL starting at `at`, advancing it past
/// what was consumed.
fn parse_gemma_value(s: &str, at: &mut usize) -> Option<Value> {
    skip_gemma_ws(s, at);
    let rest = s.get(*at..)?;

    if let Some(body) = rest.strip_prefix(GEMMA_STRING_DELIM) {
        let end = body.find(GEMMA_STRING_DELIM)?;
        *at += GEMMA_STRING_DELIM.len() + end + GEMMA_STRING_DELIM.len();
        return Some(Value::String(body[..end].to_string()));
    }

    if rest.starts_with('{') {
        *at += 1;
        let mut map = serde_json::Map::new();
        loop {
            skip_gemma_ws(s, at);
            let rest = s.get(*at..)?;
            if let Some(stripped) = rest.strip_prefix('}') {
                let _ = stripped;
                *at += 1;
                break;
            }
            if rest.is_empty() {
                return None;
            }
            let key = parse_gemma_key(s, at)?;
            skip_gemma_ws(s, at);
            if !s.get(*at..)?.starts_with(':') {
                return None;
            }
            *at += 1;
            let value = parse_gemma_value(s, at)?;
            map.insert(key, value);
            skip_gemma_ws(s, at);
            if s.get(*at..)?.starts_with(',') {
                *at += 1;
            }
        }
        return Some(Value::Object(map));
    }

    if rest.starts_with('[') {
        *at += 1;
        let mut items = Vec::new();
        loop {
            skip_gemma_ws(s, at);
            let rest = s.get(*at..)?;
            if rest.starts_with(']') {
                *at += 1;
                break;
            }
            if rest.is_empty() {
                return None;
            }
            items.push(parse_gemma_value(s, at)?);
            skip_gemma_ws(s, at);
            if s.get(*at..)?.starts_with(',') {
                *at += 1;
            }
        }
        return Some(Value::Array(items));
    }

    // Bare scalar: runs until the next structural character.
    let end = rest.find([',', '}', ']']).unwrap_or(rest.len());
    let token = rest[..end].trim();
    *at += end;
    Some(match token {
        "null" => Value::Null,
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        _ => token
            .parse::<i64>()
            .map(Value::from)
            .or_else(|_| token.parse::<f64>().map(Value::from))
            .unwrap_or_else(|_| Value::String(token.to_string())),
    })
}

/// Object keys are either string-wrapped (nested objects) or bare (top-level
/// argument names, which the template renders with `escape_keys=False`).
fn parse_gemma_key(s: &str, at: &mut usize) -> Option<String> {
    skip_gemma_ws(s, at);
    let rest = s.get(*at..)?;

    if let Some(body) = rest.strip_prefix(GEMMA_STRING_DELIM) {
        let end = body.find(GEMMA_STRING_DELIM)?;
        *at += GEMMA_STRING_DELIM.len() + end + GEMMA_STRING_DELIM.len();
        return Some(body[..end].to_string());
    }

    let end = rest.find(':')?;
    *at += end;
    let key = rest[..end].trim();
    if key.is_empty() {
        None
    } else {
        Some(key.to_string())
    }
}

/// Extract a single `(name, arguments)` tool call from a JSON value that is
/// either `{"name": ..., "arguments": ...}` or the OpenAI-shaped
/// `{"function": {"name": ..., "arguments": ...}}`.
fn json_value_to_tool_call(value: &Value) -> Option<(String, Value)> {
    let name = value.get("name").and_then(Value::as_str).or_else(|| {
        value
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
    })?;
    let arguments = match value.get("arguments") {
        Some(Value::String(s)) => {
            serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
        }
        Some(other) => other.clone(),
        None => value
            .get("function")
            .and_then(|f| f.get("arguments"))
            .map(|a| match a {
                Value::String(s) => {
                    serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
                }
                other => other.clone(),
            })
            .unwrap_or(Value::Null),
    };
    Some((name.to_string(), arguments))
}

/// Parse the Qwen parameter form (`<function=NAME>` + `<parameter=KEY>` blocks).
fn parse_parameter_form(block: &str) -> Option<(String, Value)> {
    let func_start = block.find("<function=")?;
    let after_eq = &block[func_start + "<function=".len()..];
    let func_name_end = after_eq.find('>')?;
    let func_name = after_eq[..func_name_end].trim().to_string();

    let mut args = serde_json::Map::new();
    let mut search_from = 0;
    while let Some(param_start) = block[search_from..].find("<parameter=") {
        let abs_start = search_from + param_start;
        let after_param_eq = &block[abs_start + "<parameter=".len()..];
        let Some(key_end) = after_param_eq.find('>') else {
            break;
        };
        let key = after_param_eq[..key_end].trim();

        let value_start = abs_start + "<parameter=".len() + key_end + 1;
        let Some(param_end) = block[value_start..].find("</parameter>") else {
            break;
        };
        let value = block[value_start..value_start + param_end].trim();

        args.insert(key.to_string(), Value::String(value.to_string()));
        search_from = value_start + param_end + "</parameter>".len();
    }

    if func_name.is_empty() {
        None
    } else {
        Some((func_name, Value::Object(args)))
    }
}

/// Reasoning delimiters, as `(open, close)` pairs, for the model families this
/// crate targets. Models disagree on the markup, so we recognise each shape
/// rather than assuming the `<think>` convention is universal.
const THINKING_MARKERS: &[(&str, &str)] = &[
    // Qwen, DeepSeek-R1, and most reasoning models.
    ("<think>", "</think>"),
    // Gemma-4's thought channel.
    ("<|channel>thought", "<channel|>"),
];

/// Split reasoning from the visible response. Returns `(reasoning, text)` when
/// a thinking block is present, using whichever known marker pair appears
/// first. An unterminated block — the token cap hit mid-thought — is treated as
/// reasoning all the way to the end, leaving no visible text.
fn split_thinking(output: &str) -> Option<(String, String)> {
    let (open, close, start) = THINKING_MARKERS
        .iter()
        .filter_map(|&(open, close)| output.find(open).map(|at| (open, close, at)))
        .min_by_key(|&(_, _, at)| at)?;

    let body = start + open.len();
    let (reasoning, tail) = match output[body..].find(close) {
        Some(rel) => (
            &output[body..body + rel],
            &output[body + rel + close.len()..],
        ),
        None => (&output[body..], ""),
    };

    let mut text = String::with_capacity(start + tail.len());
    text.push_str(&output[..start]);
    text.push_str(tail);
    // Defensive: drop any stray closer left by a second block.
    let text = text.replace(close, "");
    Some((reasoning.trim().to_string(), text.trim().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_structured_json_plain_object() {
        let out = extract_structured_json(r#"{"name":"Ada","age":36}"#).unwrap();
        assert_eq!(out, r#"{"name":"Ada","age":36}"#);
    }

    #[test]
    fn extract_structured_json_strips_markdown_fence() {
        let raw = "```json\n{\n  \"name\": \"Ada\",\n  \"age\": 36\n}\n```";
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, "{\n  \"name\": \"Ada\",\n  \"age\": 36\n}");
    }

    #[test]
    fn extract_structured_json_strips_plain_fence() {
        let raw = "```\n{\"ok\": true}\n```";
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, r#"{"ok": true}"#);
    }

    #[test]
    fn extract_structured_json_strips_chatml_role_prefix() {
        let raw = "<|im_start|>assistant\n```json\n{\"value\": 1}\n```";
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, r#"{"value": 1}"#);
    }

    #[test]
    fn extract_structured_json_strips_leading_prose() {
        let raw = "Sure, here is the answer: {\"answer\": 42}";
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, r#"{"answer": 42}"#);
    }

    #[test]
    fn extract_structured_json_handles_nested_objects() {
        let raw = r#"```json
{"person": {"name": "Ada", "skills": {"lang": "rust"}}, "age": 36}
```"#;
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(
            out,
            r#"{"person": {"name": "Ada", "skills": {"lang": "rust"}}, "age": 36}"#
        );
    }

    #[test]
    fn extract_structured_json_ignores_braces_inside_strings() {
        let raw = r#"{"text": "an { inside } string", "ok": true}"#;
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, raw);
    }

    #[test]
    fn extract_structured_json_handles_escaped_quotes_in_strings() {
        let raw = r#"{"text": "she said \"hi\"", "brace": "}"}"#;
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, raw);
    }

    #[test]
    fn extract_structured_json_stops_at_first_balanced_object() {
        let raw = r#"{"first": 1} and then {"second": 2}"#;
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, r#"{"first": 1}"#);
    }

    #[test]
    fn extract_structured_json_returns_none_when_unbalanced() {
        assert!(extract_structured_json(r#"{"broken": "#).is_none());
    }

    #[test]
    fn extract_structured_json_returns_none_when_no_object() {
        assert!(extract_structured_json("just plain text, no json").is_none());
        assert!(extract_structured_json("").is_none());
    }

    #[test]
    fn extract_structured_json_handles_real_qwen_output() {
        // Shape observed in practice: ChatML role token, markdown fence, indented body.
        let raw = "<|im_start|>assistant\n```json\n{\n  \"age\": 36,\n  \"name\": \"Ada\",\n  \"occupation\": \"Software Engineer\"\n}\n```";
        let out = extract_structured_json(raw).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["name"], "Ada");
        assert_eq!(parsed["age"], 36);
        assert_eq!(parsed["occupation"], "Software Engineer");
    }

    #[test]
    fn parse_xml_tool_calls_json_form() {
        let raw = "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"timezone\": \"UTC\"}}\n</tool_call>";
        let out = parse_xml_tool_calls(raw).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "get_time");
        assert_eq!(out[0].1["timezone"], "UTC");
    }

    #[test]
    fn parse_xml_tool_calls_parameter_form() {
        let raw = "<tool_call>\n<function=write_file>\n<parameter=path>\noutput.txt\n</parameter>\n<parameter=content>\nHello\n</parameter>\n</function>\n</tool_call>";
        let out = parse_xml_tool_calls(raw).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "write_file");
        assert_eq!(out[0].1["path"], "output.txt");
        assert_eq!(out[0].1["content"], "Hello");
    }

    #[test]
    fn parse_xml_tool_calls_multiple_blocks() {
        let raw = "<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>\n<tool_call>{\"name\": \"b\", \"arguments\": {}}</tool_call>";
        let out = parse_xml_tool_calls(raw).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, "a");
        assert_eq!(out[1].0, "b");
    }

    #[test]
    fn parse_gemma_tool_calls_no_arguments() {
        // The shape Gemma-4 actually emitted in the e2e tool roundtrip.
        let raw =
            "<|channel>thought\nI should call it.<channel|><|tool_call>call:get_time{}<tool_call|>";
        let out = parse_gemma_tool_calls(raw).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "get_time");
        assert_eq!(out[0].1, serde_json::json!({}));
    }

    #[test]
    fn parse_gemma_tool_calls_scalar_arguments() {
        let raw = "<|tool_call>call:write_file{path:<|\"|>a.txt<|\"|>,retries:2,force:true,note:null}<tool_call|>";
        let out = parse_gemma_tool_calls(raw).unwrap();
        assert_eq!(out[0].0, "write_file");
        assert_eq!(out[0].1["path"], "a.txt");
        assert_eq!(out[0].1["retries"], 2);
        assert_eq!(out[0].1["force"], true);
        assert_eq!(out[0].1["note"], Value::Null);
    }

    #[test]
    fn parse_gemma_tool_calls_nested_and_sequence_arguments() {
        // Nested keys are string-wrapped; sequences use [a,b].
        let raw = "<|tool_call>call:cfg{opts:{<|\"|>depth<|\"|>:3},tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]}<tool_call|>";
        let out = parse_gemma_tool_calls(raw).unwrap();
        assert_eq!(out[0].1["opts"]["depth"], 3);
        assert_eq!(out[0].1["tags"], serde_json::json!(["a", "b"]));
    }

    #[test]
    fn parse_gemma_tool_calls_multiple_blocks() {
        let raw = "<|tool_call>call:a{}<tool_call|><|tool_call>call:b{x:1}<tool_call|>";
        let out = parse_gemma_tool_calls(raw).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, "a");
        assert_eq!(out[1].0, "b");
        assert_eq!(out[1].1["x"], 1);
    }

    #[test]
    fn parse_gemma_tool_calls_preserves_braces_inside_strings() {
        let raw = "<|tool_call>call:echo{msg:<|\"|>a{b},c<|\"|>}<tool_call|>";
        let out = parse_gemma_tool_calls(raw).unwrap();
        assert_eq!(out[0].1["msg"], "a{b},c");
    }

    #[test]
    fn parse_gemma_tool_calls_returns_none_without_blocks() {
        assert!(parse_gemma_tool_calls("just prose").is_none());
        // A Qwen-style block must not be picked up by the Gemma parser.
        assert!(parse_gemma_tool_calls("<tool_call>{\"name\":\"a\"}</tool_call>").is_none());
    }

    #[test]
    fn parse_tool_calls_dispatches_to_the_gemma_dialect() {
        let raw = "<|tool_call>call:get_time{}<tool_call|>";
        let out = parse_tool_calls(raw).unwrap();
        assert_eq!(out[0].0, "get_time");
    }

    #[test]
    fn parse_completion_output_surfaces_gemma_tool_call() {
        let raw = "<|channel>thought\nreasoning<channel|><|tool_call>call:get_time{}<tool_call|>";
        let content = parse_completion_output(raw, false).unwrap();
        let call = content.iter().find_map(|c| match c {
            AssistantContent::ToolCall(tc) => Some(tc.clone()),
            _ => None,
        });
        assert_eq!(
            call.expect("expected a tool call").function.name,
            "get_time"
        );
    }

    #[test]
    fn split_thinking_extracts_reasoning() {
        let raw = "<think>let me consider</think>The answer is 42.";
        let (reasoning, text) = split_thinking(raw).unwrap();
        assert_eq!(reasoning, "let me consider");
        assert_eq!(text, "The answer is 42.");
    }

    #[test]
    fn split_thinking_extracts_gemma_thought_channel() {
        let raw = "<|channel>thought\nweigh the options\n<channel|>The answer is 42.";
        let (reasoning, text) = split_thinking(raw).unwrap();
        assert_eq!(reasoning, "weigh the options");
        assert_eq!(text, "The answer is 42.");
    }

    #[test]
    fn split_thinking_handles_unterminated_block() {
        // Token cap hit mid-thought: everything is reasoning, no visible text.
        let raw = "<think>still working through it";
        let (reasoning, text) = split_thinking(raw).unwrap();
        assert_eq!(reasoning, "still working through it");
        assert_eq!(text, "");
    }

    #[test]
    fn split_thinking_keeps_prose_before_the_block() {
        let raw = "Sure.<think>hmm</think>Done.";
        let (reasoning, text) = split_thinking(raw).unwrap();
        assert_eq!(reasoning, "hmm");
        assert_eq!(text, "Sure.Done.");
    }

    #[test]
    fn split_thinking_returns_none_without_markers() {
        assert!(split_thinking("just a plain answer").is_none());
    }

    #[test]
    fn parse_completion_output_surfaces_gemma_reasoning() {
        let raw = "<|channel>thought\nreason about it\n<channel|>Because of Rayleigh scattering.";
        let content = parse_completion_output(raw, false).unwrap();
        let has_reasoning = content
            .iter()
            .any(|c| matches!(c, AssistantContent::Reasoning(_)));
        assert!(
            has_reasoning,
            "expected reasoning content, got: {content:?}"
        );
    }
}
