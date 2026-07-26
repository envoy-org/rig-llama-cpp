# Changelog

All notable changes to `rig-llama-cpp` are documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The crate is pre-1.0, so the [SemVer](https://semver.org/) policy below applies.

## Versioning policy

While the crate is on `0.x`:

- A bump to `0.Y` (e.g. `0.1` → `0.2`) signals a **breaking change** in the public
  API or in the embedded `llama-cpp-2` / `llama-cpp-sys-2` versions.
- A bump to `0.x.Z` (e.g. `0.1.0` → `0.1.1`) is reserved for additive or
  non-breaking changes.

The public surface is fully owned by this crate — `KvCacheType`, the
parameter structs, and `LoadError` are all defined here, not re-exported
from `llama-cpp-2`. A new upstream `ggml_type` is therefore an additive
`0.1.x` change here (we add a corresponding shim variant), not a breaking
release.

## [0.5.0] — 2026-07-26

### Changed

- **Bumped `llama-cpp-2` / `llama-cpp-sys-2` to `0.1.152`.** Upstream `0.1.152`
  replaced the all-in-one `LlamaSampler::llguidance(model, kind, data)`
  constructor with a bring-your-own-`Matcher` API: callers now build the
  token environment (`LlamaSampler::llguidance_tok_env`), a
  `ParserFactory`, and a `Matcher` themselves, then convert it with
  `LlamaSampler::from`. That pipeline lives in `build_schema_sampler`
  (`src/sampling.rs`), and `llguidance` is now a direct dependency pinned to
  the `1.7` line `llama-cpp-2` resolves — the `Matcher` crosses the crate
  boundary, so both must see the same `llguidance` / `toktrie`. Behaviour of
  `json_schema` requests is unchanged, including the fall back to
  unconstrained sampling when a schema cannot be compiled.

  Nothing else in `0.1.151` / `0.1.152` is breaking for this crate: the MTP
  speculative-decoding API, the model-load progress callback, the extra
  KV-cache and `seq_state` bindings, the tensor-buft-override accessor, and
  the `mkl` feature are all additive.

### Performance

- **The llguidance token environment is built once per loaded model instead
  of once per request.** Building it walks the whole vocabulary and
  detokenizes every id — hundreds of milliseconds on a large vocab — and the
  pre-`0.1.152` API did that inside every `LlamaSampler::llguidance()` call,
  so every structured-output request paid it. It depends only on the
  vocabulary, so it is now cached in `WorkerModel` behind a `OnceCell` and
  shared by all later requests; a model reload drops it along with the
  vocabulary it describes. Initialisation stays lazy, so callers that never
  pass a JSON schema never pay the cost at all.

## [0.4.1] — 2026-07-26

### Fixed

- **Streaming structured output no longer emits the JSON object twice.** A
  `json_schema` request without tools streamed every token piece as it was
  sampled *and* then emitted the whole cleaned object again as a corrective
  chunk, so consumers concatenating the stream got the object back-to-back and
  `serde_json` rejected it with "trailing characters". Structured-output turns
  now buffer like tool-calling turns and emit exactly one cleaned chunk, which
  also keeps the streamed text identical to the aggregated assistant message.
  When no balanced object can be extracted the raw output is emitted instead,
  so a failed extraction can no longer end a stream with no text at all.

- **Models whose chat template llama.cpp cannot apply are no longer prompted
  with the wrong format.** `apply_chat_template` is llama.cpp's *non-jinja*
  applier: it recognises a fixed set of known template shapes and returns `-1`
  for anything else. Models shipping a bespoke jinja template — Gemma-4, whose
  template renders `<|turn>role` turns — failed it and were silently downgraded
  to ChatML, a format they were never trained on. Such templates are now
  rendered with `minijinja` (new dependency) before ChatML is considered, and a
  genuinely unusable template warns instead of failing quietly. llama.cpp's
  applier remains the primary path, so every model it already handles renders
  byte-for-byte as before.

- **`enable_thinking` is no longer inert.** The flag parsed from
  `additional_params` (`{"thinking": true}`) had nowhere to go once
  `llama-cpp-2` 0.1.147 dropped `chat_template_kwargs`, so templates gating
  reasoning behind it — Gemma-4 defaults it to `false` — never produced any.
  It is now forwarded as a real template variable on the minijinja path. On
  llama.cpp's own path it necessarily stays advisory: that C API takes only
  `(role, content)` pairs.

- **Gemma-4's native tool-call dialect is now parsed.** Prompted in its own
  turn format, Gemma emits `<|tool_call>call:name{k:v}<tool_call|>` — a bespoke
  DSL, not JSON — rather than the portable `<tool_call>` protocol the injected
  system directive asks for, so tool calls were silently missed.
  `parse_tool_calls` now recognises it alongside the existing formats,
  including its `<|"|>`-delimited strings and nested objects/arrays.

- **Reasoning is recognised beyond `<think>`.** `split_thinking` matched only
  `<think>…</think>`, so Gemma-4's `<|channel>thought…<channel|>` never
  surfaced as `AssistantContent::Reasoning`. Both markers are now understood,
  and a block left unterminated by the token cap is treated as reasoning to the
  end rather than being dropped.

## [0.4.0] — 2026-07-26

### Changed

- **Bumped `rig-core` to `0.40.0`.** The crate's own source needed no changes —
  every upstream break landed on the *consumer* side of the API (tool authoring
  and the agent stream), not on the provider side this crate implements. Because
  `rig-core` is a public dependency, the bump is still a breaking `0.Y` release
  for downstream users, who must move to `rig-core` 0.40 in lockstep. The
  upstream breaks that touch code written against this crate:
  - **`Tool::definition()` is gone.** Tool authors now implement flat metadata
    directly: `fn description(&self) -> String` and
    `fn parameters(&self) -> serde_json::Value`. Rig builds the provider-facing
    `ToolDefinition` itself when the tool is registered on an agent, so the
    `rig_core::completion::ToolDefinition` import usually disappears with it.
    The bundled examples are updated accordingly.
  - **`agent::FinalResponse` is replaced by `agent::PromptResponse`**, the
    unified result type now shared with the blocking prompt surface.
    `MultiTurnStreamItem::FinalResponse` carries it. `empty()` and `usage()`
    carry over unchanged; `response()` is now `output()`.
  - **`StreamedAssistantContent` gained an `Unknown(serde_json::Value)`
    variant**, carrying provider-native output items that Rig does not model
    (hosted-tool results and the like) rather than dropping them silently.
    Exhaustive `match`es over the enum need a new arm. This adapter never emits
    the variant — llama.cpp has no unmodeled output items — so the arm is inert
    here, but it is required to compile.
  - `MultiTurnStreamItem` gained a `ToolExecutionStart` variant. The enum was
    already `#[non_exhaustive]`, so existing wildcard arms absorb it.

  Two upstream changes look like they should affect this crate but do not:
  the new `ToolChoice::Function { name }` variant is on the *OpenAI provider's*
  `ToolChoice`, not `rig_core::message::ToolChoice`, so `prepare_request`'s
  exhaustive match is untouched; and the `evals` module and `ProviderClient`
  derive macro removals cover surfaces this crate never used.

### Fixed

- **Streaming responses now report real token usage when converted to a
  `CompletionResponse`.** Upstream's
  `From<StreamingCompletionResponse<R>> for CompletionResponse<Option<R>>`
  previously hardcoded `Usage::new()` with the note that "usage is not tracked
  in streaming responses"; it now derives usage from the final response. Since
  this crate's `StreamChunk` already carried prompt/completion counts through
  its `GetTokenUsage` impl, the numbers propagate with no change on our side.

## [0.3.0] — 2026-06-28

### Changed

- **Bumped `rig-core` to `0.39.0`.** Upstream made `GetTokenUsage::token_usage`
  return `Usage` directly instead of `Option<Usage>`, using the zero-valued
  `Usage::new()` as the documented sentinel for missing provider metrics. The
  `GetTokenUsage` impl for `StreamChunk` now follows suit: when prompt and
  completion token counts are unavailable it returns `Usage::new()` rather than
  `None`. Because this changes the crate's public trait-impl signature, it is a
  breaking `0.Y` release. The other breaking changes in `rig-core` 0.39 (the
  sans-IO `AgentRun` state machine and deterministic tool registration) affect
  the agent loop and `ToolSet`, neither of which this crate uses.

## [0.2.1] — 2026-06-18

### Fixed

- **Image inference builds against `llama-cpp-2` 0.1.150.** Upstream
  `MtmdBitmap::from_buffer` gained a `placeholder: bool` parameter (pass
  `false` to decode and load the real pixels/audio, `true` for a data-less
  placeholder used only for token counting). `run_image_inference` still
  called it with two arguments, so the crate failed to compile under the
  `mtmd` feature. It now passes `placeholder = false`, matching the previous
  behaviour of decoding the actual media for multimodal inference.

## [0.2.0] — 2026-06-18

### Changed

- **Bumped `rig-core` to `0.38.2`.** The upstream library renamed its crate
  root from `rig` to `rig_core` (now `use rig_core::…`); the two new
  `Usage` fields (`tool_use_prompt_tokens`, `reasoning_tokens`) are populated
  as `0` since llama.cpp does not report them separately.
- **Bumped `llama-cpp-2` to `0.1.150` / `llama-cpp-sys-2` to `0.1.150`,
  migrating off the removed `openai` module.** `llama-cpp-2` 0.1.147 deleted
  the entire `openai` module
  (`apply_chat_template_oaicompat`, `OpenAIChatTemplateParams`,
  `ChatTemplateResult`, `parse_response_oaicompat`,
  `streaming_state_oaicompat`) plus `GrammarTriggerType`. This release
  replaces every consumer of that API:

  - **Prompt rendering** now uses `apply_chat_template` (role + content
    messages). Tool schemas are injected into the system prompt (the
    portable pattern now that the jinja engine no longer receives a
    `tools` parameter), and `<tool_call>` XML is requested as the emission
    format.
  - **Structured output** (`json_schema`) is enforced by the new
    `llguidance` sampler (`LlamaSampler::llguidance(model, "json", schema)`),
    which is cleaner than the old oaicompat path. The `common` and
    `llguidance` features of `llama-cpp-2` are now always enabled.
  - **Tool-call parsing** is unified in `parse_tool_calls`, which recognises
    `<tool_call>` blocks (both Qwen XML parameter form and JSON form) and
    bare / markdown-fenced `{"name":…,"arguments":…}` JSON.
  - **Streaming** emits raw text pieces incrementally for plain and
    structured turns, and buffers tool-calling turns so the complete output
    is parsed for tool calls at flush (the incremental OAI streaming parser
    is gone with the `openai` module).

### Fixed

- **Multi-text embedding on mixture-of-experts models.** Packing multiple
  sequences into one `encode` batch tripped a `GGML_ASSERT(ggml_can_mul_mat)`
  on MoE architectures such as `nomic-embed-text-v2-moe`. Embeddings are now
  encoded one text at a time, which is correct for every architecture.

### Removed

- **Template-derived grammar / `chat_template_kwargs`.** The
  `apply_chat_template_oaicompat` plumbing that forwarded `enable_thinking`,
  `grammar`, `grammar_lazy`/`grammar_triggers`, `preserved_tokens`, and
  `additional_stops` to the jinja engine no longer exists upstream. The
  `enable_thinking` flag is parsed from `additional_params` but is advisory
  only: thinking-enabled remains the template default; thinking-disabled can
  no longer be enforced through the template.

## [0.1.4] — 2026-05-06

### Fixed

- **Avoided a `GGML_ASSERT(!stacks.empty())` abort in grammar-constrained
  sampling.** Upstream
  [llama-cpp-rs#1007](https://github.com/utilityai/llama-cpp-rs/issues/1007)
  reports that `LlamaSampler::sample(ctx, idx)` aborts on the first
  sample call whenever the chain contains `LlamaSampler::grammar(...)`,
  even with a trivial `root ::= "a"` grammar (`llama-grammar.cpp:940`).
  Both of our grammar consumers go through that API: tool-call grammar
  from `ChatTemplateResult.grammar` and the GBNF that llama.cpp
  synthesizes for `output_schema` / json_schema requests. We could not
  reproduce the abort locally on Qwen3.5-2B Q4_K_M with the default
  Vulkan backend, but the upstream API combination is identical and the
  failure mode is `abort()` — not catchable from Rust — so we work
  around it preemptively. When grammar is present we now sample via the
  manual `LlamaTokenDataArray` + `apply_sampler` path that the issue
  reporter confirmed is crash-free, and call `sampler.accept(token)`
  explicitly (the manual path doesn't auto-accept the way `sample()`
  does). The non-grammar hot path is unchanged. Removable once upstream
  llama.cpp resolves the assert and `llama-cpp-2` ships a release that
  resyncs to it — see the comment on `sample_one` in `src/sampling.rs`.

## [0.1.3] — 2026-05-03

### Fixed

- **Streaming structured output silently swallowed every chunk.** When
  a `json_schema` was set on the request and the stream path was used
  (`agent.stream_chat(...)`), the OAI-compatible chat-template streaming
  parser (`llama_rs_chat_parse_state_update_oaicompat`) buffered every
  partial piece and then errored out on the final flush
  (`FfiError(-3)`), leaving consumers with zero text chunks. End users
  of crates like `chatty` saw `EOF while parsing a value at line 1
  column 0` because the accumulated buffer was empty. We now bypass
  that parser entirely whenever a `json_schema` is set: pieces still
  accumulate into the inference buffer, and after the loop completes
  we emit a single corrective chunk containing the result of
  `extract_structured_json`. That strips any leading role markers a
  template may leak (`<|im_start|>assistant\n\n…`) and any trailing
  junk before the JSON is sent downstream. Reproduces against both
  Qwen-3 and Gemma-4 — new e2e tests
  `qwen_structured_output_streaming` /
  `gemma_structured_output_streaming` (gated behind the existing
  `--ignored` flag like the other model-bearing tests) keep this
  honest.

## [0.1.2] — 2026-05-03

### Fixed

- **Empty-piece tokens no longer abort generation.** When `llama.cpp`'s
  `llama_token_to_piece` returns size 0 (control / unused / unknown-
  attribute tokens like Qwen3's `<|object_ref_*|>` pair, or a
  grammar-constrained sample landing on `<|fim_pad|>`), `llama-cpp-2`
  surfaces it as `TokenToStringError::UnknownTokenType`. Previously the
  sampling loop turned this into a hard error
  (`Token to piece failed: Unknown Token Type`), aborting the whole
  generation on the first such token. Canonical `llama.cpp` treats empty
  pieces as "no text emitted, keep generating" — the token is still
  consistent with the KV cache because we add it to the batch on the
  next iteration. We now do the same: empty pieces are emitted as empty
  strings and generation continues. Real errors
  (`InsufficientBufferSpace`, `FromUtf8Error`, …) still propagate. New
  unit tests in `sampling::tests` cover the three branches.

## [0.1.1] — 2026-05-02

### Changed

- **README polish.** Centered the title, added a tagline, surfaced
  crates.io / docs.rs / license / CI shields.io badges, expanded the
  intro paragraph, and named MIT explicitly in the License section.
  No code changes.

## [0.1.0] — 2026-05-01

Initial public release.

### Highlights

- **Rig integration.** Implements `rig::client::CompletionClient` /
  `rig::completion::CompletionModel` and the matching embedding traits, so
  any GGUF model is a drop-in for cloud Rig providers.
- **Local GGUF inference.** Any architecture supported by upstream
  `llama-cpp-2` (`0.1.146`).
- **Streaming and one-shot** completion, **tool calling** on OpenAI-template
  models, **structured output** via grammar-constrained sampling, and
  **reasoning / thinking deltas** surfaced separately from the main response
  stream.
- **Vision (multimodal) inference** via the `mtmd` feature for models that
  ship an `mmproj` projector.
- **Automatic GPU/CPU layer fitting** — llama.cpp probes available device
  memory and picks `n_gpu_layers` for you. Tunable per-device margins via
  `FitParams`.
- **KV-cache prefix reuse + state checkpoints** so multi-turn conversations
  skip re-decoding the shared prefix, including a checkpoint-based fallback
  for hybrid / recurrent architectures whose memory rejects partial trims.
- **Configurable KV-cache quantization** (`F16` default, `Q8_0` / `Q4_0`
  available) for VRAM savings at long contexts.
- **Pluggable backends** as opt-in Cargo features: `vulkan`, `cuda`, `metal`,
  `rocm`, plus `openmp` (CPU threading) and `mtmd` (multimodal). Default
  build is CPU-only and works on any host.
- **Builder-pattern construction** via `Client::builder(model_path)`, with
  the legacy positional `Client::from_gguf` constructors retained for
  backward compatibility.
- **Bounded inference command channel** with backpressure, plus an
  `Arc<AtomicBool>` cancel signal that lets `Drop` (and future per-request
  cancel hooks) tear down a long generation within a single decode step.
- **Typed errors** (`LoadError`, `#[non_exhaustive]`) on every load-stage
  entry point — no `anyhow` in the public API.
- **`log` crate facade** for library-level diagnostics; configure verbosity
  via `RUST_LOG=rig_llama_cpp=debug`. The `RIG_LLAMA_CPP_LOGS=1` env var
  toggles llama.cpp's own C-side log stream.

### Known caveats

- mtmd log suppression is temporarily disabled — upstream `llama-cpp-2`
  `0.1.146` does not yet expose `mtmd::void_mtmd_logs`, so loading an
  `mmproj` projector with the `mtmd` feature on may print to stderr.
  Tracked as a follow-up; will be re-enabled when the upstream API lands.
