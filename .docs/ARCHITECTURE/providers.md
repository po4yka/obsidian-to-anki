# LLM Providers

## Provider Comparison

| Provider       | Type        | Best For                   | Cost          |
| -------------- | ----------- | -------------------------- | ------------- |
| **Ollama**     | Local/Cloud | Privacy, offline           | Free (local)  |
| **LM Studio**  | Local       | GUI, testing               | Free          |
| **OpenRouter** | Cloud       | Qwen/Deepseek/Kimi/Minimax | Pay-per-token |

## Configuration Examples

### Ollama (Local)

```yaml
llm_provider: "ollama"
ollama_base_url: "http://localhost:11434"
generator_model: "qwen3:32b"
```

```bash
brew install ollama
ollama serve
ollama pull qwen3:32b
```

### OpenRouter (Cloud)

```yaml
llm_provider: "openrouter"
openrouter_api_key: "${OPENROUTER_API_KEY}"
generator_model: "qwen/qwen-2.5-72b-instruct"
openrouter_site_url: "https://yourapp.example.com" # Optional attribution
openrouter_site_name: "Obsidian → Anki Sync" # Optional attribution
```

#### Current Flow (Dec 2025)

-   **CLI sync** routes card generation through `src/obsidian_anki_sync/apf/generator.py`,
    which instantiates the OpenAI SDK with `base_url=https://openrouter.ai/api/v1`.
-   **Agent system** builds on `src/obsidian_anki_sync/providers/openrouter/provider.py` and
    `pydantic_ai_models.py`. The provider handles retry logic, structured-output fallbacks,
    and reasoning toggles for DeepSeek models, while PydanticAI reuses that HTTP stack.
-   **Preflight checks** (`obsidian-anki-sync check`) call `/models` to ensure the API key is
    valid but do not yet query credits or prompt-cache telemetry.

#### Structured Output Behavior

1. We request JSON Schema enforcement by default.
2. If OpenRouter (or a specific model) returns an empty/invalid payload, the provider
   retries without `response_format`, then raises if the fallback is still empty.
3. Models listed in `MODELS_WITH_STRUCTURED_OUTPUT_ISSUES` automatically skip strict mode.

#### Streaming & Telemetry

-   Set `llm_streaming_enabled: true` to enable SSE streaming for card generation and
    LangGraph agents. Streams emit chunks via `log_llm_stream_chunk` so you can monitor long
    generations without waiting for completion.
-   Chunk telemetry is tied to manifests/slugs in the APF generator and to operation names
    inside `run_agent_with_streaming`.

#### Reasoning Controls

-   Use `llm_reasoning_effort` (`auto|minimal|low|medium|high|none`) to map to OpenRouter’s
    `reasoning.effort` payload for DeepSeek reasoning models.
-   Stage-level overrides via `reasoning_effort_overrides` allow LangGraph agents to request
    higher effort for generation while keeping validators lean.

#### Remaining Gaps

-   Reasoning controls only apply to Grok models; OpenRouter’s newer
    `reasoning.effort=low|medium|high` contract is not exposed.
-   Rate-limit and credit telemetry are unavailable. `/api/v1/key` is not polled, so users
    only discover exhausted credits once 429 responses bubble up.
-   Documentation previously mentioned only the API key; optional attribution headers and
    prompt-caching practices were tribal knowledge.

See `.docs/IMPLEMENTATION_NOTES/openrouter-usage.md` for the detailed baseline.

### LM Studio

```yaml
llm_provider: "lm_studio"
lm_studio_base_url: "http://localhost:1234/v1"
```

## Model Selection

Model cascade: Agent-specific -> Default -> Provider default

```yaml
default_llm_model: "deepseek/deepseek-v3.2" # Latest: Excellent reasoning, 163K context
generator_model: "deepseek/deepseek-v3.2" # Override: Latest reasoning model
```

## Recommended Models by Agent

| Agent          | Requirement  | Recommended                                                         |
| -------------- | ------------ | ------------------------------------------------------------------- |
| Pre-Validator  | Fast, cheap  | qwen/qwen-2.5-7b-instruct (latest: $0.04/$0.10, 33K context)        |
| Generator      | High quality | deepseek/deepseek-v3.2 (latest: excellent reasoning, 163K context)  |
| Post-Validator | Balanced     | deepseek/deepseek-v3.2 (latest: excellent reasoning for validation) |

## Security

```bash
# Set API keys via environment (never commit)
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Troubleshooting

**Provider not available:** Check API key, network, rate limits

```bash
curl -H "Authorization: Bearer $API_KEY" https://api.provider.com/v1/models
```

**Model not found:** Verify model name (case-sensitive), check availability

**Rate limited:** Increase retry delay, reduce frequency, upgrade plan

---

**Related**: [Agents](agents.md) | [Configuration](../GUIDES/configuration.md)
