# Framework and Library Usage Review

**Date**: 2025-01-27
**Reviewer**: AI Architect
**Scope**: Review of incorrect usage patterns in connected frameworks and libraries

## Executive Summary

This review identifies several critical issues with framework usage that could lead to runtime errors, performance degradation, and maintainability problems. The most significant issues involve async/await patterns in LangGraph nodes and inefficient resource management.

## Critical Issues

### 1. LangGraph: Using `asyncio.run()` in Synchronous Node Functions

**Severity**: HIGH
**Location**: `src/obsidian_anki_sync/agents/langgraph_orchestrator.py`

**Problem**:
All LangGraph node functions are synchronous but use `asyncio.run()` to call async PydanticAI agents:

```python
def pre_validation_node(state: PipelineState) -> PipelineState:
    # ...
    pre_result = asyncio.run(
        pre_validator.validate(...)
    )
```

**Issues**:
1. **Event Loop Conflicts**: `asyncio.run()` creates a new event loop. If called from an async context or if there's already a running loop, it will raise `RuntimeError: asyncio.run() cannot be called from a running event loop`.
2. **Performance**: Creating a new event loop for each node execution is inefficient.
3. **Best Practice Violation**: LangGraph natively supports async node functions, which is the recommended approach.

**Affected Functions**:
- `pre_validation_node()` (line 163)
- `card_splitting_node()` (line 269)
- `generation_node()` (line 357)
- `post_validation_node()` (line 451)
- `context_enrichment_node()` (line 574)
- `memorization_quality_node()` (line 681)
- `duplicate_detection_node()` (line 797)

**Recommendation**:
Convert all node functions to async and use `await` directly:

```python
async def pre_validation_node(state: PipelineState) -> PipelineState:
    # ...
    pre_result = await pre_validator.validate(...)
```

Then ensure the workflow is invoked with async support:

```python
final_state = await self.app.ainvoke(initial_state, config=...)
```

**References**:
- LangGraph documentation recommends async nodes for async operations
- PydanticAI agents are async by design

---

### 2. PydanticAI: Model Creation Inside Node Functions

**Severity**: MEDIUM
**Location**: `src/obsidian_anki_sync/agents/langgraph_orchestrator.py`

**Problem**:
PydanticAI models are created inside node functions on every execution:

```python
def pre_validation_node(state: PipelineState) -> PipelineState:
    model_name = state["config"].get_model_for_agent("pre_validator")
    model = create_openrouter_model_from_env(model_name=model_name)
    pre_validator = PreValidatorAgentAI(model=model, temperature=0.0)
```

**Issues**:
1. **Performance**: Model creation involves HTTP client initialization and configuration parsing, which is expensive.
2. **Resource Leaks**: Creating new HTTP clients repeatedly can exhaust connection pools.
3. **Inefficiency**: Models should be created once and reused across multiple invocations.

**Recommendation**:
Create models during orchestrator initialization and pass them through state or store them as instance variables:

```python
class LangGraphOrchestrator:
    def __init__(self, ...):
        # Create models once
        self.pre_validator_model = create_openrouter_model_from_env(...)
        self.generator_model = create_openrouter_model_from_env(...)
        # ...
```

Or use a model cache/factory pattern to reuse models across invocations.

---

### 3. httpx: Mixing Sync and Async Clients

**Severity**: MEDIUM
**Location**: Multiple files

**Problem**:
The codebase uses synchronous `httpx.Client` throughout, but PydanticAI internally uses async HTTP clients:

```python
# In pydantic_ai_models.py
model = OpenAIModel(
    model_name,
    base_url=base_url,
    api_key=api_key,
    http_client=None,  # Will use default httpx client (async)
)
```

Meanwhile, other providers use sync clients:

```python
# In anki/client.py
self.session = httpx.Client(timeout=timeout, limits=...)
```

**Issues**:
1. **Inconsistency**: Mixing sync and async HTTP clients can cause blocking issues.
2. **Performance**: Sync clients block the event loop when used in async contexts.
3. **Resource Management**: Different client types may have different connection pooling behaviors.

**Recommendation**:
1. For PydanticAI models, explicitly pass an `httpx.AsyncClient` instance with proper configuration.
2. For AnkiClient, consider making it async-compatible or ensure it's only used in sync contexts.
3. Document which clients are sync vs async and why.

---

### 4. SQLite: Connection Management and WAL Mode

**Severity**: LOW
**Location**: `src/obsidian_anki_sync/sync/state_db.py`

**Problem**:
SQLite connection is created once and reused, but there's no explicit connection pooling or async support:

```python
def __init__(self, db_path: Path):
    self.conn = sqlite3.connect(str(db_path))
    # ...
    self.conn.execute("PRAGMA journal_mode=WAL")
```

**Issues**:
1. **Thread Safety**: SQLite connections are not thread-safe. If used from multiple threads, it could cause corruption.
2. **Async Compatibility**: `sqlite3` is synchronous. If used in async contexts, it will block the event loop.
3. **Connection Lifecycle**: No explicit connection closing in error scenarios.

**Recommendation**:
1. Use `aiosqlite` for async SQLite operations if needed in async contexts.
2. Ensure connections are properly closed using context managers.
3. Consider connection pooling if multiple threads/processes access the database.
4. Document thread-safety requirements.

**Current Status**: The code does use context managers (`__enter__`/`__exit__`), which is good, but WAL mode with concurrent access needs careful handling.

---

### 5. LangGraph: Checkpoint Configuration

**Severity**: LOW
**Location**: `src/obsidian_anki_sync/agents/langgraph_orchestrator.py`

**Problem**:
Checkpointing is configured but thread_id is generated per note, which may not be optimal:

```python
self.checkpointer = MemorySaver()
self.app = self.workflow.compile(checkpointer=self.checkpointer)

# Later:
final_state = self.app.invoke(
    initial_state,
    config={"configurable": {"thread_id": f"note-{metadata.title}"}},
)
```

**Issues**:
1. **Memory Growth**: `MemorySaver` stores checkpoints in memory. With many notes, this could grow unbounded.
2. **No Cleanup**: No mechanism to clean up old checkpoints.
3. **Thread ID Collision**: Using note title as thread_id could cause collisions if titles are similar.

**Recommendation**:
1. Consider using a persistent checkpoint store (e.g., SQLite-based) for production.
2. Implement checkpoint cleanup/expiration.
3. Use more unique thread IDs (e.g., include timestamp or UUID).
4. Document checkpoint retention policy.

---

## Medium Priority Issues

### 6. PydanticAI: HTTP Client Configuration

**Severity**: LOW
**Location**: `src/obsidian_anki_sync/providers/pydantic_ai_models.py`

**Problem**:
PydanticAI models use default HTTP client without explicit configuration:

```python
model = OpenAIModel(
    model_name,
    base_url=base_url,
    api_key=api_key,
    http_client=None,  # Will use default httpx client
)
```

**Issues**:
1. **No Timeout Control**: Default client may have different timeout settings than desired.
2. **No Connection Pooling**: Default client may not have optimal connection pooling.
3. **No Retry Configuration**: Default client may not have retry logic configured.

**Recommendation**:
Explicitly create and configure an `httpx.AsyncClient`:

```python
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(30.0),
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
)
model = OpenAIModel(model_name, base_url=base_url, api_key=api_key, http_client=http_client)
```

---

### 7. Error Handling: Exception Types

**Severity**: LOW
**Location**: Multiple files

**Problem**:
Some exception handling catches generic `Exception`:

```python
except Exception as e:
    logger.error("langgraph_pre_validation_unexpected_error", error=str(e))
```

**Issues**:
1. **Overly Broad**: Catches all exceptions, including system exits and keyboard interrupts.
2. **Debugging Difficulty**: Makes it harder to identify specific failure modes.

**Recommendation**:
Catch specific exception types and let unexpected exceptions propagate:

```python
except (PreValidationError, StructuredOutputError, ModelError) as e:
    # Handle known errors
except BaseException:
    # Re-raise system exits, keyboard interrupts, etc.
    raise
except Exception as e:
    # Log unexpected errors but re-raise for debugging
    logger.exception("unexpected_error")
    raise
```

---

## Recommendations Summary

### Immediate Actions (Critical)

1. **Convert LangGraph nodes to async**: Change all node functions from sync to async and use `await` instead of `asyncio.run()`.
2. **Update workflow invocation**: Use `ainvoke()` instead of `invoke()` for async workflows.
3. **Test async compatibility**: Ensure all callers of the orchestrator can handle async operations.

### Short-term Improvements (High Priority)

1. **Model caching**: Create PydanticAI models once during initialization and reuse them.
2. **HTTP client consistency**: Standardize on async HTTP clients throughout the codebase.
3. **Connection management**: Review SQLite connection usage in async contexts.

### Long-term Enhancements (Medium Priority)

1. **Persistent checkpointing**: Replace `MemorySaver` with a persistent checkpoint store.
2. **Connection pooling**: Implement proper connection pooling for SQLite if needed.
3. **Error handling**: Refine exception handling to be more specific.

---

## Testing Recommendations

1. **Async Testing**: Add tests that verify async node functions work correctly.
2. **Event Loop Testing**: Test behavior when called from existing event loops.
3. **Concurrency Testing**: Test SQLite access from multiple threads/async contexts.
4. **Resource Leak Testing**: Verify that models and HTTP clients are properly cleaned up.

---

## References

- [LangGraph Async Nodes Documentation](https://langchain-ai.github.io/langgraph/how-tos/async/)
- [PydanticAI Best Practices](https://ai.pydantic.dev/)
- [httpx Async Client Guide](https://www.python-httpx.org/async/)
- [SQLite WAL Mode](https://www.sqlite.org/wal.html)

---

## Review Checklist

- [x] LangGraph async/await patterns reviewed
- [x] PydanticAI model creation reviewed
- [x] httpx client usage reviewed
- [x] SQLite connection management reviewed
- [x] Error handling patterns reviewed
- [x] Resource management reviewed
- [x] Documentation references added

---

**Next Steps**: Prioritize fixing critical async/await issues, then proceed with model caching and HTTP client standardization.

