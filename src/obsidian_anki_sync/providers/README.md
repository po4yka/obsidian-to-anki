# LLM Provider System - API Reference

This package provides a unified interface for multiple LLM providers, allowing seamless switching between Ollama, LM Studio, and OpenRouter.

## Architecture

```
BaseLLMProvider (Abstract)
    ├── OllamaProvider
    ├── LMStudioProvider
    └── OpenRouterProvider

ProviderFactory
    └── Creates provider instances based on configuration
```

## Quick Start

### Using the Factory

```python
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.providers import ProviderFactory

# Create provider from config
config = Config(...)
provider = ProviderFactory.create_from_config(config)

# Or create directly
provider = ProviderFactory.create_provider(
    "ollama",
    base_url="http://localhost:11434"
)

# Use the provider
response = provider.generate(
    model="qwen3:8b",
    prompt="What is the capital of France?",
    temperature=0.7
)
print(response["response"])
```

## BaseLLMProvider API

All providers implement this interface:

### Methods

#### `generate(model, prompt, system="", temperature=0.7, format="", stream=False)`

Generate a completion from the LLM.

**Parameters:**
- `model` (str): Model identifier
- `prompt` (str): User prompt/question
- `system` (str, optional): System prompt
- `temperature` (float): Sampling temperature (0.0-1.0)
- `format` (str): Response format ("json" for structured output)
- `stream` (bool): Enable streaming (not yet implemented)

**Returns:**
- `dict`: Response with at least `"response"` key containing generated text

**Example:**
```python
result = provider.generate(
    model="qwen3:8b",
    prompt="Explain quantum computing",
    system="You are a physics teacher",
    temperature=0.3
)
print(result["response"])
```

---

#### `generate_json(model, prompt, system="", temperature=0.7)`

Generate a JSON response from the LLM.

**Parameters:**
- `model` (str): Model identifier
- `prompt` (str): User prompt (should request JSON format)
- `system` (str, optional): System prompt
- `temperature` (float): Sampling temperature

**Returns:**
- `dict`: Parsed JSON response

**Example:**
```python
result = provider.generate_json(
    model="qwen3:8b",
    prompt="List 3 programming languages in JSON format with name and year",
    temperature=0.0
)
print(result)  # {"languages": [{"name": "Python", "year": 1991}, ...]}
```

---

#### `check_connection()`

Check if the provider is accessible and healthy.

**Returns:**
- `bool`: True if provider is accessible

**Example:**
```python
if provider.check_connection():
    print("Provider is ready")
else:
    print("Cannot connect to provider")
```

---

#### `list_models()`

List available models from the provider.

**Returns:**
- `list[str]`: List of model identifiers

**Example:**
```python
models = provider.list_models()
for model in models:
    print(f"Available: {model}")
```

---

#### `get_provider_name()`

Get the human-readable name of the provider.

**Returns:**
- `str`: Provider name (e.g., "Ollama", "LM Studio")

**Example:**
```python
print(f"Using provider: {provider.get_provider_name()}")
```

---

## Provider-Specific Details

### OllamaProvider

Supports both local and cloud Ollama deployments.

**Constructor:**
```python
OllamaProvider(
    base_url="http://localhost:11434",
    api_key=None,  # Only for Ollama Cloud
    timeout=120.0
)
```

**Additional Methods:**
- `pull_model(model: str) -> bool`: Pull a model from Ollama registry

**Example:**
```python
from obsidian_anki_sync.providers import OllamaProvider

# Local Ollama
provider = OllamaProvider(base_url="http://localhost:11434")

# Ollama Cloud
provider = OllamaProvider(
    base_url="https://api.ollama.com",
    api_key="your-api-key"
)

# Pull a model
if provider.pull_model("qwen3:8b"):
    print("Model downloaded successfully")
```

---

### LMStudioProvider

Provides OpenAI-compatible API for LM Studio.

**Constructor:**
```python
LMStudioProvider(
    base_url="http://localhost:1234/v1",
    timeout=120.0,
    max_tokens=2048
)
```

**Example:**
```python
from obsidian_anki_sync.providers import LMStudioProvider

provider = LMStudioProvider(
    base_url="http://localhost:1234/v1",
    max_tokens=4096
)

response = provider.generate(
    model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    prompt="Explain Python decorators",
    temperature=0.5
)
```

**Note:** Model identifier must match the loaded model in LM Studio.

---

### OpenRouterProvider

Provides access to multiple cloud LLM providers.

**Constructor:**
```python
OpenRouterProvider(
    api_key=None,  # Required, or set OPENROUTER_API_KEY env var
    base_url="https://openrouter.ai/api/v1",
    timeout=120.0,
    max_tokens=2048,
    site_url=None,  # Optional, for rankings
    site_name=None  # Optional, for rankings
)
```

**Example:**
```python
from obsidian_anki_sync.providers import OpenRouterProvider
import os

provider = OpenRouterProvider(
    api_key=os.environ["OPENROUTER_API_KEY"],
    site_url="https://myapp.com",
    site_name="My Application"
)

response = provider.generate(
    model="anthropic/claude-3-opus",
    prompt="Write a haiku about Python",
    temperature=0.8
)
```

**Available Models:** See https://openrouter.ai/models

---

## ProviderFactory

Factory class for creating provider instances.

### Methods

#### `create_provider(provider_type, **kwargs)`

Create a provider instance based on type.

**Parameters:**
- `provider_type` (str): Provider type ("ollama", "lm_studio", "openrouter")
- `**kwargs`: Provider-specific configuration

**Returns:**
- `BaseLLMProvider`: Initialized provider instance

**Example:**
```python
from obsidian_anki_sync.providers import ProviderFactory

# Create Ollama provider
provider = ProviderFactory.create_provider(
    "ollama",
    base_url="http://localhost:11434"
)

# Create LM Studio provider
provider = ProviderFactory.create_provider(
    "lm_studio",
    base_url="http://localhost:1234/v1",
    max_tokens=4096
)

# Create OpenRouter provider
provider = ProviderFactory.create_provider(
    "openrouter",
    api_key="your-api-key"
)
```

---

#### `create_from_config(config)`

Create a provider instance from a Config object.

**Parameters:**
- `config` (Config): Configuration object with provider settings

**Returns:**
- `BaseLLMProvider`: Initialized provider instance

**Example:**
```python
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.providers import ProviderFactory

config = Config(
    llm_provider="ollama",
    ollama_base_url="http://localhost:11434",
    # ... other config
)

provider = ProviderFactory.create_from_config(config)
```

---

#### `list_supported_providers()`

List all supported provider types.

**Returns:**
- `list[str]`: List of supported provider identifiers

**Example:**
```python
providers = ProviderFactory.list_supported_providers()
print(f"Supported providers: {', '.join(providers)}")
# Output: Supported providers: lm_studio, lmstudio, ollama, openrouter
```

---

## Error Handling

All providers raise standard exceptions:

```python
from obsidian_anki_sync.providers import ProviderFactory
import httpx

try:
    provider = ProviderFactory.create_provider("ollama")
    response = provider.generate(model="qwen3:8b", prompt="Hello")
except ValueError as e:
    print(f"Configuration error: {e}")
except httpx.HTTPError as e:
    print(f"Network error: {e}")
except ConnectionError as e:
    print(f"Provider connection failed: {e}")
```

---

## Integration with Agent System

The provider system integrates seamlessly with the agent orchestrator:

```python
from obsidian_anki_sync.agents import LangGraphOrchestrator
from obsidian_anki_sync.providers import ProviderFactory
from obsidian_anki_sync.config import Config

config = Config(...)

# Option 1: Let orchestrator create provider from config
orchestrator = LangGraphOrchestrator(config)

# Option 2: Provide custom provider
provider = ProviderFactory.create_provider(
    "lm_studio",
    base_url="http://192.168.1.100:1234/v1"
)
orchestrator = LangGraphOrchestrator(config, provider=provider)
```

---

## Backward Compatibility

The legacy `OllamaClient` class now inherits from `BaseLLMProvider`:

```python
from obsidian_anki_sync.agents.ollama_client import OllamaClient

# Old code still works
client = OllamaClient(base_url="http://localhost:11434")
result = client.generate(model="qwen3:8b", prompt="Hello")

# But it's now a BaseLLMProvider instance
assert isinstance(client, BaseLLMProvider)
```

**Note:** New code should use the provider system from `obsidian_anki_sync.providers`.

---

## Testing

### Mock Provider for Testing

```python
from obsidian_anki_sync.providers.base import BaseLLMProvider

class MockProvider(BaseLLMProvider):
    def generate(self, model, prompt, **kwargs):
        return {"response": "Mock response"}

    def check_connection(self):
        return True

    def list_models(self):
        return ["mock-model-1", "mock-model-2"]

# Use in tests
provider = MockProvider()
orchestrator = LangGraphOrchestrator(config, provider=provider)
```

---

## Contributing

To add a new provider:

1. **Create provider class:**
   ```python
   from obsidian_anki_sync.providers.base import BaseLLMProvider

   class MyProvider(BaseLLMProvider):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           # Initialize your provider

       def generate(self, model, prompt, **kwargs):
           # Implement generation
           pass

       def check_connection(self):
           # Implement health check
           pass

       def list_models(self):
           # Implement model listing
           pass
   ```

2. **Add to factory:**
   ```python
   # In factory.py
   PROVIDER_MAP = {
       "ollama": OllamaProvider,
       "lm_studio": LMStudioProvider,
       "openrouter": OpenRouterProvider,
       "myprovider": MyProvider,  # Add here
   }
   ```

3. **Update configuration:**
   ```python
   # In config.py
   @dataclass
   class Config:
       llm_provider: str = "ollama"
       # Add provider-specific settings
       myprovider_api_key: str = ""
   ```

4. **Add factory support:**
   ```python
   # In factory.py create_from_config()
   elif provider_type == "myprovider":
       kwargs = {
           "api_key": getattr(config, "myprovider_api_key", None),
           # ... other settings
       }
   ```

5. **Write tests and documentation**

---

## License

Same as parent project.
