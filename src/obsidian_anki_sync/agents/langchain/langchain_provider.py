"""LangChain provider integration for the agent system.

This module wraps existing LLM providers (Ollama, LM Studio, OpenRouter)
and makes them compatible with LangChain's BaseChatModel interface.
"""

import json
from typing import Any, Iterator, Optional, cast

from obsidian_anki_sync.providers.base import BaseLLMProvider
from obsidian_anki_sync.providers.factory import ProviderFactory
from obsidian_anki_sync.utils.logging import get_logger

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Define dummy classes for type checking
    BaseChatModel = object  # type: ignore
    BaseMessage = object  # type: ignore
    ChatResult = object  # type: ignore
    AIMessage = object  # type: ignore

logger = get_logger(__name__)


class LangChainProviderAdapter(BaseChatModel):  # type: ignore
    """Adapter that wraps existing LLM providers for LangChain compatibility.

    This class implements the LangChain BaseChatModel interface and delegates
    to our existing provider system (Ollama, LM Studio, OpenRouter).

    Attributes:
        provider: The underlying LLM provider
        model_name: Name of the model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
    """

    provider: BaseLLMProvider
    model_name: str
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: float = 600.0

    def __init__(
        self,
        provider: BaseLLMProvider,
        model_name: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 600.0,
        **kwargs: Any,
    ):
        """Initialize the LangChain provider adapter.

        Args:
            provider: The underlying LLM provider
            model_name: Name of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to BaseChatModel
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with: pip install langchain-core"
            )

        # Initialize parent class
        super().__init__(**kwargs)

        # Set attributes
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        logger.info(
            "langchain_adapter_initialized",
            provider=type(provider).__name__,
            model=model_name,
            temperature=temperature,
        )

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "langchain_provider_adapter"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the LLM.

        Args:
            messages: List of messages (conversation history)
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional generation parameters

        Returns:
            ChatResult with generated response
        """
        # Convert LangChain messages to our format
        system_message = ""
        user_messages = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_message = msg.content
            elif isinstance(msg, HumanMessage):
                user_messages.append(msg.content)
            elif isinstance(msg, AIMessage):
                # For multi-turn conversations, we'd need to handle this
                # For now, just log it
                logger.debug("ai_message_in_history", content=msg.content[:100])

        # Combine user messages into a single prompt
        prompt = "\n\n".join(user_messages) if user_messages else ""

        # Extract format from kwargs if present
        format_type = kwargs.get("format", None)

        logger.debug(
            "langchain_generate",
            model=self.model_name,
            prompt_length=len(prompt),
            system_length=len(system_message),
        )

        # Call the underlying provider
        try:
            result = self.provider.generate(
                model=self.model_name,
                prompt=prompt,
                system=system_message or "",
                temperature=kwargs.get("temperature", self.temperature),
                format=format_type or "",
            )

            # Extract response text
            if isinstance(result, dict):
                response_text = result.get("response", "")
            else:
                response_text = str(result)

            # Create AIMessage
            ai_message = AIMessage(content=response_text)

            # Create ChatGeneration
            generation = ChatGeneration(message=ai_message)

            # Return ChatResult
            return ChatResult(generations=[generation])

        except Exception as e:
            logger.error(
                "langchain_generate_error",
                model=self.model_name,
                error=str(e),
            )
            raise

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """Stream generation is not implemented for this adapter."""
        raise NotImplementedError(
            "Streaming is not supported by this adapter. Use _generate instead."
        )

    def generate_json(
        self,
        messages: list[BaseMessage],
        schema: Optional[dict] = None,
        **kwargs: Any,
    ) -> dict:
        """Generate JSON response with optional schema validation.

        Args:
            messages: List of messages
            schema: Optional JSON schema for validation
            **kwargs: Additional generation parameters

        Returns:
            Parsed JSON response
        """
        # Request JSON format
        kwargs["format"] = "json"

        result = self._generate(messages, **kwargs)
        response_text = result.generations[0].message.content

        # Parse JSON
        try:
            json_response = json.loads(response_text)
            return cast(dict[Any, Any], json_response)
        except json.JSONDecodeError as e:
            logger.error(
                "json_parse_error",
                response=response_text[:200],
                error=str(e),
            )
            raise ValueError(f"Failed to parse JSON response: {e}")


def create_langchain_chat_model(
    config: Any,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """Create a LangChain-compatible chat model from config.

    Args:
        config: Application config object
        model_name: Optional model name override
        temperature: Optional temperature override

    Returns:
        LangChain BaseChatModel instance
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is not installed. Install with: pip install langchain-core"
        )

    # Create underlying provider using factory
    provider = ProviderFactory.create_from_config(config)

    # Determine model name
    final_model_name: str = model_name or str(
        getattr(config, "generator_model", "qwen3:14b")
    )

    # Determine temperature
    final_temperature: float = temperature or float(
        getattr(config, "llm_temperature", 0.2)
    )

    # Determine max tokens
    max_tokens: int = int(getattr(config, "llm_max_tokens", 1024))

    # Determine timeout
    timeout: float = float(getattr(config, "llm_timeout", 600.0))

    logger.info(
        "creating_langchain_chat_model",
        provider=config.llm_provider,
        model=final_model_name,
        temperature=final_temperature,
    )

    return LangChainProviderAdapter(
        provider=provider,
        model_name=final_model_name,
        temperature=final_temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def create_structured_llm(
    config: Any,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    output_schema: Optional[type] = None,
) -> BaseChatModel:
    """Create a LangChain LLM configured for structured output.

    Args:
        config: Application config object
        model_name: Optional model name override
        temperature: Optional temperature override
        output_schema: Optional Pydantic model for structured output

    Returns:
        LangChain BaseChatModel configured for structured output
    """
    chat_model = create_langchain_chat_model(config, model_name, temperature)

    # If output schema provided and LangChain supports it, bind it
    if output_schema and hasattr(chat_model, "with_structured_output"):
        return chat_model.with_structured_output(output_schema)

    return chat_model
