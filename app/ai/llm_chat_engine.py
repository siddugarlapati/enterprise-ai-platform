"""
Advanced LLM Chat Engine with OpenAI, Anthropic, and local model support.
"""
import logging
from typing import List, Dict, Any, Optional, AsyncIterator
from enum import Enum
import asyncio
from functools import lru_cache

from app.config.settings import settings

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LLMChatEngine:
    """Advanced LLM engine supporting multiple providers."""
    
    def __init__(self):
        self._openai_client = None
        self._anthropic_client = None
        self.default_provider = LLMProvider.OPENAI
        
    def _get_openai_client(self):
        """Lazy load OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import AsyncOpenAI
                api_key = getattr(settings, 'OPENAI_API_KEY', None)
                if api_key:
                    self._openai_client = AsyncOpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        return self._openai_client
    
    def _get_anthropic_client(self):
        """Lazy load Anthropic client."""
        if self._anthropic_client is None:
            try:
                from anthropic import AsyncAnthropic
                api_key = getattr(settings, 'ANTHROPIC_API_KEY', None)
                if api_key:
                    self._anthropic_client = AsyncAnthropic(api_key=api_key)
                    logger.info("Anthropic client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic: {e}")
        return self._anthropic_client
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate chat completion.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            provider: LLM provider to use
            model: Specific model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
        """
        provider = provider or self.default_provider
        
        if provider == LLMProvider.OPENAI:
            return await self._openai_chat(messages, model, temperature, max_tokens, stream)
        elif provider == LLMProvider.ANTHROPIC:
            return await self._anthropic_chat(messages, model, temperature, max_tokens, stream)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _openai_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        stream: bool
    ) -> Dict[str, Any]:
        """OpenAI chat completion."""
        client = self._get_openai_client()
        if not client:
            raise ValueError("OpenAI client not configured")
        
        model = model or "gpt-4-turbo-preview"
        
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                return {"stream": response}
            
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise
    
    async def _anthropic_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        stream: bool
    ) -> Dict[str, Any]:
        """Anthropic chat completion."""
        client = self._get_anthropic_client()
        if not client:
            raise ValueError("Anthropic client not configured")
        
        model = model or "claude-3-opus-20240229"
        
        # Convert messages format
        system_msg = None
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                formatted_messages.append(msg)
        
        try:
            response = await client.messages.create(
                model=model,
                messages=formatted_messages,
                system=system_msg,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                return {"stream": response}
            
            return {
                "content": response.content[0].text,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                "finish_reason": response.stop_reason
            }
        except Exception as e:
            logger.error(f"Anthropic chat error: {e}")
            raise
    
    async def function_calling(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]],
        provider: Optional[LLMProvider] = None
    ) -> Dict[str, Any]:
        """
        Function calling for structured outputs.
        
        Args:
            messages: Conversation messages
            functions: List of function definitions
            provider: LLM provider
        """
        provider = provider or LLMProvider.OPENAI
        
        if provider != LLMProvider.OPENAI:
            raise ValueError("Function calling only supported for OpenAI")
        
        client = self._get_openai_client()
        if not client:
            raise ValueError("OpenAI client not configured")
        
        try:
            response = await client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                functions=functions,
                function_call="auto"
            )
            
            choice = response.choices[0]
            
            if choice.message.function_call:
                return {
                    "type": "function_call",
                    "function_name": choice.message.function_call.name,
                    "arguments": choice.message.function_call.arguments,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            
            return {
                "type": "message",
                "content": choice.message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Function calling error: {e}")
            raise
    
    async def summarize(
        self,
        text: str,
        max_length: int = 200,
        provider: Optional[LLMProvider] = None
    ) -> str:
        """Summarize text."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
            {"role": "user", "content": f"Summarize the following text in {max_length} words or less:\n\n{text}"}
        ]
        
        result = await self.chat_completion(messages, provider=provider, max_tokens=max_length * 2)
        return result["content"]
    
    async def extract_structured_data(
        self,
        text: str,
        schema: Dict[str, Any],
        provider: Optional[LLMProvider] = None
    ) -> Dict[str, Any]:
        """Extract structured data from text using function calling."""
        import json
        
        functions = [{
            "name": "extract_data",
            "description": "Extract structured data from text",
            "parameters": schema
        }]
        
        messages = [
            {"role": "system", "content": "Extract the requested information from the text."},
            {"role": "user", "content": text}
        ]
        
        result = await self.function_calling(messages, functions, provider)
        
        if result["type"] == "function_call":
            return json.loads(result["arguments"])
        
        return {}
    
    async def translate(
        self,
        text: str,
        target_language: str,
        provider: Optional[LLMProvider] = None
    ) -> str:
        """Translate text to target language."""
        messages = [
            {"role": "system", "content": f"You are a professional translator. Translate the following text to {target_language}."},
            {"role": "user", "content": text}
        ]
        
        result = await self.chat_completion(messages, provider=provider)
        return result["content"]
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        client = self._get_openai_client()
        if not client:
            raise ValueError("OpenAI client not configured")
        
        try:
            response = await client.embeddings.create(
                model=model,
                input=texts
            )
            
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise
    
    def get_available_models(self, provider: LLMProvider) -> List[str]:
        """Get list of available models for provider."""
        models = {
            LLMProvider.OPENAI: [
                "gpt-4-turbo-preview",
                "gpt-4",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k"
            ],
            LLMProvider.ANTHROPIC: [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ]
        }
        return models.get(provider, [])


# Singleton instance
llm_chat_engine = LLMChatEngine()
