"""
Multi-LLM Adapter for Stock Analysis
Supports multiple AI providers: Gemini, OpenAI, Anthropic
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class BaseLLMAdapter(ABC):
    """Base class for all LLM adapters"""

    def __init__(self, max_retries: int = 3, base_delay: float = 60):
        self.max_retries = max_retries
        self.base_delay = base_delay

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available"""
        pass


class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini adapter"""

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self._initialize()

    def _initialize(self):
        try:
            import google.generativeai as genai
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                logger.warning("GEMINI_API_KEY not found")
                return

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini initialized with model: {self.model_name}")
        except ImportError:
            logger.error("google-generativeai not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        if not self.model:
            return None

        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=kwargs.get('generation_config')
                )
                return response.text if response and response.text else None
            except Exception as e:
                error_str = str(e).lower()

                if 'quota' in error_str or 'rate limit' in error_str or '429' in error_str:
                    if attempt < self.max_retries - 1:
                        retry_delay = self.base_delay * (2 ** attempt)

                        if 'retry in' in error_str:
                            import re
                            match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str)
                            if match:
                                retry_delay = float(match.group(1))

                        logger.warning(f"Gemini rate limit, retrying in {retry_delay:.0f}s... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Max retries exceeded for Gemini: {e}")
                        return None

                logger.error(f"Gemini error: {e}")
                return None

        return None

    def is_available(self) -> bool:
        return self.model is not None


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI GPT adapter"""

    def __init__(self, model_name: str = "gpt-4-turbo"):
        super().__init__()
        self.model_name = model_name
        self.client = None
        self._initialize()

    def _initialize(self):
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OPENAI_API_KEY not found")
                return

            self.client = openai.OpenAI(api_key=api_key)
            logger.info(f"OpenAI initialized with model: {self.model_name}")
        except ImportError:
            logger.error("openai not installed")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        if not self.client:
            return None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional stock analyst for Indonesian stocks."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=kwargs.get('temperature', 0.5),
                    max_tokens=kwargs.get('max_tokens', 2000)
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e).lower()

                if 'rate' in error_str and 'limit' in error_str:
                    if attempt < self.max_retries - 1:
                        retry_delay = self.base_delay * (2 ** attempt)
                        logger.warning(f"OpenAI rate limit, retrying in {retry_delay:.0f}s... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Max retries exceeded for OpenAI: {e}")
                        return None

                logger.error(f"OpenAI error: {e}")
                return None

        return None

    def is_available(self) -> bool:
        return self.client is not None


class AnthropicAdapter(BaseLLMAdapter):
    """Anthropic Claude adapter"""

    def __init__(self, model_name: str = "claude-3-sonnet-20240229"):
        super().__init__()
        self.model_name = model_name
        self.client = None
        self._initialize()

    def _initialize(self):
        try:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not found")
                return

            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Anthropic initialized with model: {self.model_name}")
        except ImportError:
            logger.error("anthropic not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        if not self.client:
            return None

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=kwargs.get('max_tokens', 2000),
                    temperature=kwargs.get('temperature', 0.5),
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text if response.content else None
            except Exception as e:
                error_str = str(e).lower()

                if 'rate' in error_str and 'limit' in error_str:
                    if attempt < self.max_retries - 1:
                        retry_delay = self.base_delay * (2 ** attempt)
                        logger.warning(f"Anthropic rate limit, retrying in {retry_delay:.0f}s... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Max retries exceeded for Anthropic: {e}")
                        return None

                logger.error(f"Anthropic error: {e}")
                return None

        return None

    def is_available(self) -> bool:
        return self.client is not None


class LLMManager:
    """Manages multiple LLM adapters and selects the best one"""

    def __init__(self):
        # Import config after environment is loaded
        from .llm_config import llm_config

        # Initialize all adapters
        self.adapters = {
            'gemini_flash': GeminiAdapter('gemini-2.0-flash'),
            'gemini_pro': GeminiAdapter('gemini-pro-latest'),
            'gemini_experimental': GeminiAdapter('gemini-2.0-flash-exp'),
            'openai_gpt4': OpenAIAdapter('gpt-4-turbo'),
            'openai_gpt35': OpenAIAdapter('gpt-3.5-turbo'),
            'claude_opus': AnthropicAdapter('claude-3-opus-20240229'),
            'claude_sonnet': AnthropicAdapter('claude-3-sonnet-20240229')
        }

        # Get model selection order from config
        self.config = llm_config
        self.model_selection_order = self.config.get_model_selection_order()

    def get_best_adapter(self) -> Optional[BaseLLMAdapter]:
        """Get the best available adapter based on configuration"""
        for provider, model in self.model_selection_order:
            # Map provider-model to adapter key
            adapter_key = self._get_adapter_key(provider, model)
            if adapter_key in self.adapters:
                adapter = self.adapters[adapter_key]
                if adapter.is_available():
                    logger.info(f"Using {provider}/{model} for analysis")
                    return adapter

        logger.error("No LLM adapter available")
        return None

    def _get_adapter_key(self, provider: str, model: str) -> str:
        """Convert provider-model to adapter key"""
        mapping = {
            'gemini': {
                'gemini-2.0-flash': 'gemini_flash',
                'gemini-2.5-flash': 'gemini_flash',
                'gemini-pro-latest': 'gemini_pro',
                'gemini-2.0-flash-exp': 'gemini_experimental',
                'gemini-1.5-flash': 'gemini_flash',  # Fallback for old config
                'gemini-1.5-pro': 'gemini_pro'  # Fallback for old config
            },
            'openai': {
                'gpt-4-turbo': 'openai_gpt4',
                'gpt-3.5-turbo': 'openai_gpt35'
            },
            'anthropic': {
                'claude-3-opus-20240229': 'claude_opus',
                'claude-3-sonnet-20240229': 'claude_sonnet'
            }
        }
        return mapping.get(provider, {}).get(model, f"{provider}_{model}")

    def get_adapter_by_name(self, name: str) -> Optional[BaseLLMAdapter]:
        """Get a specific adapter by name"""
        return self.adapters.get(name)

    def list_available(self) -> List[str]:
        """List all available adapters"""
        available = []
        for name, adapter in self.adapters.items():
            if adapter.is_available():
                available.append(name)
        return available

    def get_current_config(self) -> Dict[str, Any]:
        """Get current LLM configuration"""
        adapter = self.get_best_adapter()
        provider_model = "None"
        if adapter:
            # Find which provider/model is being used
            for provider, model in self.model_selection_order:
                adapter_key = self._get_adapter_key(provider, model)
                if adapter_key in self.adapters and self.adapters[adapter_key] == adapter:
                    provider_model = f"{provider}/{model}"
                    break

        return {
            'provider_model': provider_model,
            'config': self.config.get_current_config(),
            'available_models': self.list_available()
        }

    def print_config_help(self):
        """Print configuration help"""
        self.config.print_config_help()


# Global instance
llm_manager = LLMManager()