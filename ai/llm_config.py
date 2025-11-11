"""
LLM Configuration Manager
Handles LLM model selection from environment variables
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMConfig:
    """Manages LLM configuration from environment variables"""

    # Default model preferences for different providers
    DEFAULT_MODELS = {
        'gemini': 'gemini-2.0-flash',
        'openai': 'gpt-4-turbo',
        'anthropic': 'claude-3-sonnet-20240229'
    }

    # Available models for each provider
    AVAILABLE_MODELS = {
        'gemini': {
            'gemini-2.0-flasexith': 'Fast, latest stable model',
            'gemini-2.5-flash': 'Latest version with improved capabilities',
            'gemini-2.0-flash-exp': 'Experimental features',
            'gemini-pro-latest': 'Pro model for complex analysis'
        },
        'openai': {
            'gpt-4-turbo': 'Best for complex analysis',
            'gpt-3.5-turbo': 'Fast and cost-effective'
        },
        'anthropic': {
            'claude-3-opus-20240229': 'Most capable model',
            'claude-3-sonnet-20240229': 'Balance of speed and capability'
        }
    }

    def __init__(self):
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load LLM configuration from environment variables"""
        config = {
            'primary_provider': os.getenv('LLM_PROVIDER', 'gemini').lower(),
            'preferred_model': os.getenv('LLM_MODEL', '').strip(),
            'fallback_enabled': os.getenv('LLM_FALLBACK_ENABLED', 'true').lower() == 'true',
            'max_retries': int(os.getenv('LLM_MAX_RETRIES', '3')),
            'retry_delay': float(os.getenv('LLM_RETRY_DELAY', '60'))
        }

        # Validate provider
        if config['primary_provider'] not in self.AVAILABLE_MODELS:
            print(f"Warning: Unknown provider '{config['primary_provider']}'. Using 'gemini' instead.")
            config['primary_provider'] = 'gemini'

        # Validate model if specified
        if config['preferred_model']:
            provider_models = self.AVAILABLE_MODELS[config['primary_provider']]
            if config['preferred_model'] not in provider_models:
                print(f"Warning: Model '{config['preferred_model']}' not available for {config['primary_provider']}.")
                config['preferred_model'] = self.DEFAULT_MODELS[config['primary_provider']]
        else:
            config['preferred_model'] = self.DEFAULT_MODELS[config['primary_provider']]

        return config

    def get_model_selection_order(self) -> list:
        """Get ordered list of models to try"""
        models_to_try = []

        # Add primary provider's preferred model first
        primary_provider = self.config['primary_provider']
        preferred_model = self.config['preferred_model']

        models_to_try.append((primary_provider, preferred_model))

        # Add other models from primary provider
        for model in self.AVAILABLE_MODELS[primary_provider]:
            if model != preferred_model:
                models_to_try.append((primary_provider, model))

        # Add fallback providers if enabled
        if self.config['fallback_enabled']:
            for provider in self.AVAILABLE_MODELS:
                if provider != primary_provider:
                    for model in self.AVAILABLE_MODELS[provider]:
                        models_to_try.append((provider, model))

        return models_to_try

    def get_model_info(self, provider: str, model: str) -> str:
        """Get description of a model"""
        if provider in self.AVAILABLE_MODELS and model in self.AVAILABLE_MODELS[provider]:
            return self.AVAILABLE_MODELS[provider][model]
        return "Unknown model"

    def list_all_options(self) -> Dict[str, Dict[str, str]]:
        """List all available model options"""
        return self.AVAILABLE_MODELS

    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()

    def print_config_help(self):
        """Print help for LLM configuration"""
        print("\n" + "=" * 60)
        print("LLM Configuration Options")
        print("=" * 60)
        print("\nEnvironment Variables:")
        print("-" * 40)
        print("LLM_PROVIDER     Primary LLM provider (gemini/openai/anthropic)")
        print("                 Default: gemini")
        print(f"                 Current: {self.config['primary_provider']}")
        print()
        print("LLM_MODEL        Preferred model for the provider")
        print("                 Default: gemini-1.5-flash")
        print(f"                 Current: {self.config['preferred_model']}")
        print()
        print("LLM_FALLBACK_ENABLED")
        print("                 Enable fallback to other providers")
        print("                 Default: true")
        print(f"                 Current: {self.config['fallback_enabled']}")
        print()
        print("LLM_MAX_RETRIES  Maximum retry attempts for rate limits")
        print("                 Default: 3")
        print(f"                 Current: {self.config['max_retries']}")
        print()
        print("LLM_RETRY_DELAY  Base delay for retries (seconds)")
        print("                 Default: 60")
        print(f"                 Current: {self.config['retry_delay']}")
        print()

        print("\nAvailable Models:")
        print("-" * 40)
        for provider, models in self.AVAILABLE_MODELS.items():
            print(f"\n{provider.upper()}:")
            for model, desc in models.items():
                status = "✓" if model == self.config['preferred_model'] and provider == self.config['primary_provider'] else " "
                print(f"  {status} {model:<25} - {desc}")

        print("\n" + "=" * 60)
        print("Example .env configuration:")
        print("-" * 40)
        print("# Primary configuration")
        print("LLM_PROVIDER=gemini")
        print("LLM_MODEL=gemini-1.5-pro")
        print()
        print("# API Keys (set at least one)")
        print("GEMINI_API_KEY=your_gemini_key_here")
        print("OPENAI_API_KEY=your_openai_key_here")
        print("ANTHROPIC_API_KEY=your_anthropic_key_here")
        print()
        print("# Optional settings")
        print("LLM_FALLBACK_ENABLED=true")
        print("LLM_MAX_RETRIES=3")
        print("LLM_RETRY_DELAY=60")
        print("=" * 60)


# Global config instance
llm_config = LLMConfig()