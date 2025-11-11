"""
Model selector for different LLM providers
Allows switching between different models based on use case and availability
"""

import os
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ModelSelector:
    """Handles different LLM models for stock analysis"""

    # Model configurations
    MODELS = {
        'gemini': {
            'provider': 'google',
            'models': {
                'gemini-1.5-flash': {
                    'name': 'gemini-1.5-flash',
                    'description': 'Fast, good for quick analysis',
                    'strengths': ['speed', 'free_tier'],
                    'max_tokens': 8192,
                    'recommended_for': ['quick_analysis', 'narrative']
                },
                'gemini-1.5-pro': {
                    'name': 'gemini-1.5-pro',
                    'description': 'Better analysis, larger context',
                    'strengths': ['analysis_quality', 'context_window'],
                    'max_tokens': 2097152,  # 2M tokens
                    'recommended_for': ['deep_analysis', 'research']
                },
                'gemini-2.0-flash-exp': {
                    'name': 'gemini-2.0-flash-exp',
                    'description': 'Latest experimental model',
                    'strengths': ['latest_features'],
                    'max_tokens': 8192,
                    'recommended_for': ['experimental']
                }
            }
        },
        'openai': {
            'provider': 'openai',
            'models': {
                'gpt-4-turbo': {
                    'name': 'gpt-4-turbo',
                    'description': 'Best for complex analysis',
                    'strengths': ['reasoning', 'financial_expertise'],
                    'max_tokens': 4096,
                    'recommended_for': ['complex_analysis', 'strategy']
                },
                'gpt-3.5-turbo': {
                    'name': 'gpt-3.5-turbo',
                    'description': 'Fast and cost-effective',
                    'strengths': ['speed', 'cost'],
                    'max_tokens': 4096,
                    'recommended_for': ['quick_insights']
                }
            }
        },
        'claude': {
            'provider': 'anthropic',
            'models': {
                'claude-3-opus': {
                    'name': 'claude-3-opus-20240229',
                    'description': 'Most capable model',
                    'strengths': ['analysis', 'nuance'],
                    'max_tokens': 4096,
                    'recommended_for': ['detailed_research', 'risk_analysis']
                },
                'claude-3-sonnet': {
                    'name': 'claude-3-sonnet-20240229',
                    'description': 'Balance of speed and capability',
                    'strengths': ['balance', 'reliability'],
                    'max_tokens': 4096,
                    'recommended_for': ['general_analysis']
                }
            }
        }
    }

    @classmethod
    def get_best_model_for_usecase(cls, usecase: str) -> Dict[str, Any]:
        """Get the best model for a specific use case"""

        # Priority order for different use cases
        priorities = {
            'narrative': [
                ('gemini', 'gemini-1.5-pro'),
                ('gemini', 'gemini-1.5-flash'),
                ('openai', 'gpt-4-turbo'),
                ('claude', 'claude-3-sonnet')
            ],
            'quick_analysis': [
                ('gemini', 'gemini-1.5-flash'),
                ('openai', 'gpt-3.5-turbo'),
                ('claude', 'claude-3-sonnet')
            ],
            'deep_analysis': [
                ('claude', 'claude-3-opus'),
                ('openai', 'gpt-4-turbo'),
                ('gemini', 'gemini-1.5-pro')
            ],
            'indonesian_stocks': [
                # Gemini generally better with Indonesian language
                ('gemini', 'gemini-1.5-pro'),
                ('gemini', 'gemini-1.5-flash'),
                ('openai', 'gpt-4-turbo')
            ]
        }

        usecase_priorities = priorities.get(usecase, priorities['quick_analysis'])

        for provider, model_name in usecase_priorities:
            if cls.is_model_available(provider, model_name):
                return {
                    'provider': provider,
                    'model_name': model_name,
                    **cls.MODELS[provider]['models'][model_name]
                }

        # Fallback to available model
        return cls.get_first_available()

    @classmethod
    def is_model_available(cls, provider: str, model_name: str) -> bool:
        """Check if a model is available based on API keys"""

        if provider == 'gemini':
            return os.getenv('GEMINI_API_KEY') is not None
        elif provider == 'openai':
            return os.getenv('OPENAI_API_KEY') is not None
        elif provider == 'claude':
            return os.getenv('ANTHROPIC_API_KEY') is not None

        return False

    @classmethod
    def get_first_available(cls) -> Dict[str, Any]:
        """Get the first available model"""

        # Check in order of preference
        preference_order = [
            ('gemini', 'gemini-1.5-flash'),
            ('gemini', 'gemini-1.5-pro'),
            ('openai', 'gpt-4-turbo'),
            ('openai', 'gpt-3.5-turbo'),
            ('claude', 'claude-3-sonnet')
        ]

        for provider, model_name in preference_order:
            if cls.is_model_available(provider, model_name):
                return {
                    'provider': provider,
                    'model_name': model_name,
                    **cls.MODELS[provider]['models'][model_name]
                }

        return {}

    @classmethod
    def get_model_info(cls, provider: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        try:
            return {
                'provider': provider,
                'model_name': model_name,
                **cls.MODELS[provider]['models'][model_name]
            }
        except KeyError:
            return None

    @classmethod
    def list_available_models(cls) -> Dict[str, Any]:
        """List all available models based on API keys"""
        available = {}

        for provider, provider_config in cls.MODELS.items():
            if cls.is_model_available(provider, list(provider_config['models'].keys())[0]):
                available[provider] = {
                    'name': provider,
                    'models': provider_config['models']
                }

        return available


def get_recommended_model_config() -> Dict[str, Any]:
    """Get recommended model configuration for Indonesian stock analysis"""

    # For Indonesian stocks, prioritize models that handle Bahasa well
    model = ModelSelector.get_best_model_for_usecase('indonesian_stocks')

    if not model:
        logger.warning("No models available. Please set API keys.")
        return {}

    # Add model-specific settings
    if model['provider'] == 'gemini':
        return {
            **model,
            'settings': {
                'temperature': 0.5,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 2000
            }
        }
    elif model['provider'] == 'openai':
        return {
            **model,
            'settings': {
                'temperature': 0.3,
                'max_tokens': 2000,
                'presence_penalty': 0.1,
                'frequency_penalty': 0.1
            }
        }
    elif model['provider'] == 'claude':
        return {
            **model,
            'settings': {
                'temperature': 0.5,
                'max_tokens': 2000
            }
        }

    return model