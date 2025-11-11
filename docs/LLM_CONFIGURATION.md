# LLM Configuration Guide

This guide explains how to configure and use different LLM models for stock analysis.

## Environment Variables

Add these variables to your `.env` file in the project root:

### Primary Configuration

```bash
# Primary LLM provider (gemini, openai, anthropic)
LLM_PROVIDER=gemini

# Preferred model for the provider
LLM_MODEL=gemini-1.5-pro

# Enable fallback to other providers when primary fails
LLM_FALLBACK_ENABLED=true

# Retry settings for rate limits
LLM_MAX_RETRIES=3
LLM_RETRY_DELAY=60
```

### API Keys (set at least one)

```bash
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude API
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Available Models

### Google Gemini
- `gemini-2.0-flash` - Fast, latest stable model (recommended)
- `gemini-2.5-flash` - Latest version with improved capabilities
- `gemini-pro-latest` - Pro model for complex analysis
- `gemini-2.0-flash-exp` - Experimental with latest features

### OpenAI
- `gpt-4-turbo` - Best for complex financial analysis
- `gpt-3.5-turbo` - Fast and cost-effective

### Anthropic Claude
- `claude-3-opus-20240229` - Most capable model
- `claude-3-sonnet-20240229` - Balance of speed and capability

## Configuration Examples

### Example 1: Use Gemini 2.0 Flash (Best for Indonesian)
```bash
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
GEMINI_API_KEY=your_key_here
```

### Example 2: Use OpenAI GPT-4 for Premium Analysis
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo
OPENAI_API_KEY=your_key_here
LLM_FALLBACK_ENABLED=true  # Will fall back to Gemini if OpenAI fails
```

### Example 3: Use Claude Opus with Gemini Fallback
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-opus-20240229
ANTHROPIC_API_KEY=your_key_here
GEMINI_API_KEY=your_gemini_key  # Fallback
LLM_FALLBACK_ENABLED=true
```

## Model Selection Priority

When `LLM_FALLBACK_ENABLED=true`, the system will try models in this order:

1. **Primary Provider & Model** (from LLM_PROVIDER and LLM_MODEL)
2. **Other models from primary provider**
3. **Models from other providers** (if API keys are available)

## CLI Commands

### Check Current Configuration
```bash
llm config
```

### Show Configuration Help
```bash
llm help
```

## Getting API Keys

### Google Gemini API
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy and paste to `.env` file

### OpenAI API
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy and paste to `.env` file

### Anthropic API
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create a new API key
3. Copy and paste to `.env` file

## Recommendations

### For Indonesian Stock Analysis
1. **Primary Choice**: Gemini 2.0 Flash
   - Fast, latest stable model
   - Good Indonesian language support
   - Balanced performance and cost

2. **Premium Option**: Gemini Pro Latest
   - Best for complex analysis
   - Large context window for detailed analysis
   - Higher quality but more expensive

3. **Experimental**: Gemini 2.5 Flash
   - Latest capabilities
   - Cutting-edge features
   - May have variable performance

4. **Premium Option**: GPT-4 Turbo
   - Superior financial analysis
   - Better at complex reasoning
   - Higher cost but excellent quality

### Rate Limits
- Gemini 2.0: 15 requests/minute (free tier)
- OpenAI: Varies by plan
- Anthropic: Varies by plan

The system automatically handles rate limits with retries and exponential backoff.

## Troubleshooting

### Model Not Available
- Check API key is correct
- Verify model name spelling
- Run `llm config` to see available models

### Rate Limit Errors
- Wait for automatic retry (up to 3 attempts)
- Consider switching to a different provider
- Upgrade API plan for higher limits

### Poor Analysis Quality
- Try switching to a more capable model
- Ensure sufficient data (500+ candles for ML)
- Check if narrative mode is enabled

## Default Configuration

If no environment variables are set, the system defaults to:
- Provider: `gemini`
- Model: `gemini-2.0-flash`
- Fallback: Enabled
- Retries: 3
- Delay: 60 seconds