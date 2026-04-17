"""Groq API client for making LLM requests."""

import httpx
import asyncio
import logging
from typing import List, Dict, Any, Optional
from .config import GROQ_API_KEY, GROQ_API_URL

logger = logging.getLogger(__name__)

# Global persistent client - created once, reused across all requests
_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    """Get or create the persistent async HTTP client."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=20),
        )
    return _client


async def close_client():
    """Close the persistent client gracefully on shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via Groq API with retry logic.

    Args:
        model: Groq model identifier
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        max_retries: Number of retry attempts for transient failures

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
    }

    client = get_client()

    for attempt in range(max_retries):
        try:
            response = await client.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                logger.warning(f"Empty choices from {model}")
                return None

            message = choices[0].get("message", {})
            content = message.get("content")
            if not content:
                logger.warning(f"Empty content from {model}")
                return None

            return {
                "content": content,
                "reasoning_details": message.get("reasoning_details"),
            }

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in (429, 500, 502, 503) and attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    f"HTTP {status} from {model}, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait)
                continue
            logger.error(f"HTTP {status} error querying {model}: {e.response.text[:300]}")
            return None

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    f"Connection error from {model}: {e}, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait)
                continue
            logger.error(f"Failed to reach {model} after {max_retries} attempts: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error querying {model}: {e}")
            return None

    return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of Groq model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    tasks = [query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}
