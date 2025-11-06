# LLM generation with OpenAI and Ollama fallback

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import uuid
from datetime import datetime

from src.retrieval.retriever import RetrievedChunk
from src.generator.prompt_manager import PromptManager
from src.config import settings
from src.cache import cache_manager
from src.utils.circuit_breaker import get_circuit_breaker, CircuitBreakerOpenError

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    # response from LLM with metadata
    answer: str
    sources: List[str]
    trace_id: str
    token_usage: Dict[str, int]
    prompt_version: str
    model_used: str
    cost_estimate: float = 0.0


class LLMGenerator:
    # generates responses using LLM (OpenAI primary, Ollama fallback)
    
    def __init__(self):
        self.prompt_manager = PromptManager()
        self.openai_client = None
        self._initialize_openai()
    
    def _initialize_openai(self):
        # initialize OpenAI client
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
            if not self.openai_client.api_key:
                raise ValueError("OpenAI API key not found in settings")
            logger.info("OpenAI client initialized")
        except ImportError:
            logger.warning("openai package not installed")
            self.openai_client = None
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI: {e}")
            self.openai_client = None
    
    def generate(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        prompt_version: str = 'v1',
        model_type: str = 'openai',
        fallback: bool = True,
        use_cache: bool = True
    ) -> LLMResponse:
        # generate response from LLM using query and context chunks
        
        # Check cache first
        if use_cache:
            cached_response = cache_manager.get_llm_response(query)
            if cached_response:
                # Reconstruct LLMResponse from cached dict
                response = LLMResponse(**cached_response)
                logger.info(f"Cache hit for LLM response: {query[:50]}...")
                return response
        
        # load prompt template
        try:
            template = self.prompt_manager.load_prompt_template(prompt_version)
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            raise
        
        # format prompt with context
        context_text = self._format_context(context_chunks)
        sources = [chunk.chunk_id for chunk in context_chunks]
        
        prompt = template.format(
            query=query,
            context=context_text
        )
        
        # generate trace ID
        trace_id = str(uuid.uuid4())
        
        # call LLM with circuit breaker protection
        try:
            if model_type == 'openai' or (model_type == 'auto' and self.openai_client):
                circuit_breaker = get_circuit_breaker("openai")
                try:
                    with circuit_breaker:
                        response = self._call_openai(prompt, trace_id)
                except CircuitBreakerOpenError:
                    logger.warning("OpenAI circuit breaker is OPEN. Falling back to Ollama...")
                    if fallback:
                        response = self._call_ollama(prompt, trace_id)
                    else:
                        raise
            elif model_type == 'ollama' or (fallback and model_type == 'auto'):
                circuit_breaker = get_circuit_breaker("ollama")
                try:
                    with circuit_breaker:
                        response = self._call_ollama(prompt, trace_id)
                except CircuitBreakerOpenError:
                    logger.warning("Ollama circuit breaker is OPEN.")
                    raise
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # add prompt version and sources
            response.prompt_version = prompt_version
            response.sources = sources
            
            # Cache response
            if use_cache:
                cache_manager.cache_llm_response(query, {
                    "answer": response.answer,
                    "sources": response.sources,
                    "trace_id": response.trace_id,
                    "token_usage": response.token_usage,
                    "prompt_version": response.prompt_version,
                    "model_used": response.model_used,
                    "cost_estimate": response.cost_estimate
                })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            if fallback and model_type == 'openai':
                logger.info("Falling back to Ollama...")
                try:
                    response = self._call_ollama(prompt, trace_id)
                    response.prompt_version = prompt_version
                    response.sources = sources
                    return response
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise
            raise
    
    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        # format retrieved chunks into context string
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[{i}] {chunk.text}")
        
        return "\n\n".join(context_parts)
    
    def _call_openai(self, prompt: str, trace_id: str) -> LLMResponse:
        # call OpenAI API
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # extract response
            answer = response.choices[0].message.content
            
            # calculate token usage
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # estimate cost (rough estimates for GPT-4)
            cost_estimate = self._estimate_openai_cost(token_usage)
            
            return LLMResponse(
                answer=answer,
                sources=[],
                trace_id=trace_id,
                token_usage=token_usage,
                prompt_version="",
                model_used=settings.openai_model,
                cost_estimate=cost_estimate
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _call_ollama(self, prompt: str, trace_id: str) -> LLMResponse:
        # call Ollama API (local LLM)
        try:
            import requests
            
            ollama_url = f"{settings.ollama_base_url}/api/generate"
            
            response = requests.post(
                ollama_url,
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise ValueError(f"Ollama API error: {response.status_code}")
            
            data = response.json()
            answer = data.get("response", "")
            
            # Ollama doesn't provide detailed token counts, estimate
            token_usage = {
                "prompt_tokens": len(prompt.split()),  # rough estimate
                "completion_tokens": len(answer.split()),
                "total_tokens": len(prompt.split()) + len(answer.split())
            }
            
            # Ollama is free (local)
            cost_estimate = 0.0
            
            return LLMResponse(
                answer=answer,
                sources=[],
                trace_id=trace_id,
                token_usage=token_usage,
                prompt_version="",
                model_used=settings.ollama_model,
                cost_estimate=cost_estimate
            )
            
        except ImportError:
            raise ValueError("requests package required for Ollama")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def _estimate_openai_cost(self, token_usage: Dict[str, int]) -> float:
        # estimate cost based on token usage (rough estimates)
        # GPT-4: ~$0.03 per 1K prompt tokens, ~$0.06 per 1K completion tokens
        # GPT-3.5-turbo: ~$0.0015 per 1K prompt tokens, ~$0.002 per 1K completion tokens
        
        model = settings.openai_model.lower()
        
        if "gpt-4" in model:
            prompt_cost = (token_usage["prompt_tokens"] / 1000) * 0.03
            completion_cost = (token_usage["completion_tokens"] / 1000) * 0.06
        elif "gpt-3.5" in model or "gpt-35" in model:
            prompt_cost = (token_usage["prompt_tokens"] / 1000) * 0.0015
            completion_cost = (token_usage["completion_tokens"] / 1000) * 0.002
        else:
            # default to GPT-3.5 pricing
            prompt_cost = (token_usage["prompt_tokens"] / 1000) * 0.0015
            completion_cost = (token_usage["completion_tokens"] / 1000) * 0.002
        
        return prompt_cost + completion_cost

