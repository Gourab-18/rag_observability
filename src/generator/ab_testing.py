"""
A/B testing framework for prompt versions.
Compares different prompt versions side-by-side and logs metrics.
"""

import json
import logging
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from src.generator import LLMGenerator, LLMResponse
from src.retrieval.retriever import RetrievedChunk
from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Result of A/B test comparison."""
    query_id: str
    query: str
    prompt_version_a: str
    prompt_version_b: str
    response_a: LLMResponse
    response_b: LLMResponse
    metrics_a: Dict[str, Any]
    metrics_b: Dict[str, Any]
    timestamp: str
    winner: Optional[str] = None  # "a", "b", or None for tie


class ABTester:
    """
    A/B testing manager for comparing prompt versions.
    
    Runs queries with multiple prompt versions and compares:
    - Response quality (length, structure)
    - Token usage and cost
    - Latency
    - Source citation quality
    """
    
    def __init__(self, results_dir: str = "data/ab_tests"):
        self.generator = LLMGenerator()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _calculate_metrics(self, response: LLMResponse, latency: float) -> Dict[str, Any]:
        """
        Calculate metrics for a response.
        
        Args:
            response: LLM response
            latency: Response latency in seconds
            
        Returns:
            Metrics dictionary
        """
        return {
            "answer_length": len(response.answer),
            "num_sources": len(response.sources),
            "token_usage": response.token_usage,
            "cost_estimate": response.cost_estimate,
            "latency": latency,
            "model_used": response.model_used,
            "prompt_version": response.prompt_version,
        }
    
    def _determine_winner(
        self,
        metrics_a: Dict[str, Any],
        metrics_b: Dict[str, Any]
    ) -> Optional[str]:
        """
        Determine winner based on metrics.
        
        Prefers:
        - Lower cost
        - Lower latency
        - More sources (better citations)
        - Reasonable answer length
        
        Args:
            metrics_a: Metrics for version A
            metrics_b: Metrics for version B
            
        Returns:
            "a", "b", or None for tie
        """
        # Score based on multiple factors
        score_a = 0
        score_b = 0
        
        # Cost (lower is better)
        if metrics_a["cost_estimate"] < metrics_b["cost_estimate"]:
            score_a += 2
        elif metrics_b["cost_estimate"] < metrics_a["cost_estimate"]:
            score_b += 2
        
        # Latency (lower is better)
        if metrics_a["latency"] < metrics_b["latency"]:
            score_a += 2
        elif metrics_b["latency"] < metrics_a["latency"]:
            score_b += 2
        
        # Sources (more is better, up to a point)
        if metrics_a["num_sources"] > metrics_b["num_sources"]:
            score_a += 1
        elif metrics_b["num_sources"] > metrics_a["num_sources"]:
            score_b += 1
        
        # Answer length (reasonable length is better)
        len_a = metrics_a["answer_length"]
        len_b = metrics_b["answer_length"]
        if 100 <= len_a <= 1000 and not (100 <= len_b <= 1000):
            score_a += 1
        elif 100 <= len_b <= 1000 and not (100 <= len_a <= 1000):
            score_b += 1
        
        if score_a > score_b:
            return "a"
        elif score_b > score_a:
            return "b"
        else:
            return None
    
    def ab_test_prompt(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        variants: List[str] = None,
        model_type: str = 'openai'
    ) -> ABTestResult:
        """
        Run A/B test comparing prompt versions.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            variants: List of prompt versions to test (default: ['v1', 'v2'])
            model_type: Model type to use ('openai' or 'ollama')
            
        Returns:
            ABTestResult with comparison
        """
        if variants is None:
            variants = ['v1', 'v2']
        
        if len(variants) < 2:
            raise ValueError("Need at least 2 prompt variants for A/B testing")
        
        query_id = str(uuid.uuid4())
        logger.info(f"Running A/B test [{query_id}] for query: {query[:50]}...")
        
        # Generate responses for each variant
        responses = []
        latencies = []
        
        for variant in variants[:2]:  # Compare first two variants
            import time
            start_time = time.time()
            
            try:
                response = self.generator.generate(
                    query=query,
                    context_chunks=context_chunks,
                    prompt_version=variant,
                    model_type=model_type,
                    fallback=False
                )
                latency = time.time() - start_time
                responses.append(response)
                latencies.append(latency)
            except Exception as e:
                logger.error(f"Error generating response for variant {variant}: {e}")
                # Create a dummy response for failed variant
                from src.generator.generate import LLMResponse
                failed_response = LLMResponse(
                    answer=f"Error: {str(e)}",
                    sources=[],
                    trace_id=query_id,
                    token_usage={"prompt": 0, "completion": 0, "total": 0},
                    prompt_version=variant,
                    model_used=model_type,
                    cost_estimate=0.0
                )
                responses.append(failed_response)
                latencies.append(float('inf'))
        
        # Calculate metrics
        metrics_a = self._calculate_metrics(responses[0], latencies[0])
        metrics_b = self._calculate_metrics(responses[1], latencies[1])
        
        # Determine winner
        winner = self._determine_winner(metrics_a, metrics_b)
        
        result = ABTestResult(
            query_id=query_id,
            query=query,
            prompt_version_a=variants[0],
            prompt_version_b=variants[1],
            response_a=responses[0],
            response_b=responses[1],
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            timestamp=datetime.now().isoformat(),
            winner=winner
        )
        
        # Log result
        self._log_result(result)
        
        return result
    
    def _log_result(self, result: ABTestResult):
        """
        Log A/B test result to file.
        
        Args:
            result: ABTestResult to log
        """
        try:
            # Convert to serializable format
            log_entry = {
                "query_id": result.query_id,
                "query": result.query,
                "prompt_version_a": result.prompt_version_a,
                "prompt_version_b": result.prompt_version_b,
                "response_a": {
                    "answer": result.response_a.answer,
                    "sources": result.response_a.sources,
                    "token_usage": result.response_a.token_usage,
                    "cost_estimate": result.response_a.cost_estimate,
                    "model_used": result.response_a.model_used,
                },
                "response_b": {
                    "answer": result.response_b.answer,
                    "sources": result.response_b.sources,
                    "token_usage": result.response_b.token_usage,
                    "cost_estimate": result.response_b.cost_estimate,
                    "model_used": result.response_b.model_used,
                },
                "metrics_a": result.metrics_a,
                "metrics_b": result.metrics_b,
                "winner": result.winner,
                "timestamp": result.timestamp,
            }
            
            # Append to JSONL file
            log_file = self.results_dir / "ab_test_results.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            logger.info(
                f"A/B test [{result.query_id}] logged. "
                f"Winner: {result.winner or 'tie'} "
                f"(A: {result.metrics_a['cost_estimate']:.4f}, "
                f"B: {result.metrics_b['cost_estimate']:.4f})"
            )
        except Exception as e:
            logger.error(f"Failed to log A/B test result: {e}")
    
    def analyze_results(self, limit: int = 100) -> Dict[str, Any]:
        """
        Analyze A/B test results from log file.
        
        Args:
            limit: Maximum number of results to analyze
            
        Returns:
            Analysis summary
        """
        log_file = self.results_dir / "ab_test_results.jsonl"
        
        if not log_file.exists():
            return {"error": "No A/B test results found"}
        
        results = []
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if len(results) >= limit:
                        break
                    results.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read A/B test results: {e}")
            return {"error": str(e)}
        
        if not results:
            return {"error": "No results to analyze"}
        
        # Calculate statistics
        total = len(results)
        wins_a = sum(1 for r in results if r.get("winner") == "a")
        wins_b = sum(1 for r in results if r.get("winner") == "b")
        ties = sum(1 for r in results if r.get("winner") is None)
        
        avg_cost_a = sum(r["metrics_a"]["cost_estimate"] for r in results) / total
        avg_cost_b = sum(r["metrics_b"]["cost_estimate"] for r in results) / total
        
        avg_latency_a = sum(r["metrics_a"]["latency"] for r in results) / total
        avg_latency_b = sum(r["metrics_b"]["latency"] for r in results) / total
        
        return {
            "total_tests": total,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "win_rate_a": wins_a / total,
            "win_rate_b": wins_b / total,
            "avg_cost_a": avg_cost_a,
            "avg_cost_b": avg_cost_b,
            "cost_savings": (avg_cost_b - avg_cost_a) / avg_cost_b * 100 if avg_cost_b > 0 else 0,
            "avg_latency_a": avg_latency_a,
            "avg_latency_b": avg_latency_b,
            "latency_improvement": (avg_latency_b - avg_latency_a) / avg_latency_b * 100 if avg_latency_b > 0 else 0,
        }

