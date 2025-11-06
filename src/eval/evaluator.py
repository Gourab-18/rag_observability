"""
Evaluation harness for RAG pipeline.
Computes Precision@k, BLEU/ROUGE, faithfulness, and retrieval recall metrics.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from src.eval.datasets import EvaluationQuery, EvaluationDataset
from src.retrieval.retriever import RetrievedChunk
from src.generator.generate import LLMResponse
from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall: float
    mean_reciprocal_rank: float


@dataclass
class GenerationMetrics:
    """Generation evaluation metrics."""
    bleu_score: float
    rouge_l_score: float
    answer_length: int
    reference_length: int


@dataclass
class FaithfulnessScore:
    """Faithfulness evaluation score."""
    is_faithful: bool
    confidence: float
    unsupported_claims: List[str]
    supported_claims: List[str]


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    query_id: str
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    faithfulness_score: FaithfulnessScore
    latency: float
    token_usage: Dict[str, int]
    cost_estimate: float


class Evaluator:
    """
    Evaluator for RAG pipeline components.
    
    Computes:
    - Precision@k (k=1,3,5) for retrieval
    - BLEU/ROUGE scores for generation
    - LLM-based faithfulness checking
    - Retrieval recall
    """
    
    def __init__(self):
        self._initialize_nltk()
    
    def _initialize_nltk(self):
        """Initialize NLTK data if needed."""
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
        except ImportError:
            logger.warning("nltk not installed. Some metrics may not work.")
    
    def evaluate_retrieval(
        self,
        query: EvaluationQuery,
        retrieved_chunks: List[RetrievedChunk],
        k_values: List[int] = None
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality using Precision@k and Recall.
        
        Args:
            query: Evaluation query with ground truth
            retrieved_chunks: Retrieved chunks from retriever
            k_values: List of k values for Precision@k (default: [1, 3, 5])
            
        Returns:
            RetrievalMetrics
        """
        if k_values is None:
            k_values = settings.precision_at_k
        
        # Get relevant doc IDs
        relevant_doc_ids = set(query.relevant_doc_ids)
        if not relevant_doc_ids:
            logger.warning(f"No relevant doc IDs for query {query.query_id}")
            return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate Precision@k
        precision_scores = {}
        for k in k_values:
            top_k_chunks = retrieved_chunks[:k]
            if not top_k_chunks:
                precision_scores[k] = 0.0
                continue
            
            # Count how many retrieved chunks are from relevant docs
            relevant_retrieved = sum(
                1 for chunk in top_k_chunks
                if chunk.doc_id in relevant_doc_ids
            )
            precision_scores[k] = relevant_retrieved / len(top_k_chunks)
        
        # Calculate Recall
        all_retrieved_doc_ids = set(chunk.doc_id for chunk in retrieved_chunks)
        relevant_retrieved = len(relevant_doc_ids.intersection(all_retrieved_doc_ids))
        recall = relevant_retrieved / len(relevant_doc_ids) if relevant_doc_ids else 0.0
        
        # Calculate Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if chunk.doc_id in relevant_doc_ids:
                mrr = 1.0 / rank
                break
        
        return RetrievalMetrics(
            precision_at_1=precision_scores.get(1, 0.0),
            precision_at_3=precision_scores.get(3, 0.0),
            precision_at_5=precision_scores.get(5, 0.0),
            recall=recall,
            mean_reciprocal_rank=mrr
        )
    
    def evaluate_generation(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> GenerationMetrics:
        """
        Evaluate generation quality using BLEU and ROUGE-L scores.
        
        Args:
            generated_answer: Generated answer from LLM
            reference_answer: Ground truth reference answer
            
        Returns:
            GenerationMetrics
        """
        # Calculate BLEU score
        bleu_score = self._calculate_bleu(generated_answer, reference_answer)
        
        # Calculate ROUGE-L score
        rouge_l_score = self._calculate_rouge_l(generated_answer, reference_answer)
        
        return GenerationMetrics(
            bleu_score=bleu_score,
            rouge_l_score=rouge_l_score,
            answer_length=len(generated_answer),
            reference_length=len(reference_answer)
        )
    
    def _calculate_bleu(self, candidate: str, reference: str) -> float:
        """
        Calculate BLEU score between candidate and reference.
        
        Args:
            candidate: Generated text
            reference: Reference text
            
        Returns:
            BLEU score (0-1)
        """
        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # Tokenize
            candidate_tokens = nltk.word_tokenize(candidate.lower())
            reference_tokens = nltk.word_tokenize(reference.lower())
            
            # Calculate BLEU with smoothing
            smoothing = SmoothingFunction().method1
            score = sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                smoothing_function=smoothing
            )
            return score
        except Exception as e:
            logger.warning(f"Failed to calculate BLEU: {e}")
            return 0.0
    
    def _calculate_rouge_l(self, candidate: str, reference: str) -> float:
        """
        Calculate ROUGE-L score (Longest Common Subsequence).
        
        Args:
            candidate: Generated text
            reference: Reference text
            
        Returns:
            ROUGE-L F1 score (0-1)
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(reference, candidate)
            return scores['rougeL'].fmeasure
        except ImportError:
            logger.warning("rouge-score not installed. Using simple LCS calculation.")
            # Fallback: simple LCS-based calculation
            return self._simple_lcs_score(candidate, reference)
        except Exception as e:
            logger.warning(f"Failed to calculate ROUGE-L: {e}")
            return 0.0
    
    def _simple_lcs_score(self, candidate: str, reference: str) -> float:
        """
        Simple LCS-based score calculation (fallback).
        
        Args:
            candidate: Generated text
            reference: Reference text
            
        Returns:
            LCS-based F1 score
        """
        def lcs_length(s1: str, s2: str) -> int:
            """Calculate length of longest common subsequence."""
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs_len = lcs_length(candidate.lower(), reference.lower())
        if lcs_len == 0:
            return 0.0
        
        precision = lcs_len / len(candidate) if candidate else 0.0
        recall = lcs_len / len(reference) if reference else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def evaluate_faithfulness(
        self,
        answer: str,
        sources: List[str],
        use_llm_judge: bool = True
    ) -> FaithfulnessScore:
        """
        Evaluate faithfulness of answer to sources using LLM judge.
        
        Args:
            answer: Generated answer
            sources: Source chunks used for generation
            use_llm_judge: Whether to use LLM-based factuality checking
            
        Returns:
            FaithfulnessScore
        """
        if not sources:
            return FaithfulnessScore(
                is_faithful=False,
                confidence=0.0,
                unsupported_claims=[],
                supported_claims=[]
            )
        
        if use_llm_judge:
            return self._llm_judge_faithfulness(answer, sources)
        else:
            # Simple keyword-based check (fallback)
            return self._simple_faithfulness_check(answer, sources)
    
    def _llm_judge_faithfulness(self, answer: str, sources: List[str]) -> FaithfulnessScore:
        """
        Use LLM to judge faithfulness of answer to sources.
        
        Args:
            answer: Generated answer
            sources: Source chunks
            
        Returns:
            FaithfulnessScore
        """
        try:
            from openai import OpenAI
            from src.config import settings
            
            client = OpenAI(api_key=settings.openai_api_key)
            
            # Create prompt for LLM judge
            sources_text = "\n\n".join([f"[Source {i+1}]: {src}" for i, src in enumerate(sources)])
            
            judge_prompt = f"""You are a factuality checker. Determine if the answer is fully supported by the provided sources.

Sources:
{sources_text}

Answer to check:
{answer}

Instructions:
1. Identify any claims in the answer that are NOT supported by the sources
2. List unsupported claims (if any)
3. List supported claims
4. Provide a confidence score (0.0 to 1.0) indicating how faithful the answer is

Respond in JSON format:
{{
    "is_faithful": true/false,
    "confidence": 0.0-1.0,
    "unsupported_claims": ["claim1", "claim2"],
    "supported_claims": ["claim1", "claim2"]
}}"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a factuality checker. Respond only with valid JSON."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            # Try to extract JSON from response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            return FaithfulnessScore(
                is_faithful=result.get("is_faithful", False),
                confidence=result.get("confidence", 0.0),
                unsupported_claims=result.get("unsupported_claims", []),
                supported_claims=result.get("supported_claims", [])
            )
        except Exception as e:
            logger.warning(f"LLM judge failed: {e}. Falling back to simple check.")
            return self._simple_faithfulness_check(answer, sources)
    
    def _simple_faithfulness_check(self, answer: str, sources: List[str]) -> FaithfulnessScore:
        """
        Simple keyword-based faithfulness check (fallback).
        
        Args:
            answer: Generated answer
            sources: Source chunks
            
        Returns:
            FaithfulnessScore
        """
        # Simple heuristic: check if key phrases from answer appear in sources
        sources_text = " ".join(sources).lower()
        answer_lower = answer.lower()
        
        # Split answer into sentences/claims
        import re
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        supported = []
        unsupported = []
        
        for sentence in sentences[:5]:  # Check first 5 sentences
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Check if key words from sentence appear in sources
            words = set(re.findall(r'\b\w+\b', sentence.lower()))
            if len(words) < 3:
                continue
            
            # Count matching words
            matching_words = sum(1 for word in words if word in sources_text)
            match_ratio = matching_words / len(words) if words else 0
            
            if match_ratio > 0.3:  # At least 30% of words match
                supported.append(sentence)
            else:
                unsupported.append(sentence)
        
        is_faithful = len(unsupported) == 0
        confidence = len(supported) / len(sentences) if sentences else 0.0
        
        return FaithfulnessScore(
            is_faithful=is_faithful,
            confidence=confidence,
            unsupported_claims=unsupported,
            supported_claims=supported
        )
    
    def run_evaluation_suite(
        self,
        dataset: EvaluationDataset,
        retriever,
        generator,
        output_path: str = None
    ) -> List[EvaluationReport]:
        """
        Run complete evaluation suite on dataset.
        
        Args:
            dataset: Evaluation dataset
            retriever: DenseRetriever instance
            generator: LLMGenerator instance
            output_path: Optional path to save results JSON
            
        Returns:
            List of EvaluationReport objects
        """
        reports = []
        
        logger.info(f"Running evaluation suite on {len(dataset)} queries...")
        
        for i, query in enumerate(dataset.queries, 1):
            logger.info(f"Evaluating query {i}/{len(dataset)}: {query.query_id}")
            
            import time
            start_time = time.time()
            
            try:
                # Retrieve chunks
                retrieved_chunks = retriever.retrieve(query.query, top_k=10)
                
                # Generate answer
                response = generator.generate(
                    query=query.query,
                    context_chunks=retrieved_chunks,
                    prompt_version='v1',
                    model_type='openai',
                    fallback=False
                )
                
                latency = time.time() - start_time
                
                # Evaluate retrieval
                retrieval_metrics = self.evaluate_retrieval(query, retrieved_chunks)
                
                # Evaluate generation
                generation_metrics = self.evaluate_generation(
                    response.answer,
                    query.ground_truth_answer
                )
                
                # Evaluate faithfulness
                faithfulness_score = self.evaluate_faithfulness(
                    response.answer,
                    [chunk.text for chunk in retrieved_chunks[:5]]  # Use top 5 sources
                )
                
                report = EvaluationReport(
                    query_id=query.query_id,
                    retrieval_metrics=retrieval_metrics,
                    generation_metrics=generation_metrics,
                    faithfulness_score=faithfulness_score,
                    latency=latency,
                    token_usage=response.token_usage,
                    cost_estimate=response.cost_estimate
                )
                
                reports.append(report)
                
            except Exception as e:
                logger.error(f"Error evaluating query {query.query_id}: {e}")
                # Create empty report for failed query
                reports.append(EvaluationReport(
                    query_id=query.query_id,
                    retrieval_metrics=RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0),
                    generation_metrics=GenerationMetrics(0.0, 0.0, 0, 0),
                    faithfulness_score=FaithfulnessScore(False, 0.0, [], []),
                    latency=0.0,
                    token_usage={},
                    cost_estimate=0.0
                ))
        
        # Save results if output path provided
        if output_path:
            self._save_results(reports, output_path)
        
        return reports
    
    def _save_results(self, reports: List[EvaluationReport], output_path: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            reports: List of EvaluationReport objects
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        results = []
        for report in reports:
            results.append({
                "query_id": report.query_id,
                "retrieval_metrics": {
                    "precision_at_1": report.retrieval_metrics.precision_at_1,
                    "precision_at_3": report.retrieval_metrics.precision_at_3,
                    "precision_at_5": report.retrieval_metrics.precision_at_5,
                    "recall": report.retrieval_metrics.recall,
                    "mean_reciprocal_rank": report.retrieval_metrics.mean_reciprocal_rank,
                },
                "generation_metrics": {
                    "bleu_score": report.generation_metrics.bleu_score,
                    "rouge_l_score": report.generation_metrics.rouge_l_score,
                    "answer_length": report.generation_metrics.answer_length,
                    "reference_length": report.generation_metrics.reference_length,
                },
                "faithfulness_score": {
                    "is_faithful": report.faithfulness_score.is_faithful,
                    "confidence": report.faithfulness_score.confidence,
                    "unsupported_claims": report.faithfulness_score.unsupported_claims,
                    "supported_claims": report.faithfulness_score.supported_claims,
                },
                "latency": report.latency,
                "token_usage": report.token_usage,
                "cost_estimate": report.cost_estimate,
            })
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_path}")
    
    def compute_summary_stats(self, reports: List[EvaluationReport]) -> Dict[str, Any]:
        """
        Compute summary statistics from evaluation reports.
        
        Args:
            reports: List of EvaluationReport objects
            
        Returns:
            Summary statistics dictionary
        """
        if not reports:
            return {}
        
        n = len(reports)
        
        # Average retrieval metrics
        avg_precision_1 = sum(r.retrieval_metrics.precision_at_1 for r in reports) / n
        avg_precision_3 = sum(r.retrieval_metrics.precision_at_3 for r in reports) / n
        avg_precision_5 = sum(r.retrieval_metrics.precision_at_5 for r in reports) / n
        avg_recall = sum(r.retrieval_metrics.recall for r in reports) / n
        avg_mrr = sum(r.retrieval_metrics.mean_reciprocal_rank for r in reports) / n
        
        # Average generation metrics
        avg_bleu = sum(r.generation_metrics.bleu_score for r in reports) / n
        avg_rouge_l = sum(r.generation_metrics.rouge_l_score for r in reports) / n
        
        # Faithfulness metrics
        faithful_count = sum(1 for r in reports if r.faithfulness_score.is_faithful)
        hallucination_rate = 1.0 - (faithful_count / n)
        avg_confidence = sum(r.faithfulness_score.confidence for r in reports) / n
        
        # Performance metrics
        avg_latency = sum(r.latency for r in reports) / n
        total_cost = sum(r.cost_estimate for r in reports)
        total_tokens = sum(r.token_usage.get("total", 0) for r in reports)
        
        return {
            "num_queries": n,
            "retrieval": {
                "precision_at_1": avg_precision_1,
                "precision_at_3": avg_precision_3,
                "precision_at_5": avg_precision_5,
                "recall": avg_recall,
                "mean_reciprocal_rank": avg_mrr,
            },
            "generation": {
                "bleu_score": avg_bleu,
                "rouge_l_score": avg_rouge_l,
            },
            "faithfulness": {
                "hallucination_rate": hallucination_rate,
                "faithful_rate": 1.0 - hallucination_rate,
                "avg_confidence": avg_confidence,
            },
            "performance": {
                "avg_latency": avg_latency,
                "total_cost": total_cost,
                "total_tokens": total_tokens,
            }
        }

