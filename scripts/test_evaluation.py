#!/usr/bin/env python3
"""
Test script for evaluation harness.
Runs evaluation on a sample dataset and prints results.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.eval import Evaluator, DatasetLoader
from src.retrieval import DenseRetriever
from src.generator import LLMGenerator
from src.config import settings

def main():
    print("=" * 60)
    print("RAG Pipeline Evaluation Test")
    print("=" * 60)
    
    # Create sample dataset if it doesn't exist
    dataset_path = "data/eval/sample_dataset.json"
    dataset_file = Path(dataset_path)
    
    if not dataset_file.exists():
        print(f"\n📝 Creating sample dataset at {dataset_path}...")
        dataset_file.parent.mkdir(parents=True, exist_ok=True)
        DatasetLoader.create_sample_dataset(dataset_path)
        print("✅ Sample dataset created!")
        print("\n⚠️  Note: This is a sample dataset. Update it with your own queries and ground truth.")
        print("   Format: JSON with queries, ground_truth_answer, and relevant_doc_ids")
        return
    
    # Load dataset
    print(f"\n📂 Loading dataset from {dataset_path}...")
    try:
        dataset = DatasetLoader.load_from_json(dataset_path)
        print(f"✅ Loaded dataset '{dataset.name}' with {len(dataset)} queries")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Initialize components
    print("\n🔧 Initializing components...")
    evaluator = Evaluator()
    retriever = DenseRetriever()
    generator = LLMGenerator()
    print("✅ Components initialized")
    
    # Run evaluation
    print(f"\n🚀 Running evaluation on {len(dataset)} queries...")
    print("   This may take a while depending on the number of queries...\n")
    
    try:
        reports = evaluator.run_evaluation_suite(
            dataset=dataset,
            retriever=retriever,
            generator=generator,
            output_path="data/eval/results.json"
        )
        
        print(f"\n✅ Evaluation complete! Generated {len(reports)} reports")
        
        # Compute summary statistics
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        
        stats = evaluator.compute_summary_stats(reports)
        
        print(f"\n📊 Retrieval Metrics:")
        print(f"   Precision@1: {stats['retrieval']['precision_at_1']:.3f}")
        print(f"   Precision@3: {stats['retrieval']['precision_at_3']:.3f}")
        print(f"   Precision@5: {stats['retrieval']['precision_at_5']:.3f}")
        print(f"   Recall: {stats['retrieval']['recall']:.3f}")
        print(f"   Mean Reciprocal Rank: {stats['retrieval']['mean_reciprocal_rank']:.3f}")
        
        print(f"\n📝 Generation Metrics:")
        print(f"   BLEU Score: {stats['generation']['bleu_score']:.3f}")
        print(f"   ROUGE-L Score: {stats['generation']['rouge_l_score']:.3f}")
        
        print(f"\n✅ Faithfulness Metrics:")
        print(f"   Hallucination Rate: {stats['faithfulness']['hallucination_rate']:.3f}")
        print(f"   Faithful Rate: {stats['faithfulness']['faithful_rate']:.3f}")
        print(f"   Avg Confidence: {stats['faithfulness']['avg_confidence']:.3f}")
        
        print(f"\n⚡ Performance Metrics:")
        print(f"   Avg Latency: {stats['performance']['avg_latency']:.2f}s")
        print(f"   Total Cost: ${stats['performance']['total_cost']:.4f}")
        print(f"   Total Tokens: {stats['performance']['total_tokens']:,}")
        
        # Check against targets
        print(f"\n🎯 Target Comparison:")
        hallucination_rate = stats['faithfulness']['hallucination_rate']
        if hallucination_rate < settings.hallucination_threshold:
            print(f"   ✅ Hallucination rate ({hallucination_rate:.3f}) < threshold ({settings.hallucination_threshold})")
        else:
            print(f"   ❌ Hallucination rate ({hallucination_rate:.3f}) >= threshold ({settings.hallucination_threshold})")
        
        avg_latency = stats['performance']['avg_latency']
        if avg_latency < settings.target_latency_p95:
            print(f"   ✅ Avg latency ({avg_latency:.2f}s) < P95 target ({settings.target_latency_p95}s)")
        else:
            print(f"   ⚠️  Avg latency ({avg_latency:.2f}s) >= P95 target ({settings.target_latency_p95}s)")
        
        print(f"\n📄 Detailed results saved to: data/eval/results.json")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

