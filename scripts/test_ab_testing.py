#!/usr/bin/env python3
"""
Test script for A/B testing prompt versions.
Compares v1 and v2 prompts on sample queries.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generator.ab_testing import ABTester
from src.retrieval import DenseRetriever

def main():
    print("=" * 60)
    print("A/B Testing: Prompt Version Comparison")
    print("=" * 60)
    
    # Initialize components
    print("\n🔧 Initializing components...")
    ab_tester = ABTester()
    retriever = DenseRetriever()
    print("✅ Components initialized")
    
    # Sample queries for testing
    test_queries = [
        "What is the main topic of the document?",
        "Who is the author?",
        "What are the key points discussed?",
    ]
    
    print(f"\n🧪 Running A/B tests on {len(test_queries)} queries...")
    print("   Comparing prompt versions: v1 vs v2\n")
    
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}/{len(test_queries)}: {query[:50]}...")
        
        try:
            # Retrieve chunks
            chunks = retriever.retrieve(query, top_k=5)
            
            if not chunks:
                print(f"   ⚠️  No chunks retrieved, skipping...")
                continue
            
            # Run A/B test
            result = ab_tester.ab_test_prompt(
                query=query,
                context_chunks=chunks,
                variants=['v1', 'v2'],
                model_type='openai'
            )
            
            results.append(result)
            
            # Print comparison
            print(f"   ✅ Test complete")
            print(f"      Winner: {result.winner or 'tie'}")
            print(f"      Cost A (v1): ${result.metrics_a['cost_estimate']:.4f}")
            print(f"      Cost B (v2): ${result.metrics_b['cost_estimate']:.4f}")
            print(f"      Latency A: {result.metrics_a['latency']:.2f}s")
            print(f"      Latency B: {result.metrics_b['latency']:.2f}s")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
    
    if not results:
        print("❌ No successful A/B tests completed")
        return
    
    # Analyze results
    print("=" * 60)
    print("A/B Test Analysis")
    print("=" * 60)
    
    analysis = ab_tester.analyze_results(limit=100)
    
    if "error" in analysis:
        print(f"\n❌ {analysis['error']}")
        return
    
    print(f"\n📊 Summary:")
    print(f"   Total Tests: {analysis['total_tests']}")
    print(f"   Wins (v1): {analysis['wins_a']} ({analysis['win_rate_a']*100:.1f}%)")
    print(f"   Wins (v2): {analysis['wins_b']} ({analysis['win_rate_b']*100:.1f}%)")
    print(f"   Ties: {analysis['ties']}")
    
    print(f"\n💰 Cost Comparison:")
    print(f"   Avg Cost (v1): ${analysis['avg_cost_a']:.4f}")
    print(f"   Avg Cost (v2): ${analysis['avg_cost_b']:.4f}")
    if analysis['cost_savings'] > 0:
        print(f"   💵 Cost Savings: {analysis['cost_savings']:.1f}% (v1 is cheaper)")
    else:
        print(f"   💵 Cost Increase: {abs(analysis['cost_savings']):.1f}% (v2 is cheaper)")
    
    print(f"\n⚡ Latency Comparison:")
    print(f"   Avg Latency (v1): {analysis['avg_latency_a']:.2f}s")
    print(f"   Avg Latency (v2): {analysis['avg_latency_b']:.2f}s")
    if analysis['latency_improvement'] > 0:
        print(f"   ⚡ Latency Improvement: {analysis['latency_improvement']:.1f}% (v1 is faster)")
    else:
        print(f"   ⚡ Latency Improvement: {abs(analysis['latency_improvement']):.1f}% (v2 is faster)")
    
    print(f"\n📄 Detailed results logged to: data/ab_tests/ab_test_results.jsonl")


if __name__ == "__main__":
    main()

