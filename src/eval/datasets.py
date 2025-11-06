"""
Evaluation dataset loader and management.
Handles Q&A pairs with ground truth answers and relevant document IDs.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuery:
    """Single evaluation query with ground truth."""
    query_id: str
    query: str
    ground_truth_answer: str
    relevant_doc_ids: List[str]  # Document IDs that should be retrieved
    relevant_chunk_ids: Optional[List[str]] = None  # Specific chunk IDs (optional)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationDataset:
    """Evaluation dataset containing multiple queries."""
    name: str
    version: str
    queries: List[EvaluationQuery]
    metadata: Optional[Dict[str, Any]] = None
    
    def __len__(self):
        return len(self.queries)
    
    def get_query(self, query_id: str) -> Optional[EvaluationQuery]:
        """Get query by ID."""
        for q in self.queries:
            if q.query_id == query_id:
                return q
        return None


class DatasetLoader:
    """Loader for evaluation datasets."""
    
    @staticmethod
    def load_from_json(file_path: str) -> EvaluationDataset:
        """
        Load evaluation dataset from JSON file.
        
        Expected format:
        {
            "name": "dataset_name",
            "version": "1.0",
            "queries": [
                {
                    "query_id": "q1",
                    "query": "What is...?",
                    "ground_truth_answer": "The answer is...",
                    "relevant_doc_ids": ["doc1", "doc2"],
                    "relevant_chunk_ids": ["chunk1", "chunk2"],  # optional
                    "metadata": {}  # optional
                }
            ],
            "metadata": {}  # optional
        }
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            EvaluationDataset
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        queries = []
        for q_data in data.get("queries", []):
            query = EvaluationQuery(
                query_id=q_data["query_id"],
                query=q_data["query"],
                ground_truth_answer=q_data["ground_truth_answer"],
                relevant_doc_ids=q_data.get("relevant_doc_ids", []),
                relevant_chunk_ids=q_data.get("relevant_chunk_ids"),
                metadata=q_data.get("metadata")
            )
            queries.append(query)
        
        dataset = EvaluationDataset(
            name=data.get("name", "unnamed"),
            version=data.get("version", "1.0"),
            queries=queries,
            metadata=data.get("metadata")
        )
        
        logger.info(f"Loaded dataset '{dataset.name}' v{dataset.version} with {len(queries)} queries")
        return dataset
    
    @staticmethod
    def save_to_json(dataset: EvaluationDataset, file_path: str):
        """
        Save evaluation dataset to JSON file.
        
        Args:
            dataset: EvaluationDataset to save
            file_path: Path to save JSON file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "name": dataset.name,
            "version": dataset.version,
            "queries": [
                {
                    "query_id": q.query_id,
                    "query": q.query,
                    "ground_truth_answer": q.ground_truth_answer,
                    "relevant_doc_ids": q.relevant_doc_ids,
                    "relevant_chunk_ids": q.relevant_chunk_ids,
                    "metadata": q.metadata
                }
                for q in dataset.queries
            ],
            "metadata": dataset.metadata
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved dataset '{dataset.name}' to {file_path}")
    
    @staticmethod
    def create_sample_dataset(output_path: str = "data/eval/sample_dataset.json"):
        """
        Create a sample evaluation dataset for testing.
        
        Args:
            output_path: Path to save sample dataset
        """
        sample_queries = [
            EvaluationQuery(
                query_id="sample_1",
                query="What is the main topic of the document?",
                ground_truth_answer="The document discusses the main topic of...",
                relevant_doc_ids=["example_001"],
                metadata={"category": "general"}
            ),
            EvaluationQuery(
                query_id="sample_2",
                query="Who is the author?",
                ground_truth_answer="The author is...",
                relevant_doc_ids=["example_001"],
                metadata={"category": "metadata"}
            ),
        ]
        
        dataset = EvaluationDataset(
            name="sample_dataset",
            version="1.0",
            queries=sample_queries,
            metadata={"description": "Sample evaluation dataset for testing"}
        )
        
        DatasetLoader.save_to_json(dataset, output_path)
        return dataset

