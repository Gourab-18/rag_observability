# prompt template management with versioning

from typing import Dict, Optional
from pathlib import Path
import yaml
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class PromptTemplate:
    # represents a prompt template with versioning
    
    def __init__(self, version: str, template: str, metadata: Dict = None):
        self.version = version
        self.template = template
        self.metadata = metadata or {}
    
    def format(self, **kwargs) -> str:
        # format template with provided variables
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            raise ValueError(f"Template requires variable: {e}")


class PromptManager:
    # manages prompt templates with versioning
    
    def __init__(self, prompts_path: str = None):
        # initialize with prompts directory path
        self.prompts_path = Path(prompts_path or settings.prompts_path)
        self._templates: Dict[str, PromptTemplate] = {}
        self._versions_metadata: Dict = {}
        self._load_versions_metadata()
    
    def _load_versions_metadata(self):
        # load versions.yaml if it exists
        versions_file = self.prompts_path / "versions.yaml"
        if versions_file.exists():
            try:
                with open(versions_file, 'r') as f:
                    self._versions_metadata = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Could not load versions.yaml: {e}")
                self._versions_metadata = {}
        else:
            logger.info("versions.yaml not found, using defaults")
            self._versions_metadata = {}
    
    def load_prompt_template(self, version: str = 'v1') -> PromptTemplate:
        # load prompt template for a specific version
        if version in self._templates:
            return self._templates[version]
        
        # load template file
        template_file = self.prompts_path / version / "rag_template.txt"
        
        if not template_file.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_file}")
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # get metadata for this version
            metadata = self._versions_metadata.get(version, {})
            
            # create and cache template
            template = PromptTemplate(version, template_content, metadata)
            self._templates[version] = template
            
            logger.info(f"Loaded prompt template version {version}")
            return template
            
        except Exception as e:
            logger.error(f"Error loading prompt template {version}: {e}")
            raise
    
    def list_versions(self) -> list:
        # list all available prompt versions
        versions = []
        for item in self.prompts_path.iterdir():
            if item.is_dir() and item.name.startswith('v'):
                versions.append(item.name)
        return sorted(versions)
    
    def get_version_metadata(self, version: str) -> Dict:
        # get metadata for a specific version
        return self._versions_metadata.get(version, {})

