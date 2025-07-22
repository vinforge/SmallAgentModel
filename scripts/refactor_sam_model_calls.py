#!/usr/bin/env python3
"""
SAM 2.0 Phase 1: Model Interface Refactoring Script
===================================================

This script refactors SAM components to use the unified ModelInterface
instead of direct Ollama API calls.

Key Changes:
- Replace direct Ollama API calls with SAMModelClient
- Update import statements
- Maintain backward compatibility
- Add error handling and logging

Author: SAM Development Team
Version: 1.0.0
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAMModelRefactorer:
    """Refactors SAM components to use the unified model interface."""
    
    def __init__(self, sam_root: Path):
        """Initialize the refactorer with SAM root directory."""
        self.sam_root = sam_root
        self.refactored_files = []
        self.backup_files = []
        
    def find_files_with_ollama_calls(self) -> List[Path]:
        """Find Python files that contain direct Ollama API calls."""
        files_to_refactor = []
        
        # Patterns to search for
        patterns = [
            r'requests\.post.*api/generate',
            r'OllamaInterface',
            r'OllamaClient',
            r'localhost:11434',
            r'base_url.*11434'
        ]
        
        # Search in key directories
        search_dirs = [
            'sam',
            'memory',
            'web_ui',
            'ui',
            'reasoning'
        ]
        
        for search_dir in search_dirs:
            dir_path = self.sam_root / search_dir
            if dir_path.exists():
                for py_file in dir_path.rglob('*.py'):
                    try:
                        content = py_file.read_text(encoding='utf-8')
                        
                        # Check if file contains Ollama-related patterns
                        for pattern in patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                files_to_refactor.append(py_file)
                                logger.info(f"Found Ollama calls in: {py_file.relative_to(self.sam_root)}")
                                break
                                
                    except Exception as e:
                        logger.warning(f"Could not read {py_file}: {e}")
        
        return files_to_refactor
    
    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the file before modification."""
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
        self.backup_files.append(backup_path)
        logger.info(f"Created backup: {backup_path.relative_to(self.sam_root)}")
        return backup_path
    
    def refactor_implicit_knowledge(self, file_path: Path) -> bool:
        """Refactor sam/orchestration/skills/reasoning/implicit_knowledge.py"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Replace the OllamaInterface class with SAM model client
            new_content = re.sub(
                r'class OllamaInterface:.*?def generate\(self, prompt: str, temperature: float = 0\.7, max_tokens: int = 500\) -> str:.*?return ""',
                '''# Use SAM unified model interface
            from sam.core.sam_model_client import get_sam_model_client
            
            sam_client = get_sam_model_client()
            
            def generate_response(prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
                """Generate response using SAM's unified model interface."""
                try:
                    return sam_client.generate(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except Exception as e:
                    logger.warning(f"SAM model generation failed: {e}")
                    return ""''',
                content,
                flags=re.DOTALL
            )
            
            # Update the usage of OllamaInterface
            new_content = re.sub(
                r'ollama = OllamaInterface\(\)\s*response = ollama\.generate\(',
                'response = generate_response(',
                new_content
            )
            
            if new_content != content:
                file_path.write_text(new_content, encoding='utf-8')
                logger.info(f"✅ Refactored: {file_path.relative_to(self.sam_root)}")
                return True
            
        except Exception as e:
            logger.error(f"❌ Failed to refactor {file_path}: {e}")
            return False
        
        return False
    
    def refactor_insight_generator(self, file_path: Path) -> bool:
        """Refactor memory/synthesis/insight_generator.py"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Add SAM model client import at the top
            if 'from sam.core.sam_model_client import' not in content:
                import_section = '''
from sam.core.sam_model_client import create_ollama_compatible_client
'''
                # Insert after existing imports
                content = re.sub(
                    r'(import logging\n)',
                    r'\1' + import_section,
                    content
                )
            
            # Replace the _create_ollama_client method
            new_content = re.sub(
                r'def _create_ollama_client\(self\):.*?return OllamaClient\(\)',
                '''def _create_ollama_client(self):
        """Create SAM-compatible client."""
        return create_ollama_compatible_client()''',
                content,
                flags=re.DOTALL
            )
            
            if new_content != content:
                file_path.write_text(new_content, encoding='utf-8')
                logger.info(f"✅ Refactored: {file_path.relative_to(self.sam_root)}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to refactor {file_path}: {e}")
            return False
        
        return False
    
    def refactor_self_decide_framework(self, file_path: Path) -> bool:
        """Refactor reasoning/self_decide_framework.py"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Replace model.generate calls with SAM client
            if 'self.model.generate(' in content:
                # Add import at the top
                if 'from sam.core.sam_model_client import' not in content:
                    import_section = '''
from sam.core.sam_model_client import get_sam_model_client
'''
                    content = re.sub(
                        r'(import logging\n)',
                        r'\1' + import_section,
                        content
                    )
                
                # Replace model.generate calls
                new_content = re.sub(
                    r'self\.model\.generate\(',
                    'get_sam_model_client().generate(',
                    content
                )
                
                if new_content != content:
                    file_path.write_text(new_content, encoding='utf-8')
                    logger.info(f"✅ Refactored: {file_path.relative_to(self.sam_root)}")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Failed to refactor {file_path}: {e}")
            return False
        
        return False
    
    def refactor_generic_ollama_calls(self, file_path: Path) -> bool:
        """Refactor generic Ollama API calls in any file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            # Add SAM model client import if needed
            if ('requests.post' in content and 'api/generate' in content) or 'OllamaInterface' in content:
                if 'from sam.core.sam_model_client import' not in content:
                    import_section = '''
from sam.core.sam_model_client import create_legacy_ollama_client
'''
                    # Insert after the last import or at the beginning
                    if 'import ' in content:
                        content = re.sub(
                            r'(import [^\n]*\n(?:from [^\n]*\n)*)',
                            r'\1' + import_section,
                            content,
                            count=1
                        )
                    else:
                        content = import_section + '\n' + content
            
            # Replace direct Ollama API calls with legacy client
            content = re.sub(
                r'requests\.post\(\s*f?["\'].*?api/generate["\'].*?\)',
                'create_legacy_ollama_client().generate(prompt)',
                content,
                flags=re.DOTALL
            )
            
            # Replace OllamaInterface instantiation
            content = re.sub(
                r'OllamaInterface\(\)',
                'create_legacy_ollama_client()',
                content
            )
            
            if content != original_content:
                file_path.write_text(content, encoding='utf-8')
                logger.info(f"✅ Refactored: {file_path.relative_to(self.sam_root)}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to refactor {file_path}: {e}")
            return False
        
        return False
    
    def refactor_file(self, file_path: Path) -> bool:
        """Refactor a specific file based on its path and content."""
        # Create backup first
        self.backup_file(file_path)
        
        file_name = file_path.name
        relative_path = str(file_path.relative_to(self.sam_root))
        
        # Apply specific refactoring based on file
        if 'implicit_knowledge.py' in relative_path:
            return self.refactor_implicit_knowledge(file_path)
        elif 'insight_generator.py' in relative_path:
            return self.refactor_insight_generator(file_path)
        elif 'self_decide_framework.py' in relative_path:
            return self.refactor_self_decide_framework(file_path)
        else:
            return self.refactor_generic_ollama_calls(file_path)
    
    def run_refactoring(self) -> Dict[str, int]:
        """Run the complete refactoring process."""
        logger.info("🔄 Starting SAM model interface refactoring...")
        
        # Find files that need refactoring
        files_to_refactor = self.find_files_with_ollama_calls()
        
        if not files_to_refactor:
            logger.info("✅ No files found that need refactoring")
            return {"total": 0, "refactored": 0, "failed": 0}
        
        logger.info(f"Found {len(files_to_refactor)} files to refactor")
        
        # Refactor each file
        refactored_count = 0
        failed_count = 0
        
        for file_path in files_to_refactor:
            try:
                if self.refactor_file(file_path):
                    refactored_count += 1
                    self.refactored_files.append(file_path)
                else:
                    logger.warning(f"No changes needed for: {file_path.relative_to(self.sam_root)}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to refactor {file_path}: {e}")
                failed_count += 1
        
        # Summary
        logger.info(f"✅ Refactoring complete:")
        logger.info(f"   Total files: {len(files_to_refactor)}")
        logger.info(f"   Refactored: {refactored_count}")
        logger.info(f"   Failed: {failed_count}")
        logger.info(f"   Backups created: {len(self.backup_files)}")
        
        return {
            "total": len(files_to_refactor),
            "refactored": refactored_count,
            "failed": failed_count
        }
    
    def restore_backups(self):
        """Restore all files from backups (rollback)."""
        logger.info("🔄 Restoring files from backups...")
        
        for backup_path in self.backup_files:
            original_path = backup_path.with_suffix('')
            try:
                original_path.write_text(backup_path.read_text(encoding='utf-8'), encoding='utf-8')
                logger.info(f"Restored: {original_path.relative_to(self.sam_root)}")
            except Exception as e:
                logger.error(f"Failed to restore {original_path}: {e}")
        
        logger.info("✅ Backup restoration complete")

def main():
    """Main execution function."""
    try:
        # Get SAM root directory
        sam_root = Path(__file__).parent.parent
        
        logger.info(f"SAM root directory: {sam_root}")
        
        # Create refactorer
        refactorer = SAMModelRefactorer(sam_root)
        
        # Run refactoring
        results = refactorer.run_refactoring()
        
        if results["failed"] > 0:
            logger.warning(f"⚠️  {results['failed']} files failed to refactor")
            return 1
        else:
            logger.info("🎉 All files refactored successfully!")
            return 0
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
