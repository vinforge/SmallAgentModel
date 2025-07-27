#!/usr/bin/env python3
"""
Fix Docling Integration Issues
Addresses the PDF format and NumPy compatibility issues.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def fix_numpy_compatibility():
    """Fix NumPy compatibility issues."""
    print("🔧 Fixing NumPy compatibility...")
    
    try:
        # Downgrade NumPy to compatible version
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4"], check=True)
        print("✅ NumPy downgraded to compatible version")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to fix NumPy: {e}")
        return False

def update_docling_parser_for_text_files():
    """Update the Docling parser to handle text files properly."""
    print("🔧 Updating Docling parser for better text file handling...")
    
    try:
        parser_file = Path("multimodal_processing/docling_parser.py")
        
        if not parser_file.exists():
            print(f"❌ Parser file not found: {parser_file}")
            return False
        
        # Read current content
        with open(parser_file, 'r') as f:
            content = f.read()
        
        # Update the supported formats to prioritize text files
        old_formats = '''            self.supported_formats = {
                '.pdf': 'application/pdf',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.html': 'text/html',
                '.htm': 'text/html',
                '.txt': 'text/plain',
                '.md': 'text/markdown',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.tiff': 'image/tiff',
                '.wav': 'audio/wav',
                '.mp3': 'audio/mpeg'
            }'''
        
        new_formats = '''            # Prioritize text files and add fallback handling
            self.supported_formats = {
                '.txt': 'text/plain',
                '.md': 'text/markdown',
                '.pdf': 'application/pdf',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.html': 'text/html',
                '.htm': 'text/html',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.tiff': 'image/tiff',
                '.wav': 'audio/wav',
                '.mp3': 'audio/mpeg'
            }
            
            # Text file extensions that should use direct text reading
            self.text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.htm', '.css', '.json', '.xml', '.csv'}'''
        
        if old_formats in content:
            content = content.replace(old_formats, new_formats)
        
        # Update the parse_document method to handle text files better
        old_parse_logic = '''            # Check if file format is supported
            file_extension = file_path.suffix.lower()
            if file_extension not in self.supported_formats:
                logger.warning(f"Unsupported file format: {file_extension}")
                return self._fallback_parse(file_path)
            
            logger.info(f"🔄 Processing document with Docling: {file_path.name}")
            
            # Convert document using Docling
            result = self.converter.convert(str(file_path))'''
        
        new_parse_logic = '''            # Check if file format is supported
            file_extension = file_path.suffix.lower()
            
            # Handle text files directly without Docling
            if file_extension in self.text_extensions:
                logger.info(f"🔄 Processing text file directly: {file_path.name}")
                return self._parse_text_file(file_path)
            
            # Use Docling for other supported formats
            if file_extension not in self.supported_formats:
                logger.warning(f"Unsupported file format: {file_extension}")
                return self._fallback_parse(file_path)
            
            logger.info(f"🔄 Processing document with Docling: {file_path.name}")
            
            try:
                # Convert document using Docling
                result = self.converter.convert(str(file_path))
            except Exception as docling_error:
                logger.warning(f"Docling failed for {file_path.name}: {docling_error}")
                return self._fallback_parse(file_path, error=str(docling_error))'''
        
        if old_parse_logic in content:
            content = content.replace(old_parse_logic, new_parse_logic)
        
        # Add the new _parse_text_file method
        text_file_method = '''
    def _parse_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse text files directly without Docling."""
        try:
            # Read text content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Split into paragraphs for content blocks
            paragraphs = [p.strip() for p in content.split('\\n\\n') if p.strip()]
            
            content_blocks = []
            for i, paragraph in enumerate(paragraphs):
                block = {
                    'type': 'paragraph',
                    'content': paragraph,
                    'bbox': None,
                    'page': 1
                }
                content_blocks.append(block)
            
            # Extract basic metadata
            lines = content.split('\\n')
            title = lines[0].strip() if lines else file_path.stem
            
            return {
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_type': 'text/plain',
                'success': True,
                'parser': 'text_direct',
                'content_blocks': content_blocks,
                'metadata': {
                    'title': title,
                    'line_count': len(lines),
                    'paragraph_count': len(paragraphs)
                },
                'full_text': content,
                'markdown': content,  # For text files, content is already readable
                'tables': [],
                'images': [],
                'structure': {
                    'page_count': 1,
                    'element_count': len(content_blocks),
                    'table_count': 0,
                    'has_images': False
                }
            }
            
        except Exception as e:
            logger.error(f"Text file parsing failed: {e}")
            return self._fallback_parse(file_path, error=str(e))
'''
        
        # Insert the new method before the _fallback_parse method
        fallback_method_start = content.find('    def _fallback_parse(self, file_path: Path')
        if fallback_method_start != -1:
            content = content[:fallback_method_start] + text_file_method + '\\n' + content[fallback_method_start:]
        
        # Write updated content
        with open(parser_file, 'w') as f:
            f.write(content)
        
        print("✅ Docling parser updated for better text file handling")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update parser: {e}")
        return False

def test_improved_parser():
    """Test the improved parser with text files."""
    print("🧪 Testing improved parser...")
    
    try:
        from multimodal_processing.docling_parser import DoclingDocumentParser
        
        # Create parser
        parser = DoclingDocumentParser()
        print("✅ Parser created")
        
        # Create test text file
        import tempfile
        test_content = """CYSE493: Advanced Cybersecurity Engineering
Spring 2025 Course Brochure

Course Overview:
This advanced cybersecurity course covers cutting-edge topics in network security, 
threat intelligence, incident response, and cryptographic protocols.

Learning Objectives:
Upon completion, students will be able to:
1. Design secure network architectures
2. Implement threat detection systems
3. Conduct comprehensive security audits
4. Develop incident response plans
5. Apply cryptographic solutions

Prerequisites:
- CYSE301: Introduction to Cybersecurity
- CYSE350: Network Security Fundamentals
- Programming experience in Python or C++

Assessment Methods:
- Midterm Examination (25%)
- Final Project (35%)
- Lab Assignments (25%)
- Class Participation (15%)

Instructor: Dr. Sarah Johnson
Email: s.johnson@university.edu
Office Hours: Wednesdays 1:00-3:00 PM"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name
        
        print(f"📄 Created test file: {temp_file_path}")
        
        # Test parsing
        result = parser.parse_document(temp_file_path)
        
        print(f"📊 Parsing Results:")
        print(f"   Success: {result['success']}")
        print(f"   Parser: {result['parser']}")
        print(f"   Content blocks: {len(result['content_blocks'])}")
        print(f"   Full text length: {len(result['full_text'])}")
        print(f"   Structure: {result['structure']}")
        
        # Show content preview
        if result['content_blocks']:
            print(f"\\n📋 Content Blocks:")
            for i, block in enumerate(result['content_blocks'][:3]):
                content = block['content'][:100] + "..." if len(block['content']) > 100 else block['content']
                print(f"   {i+1}. {content}")
        
        # Cleanup
        os.unlink(temp_file_path)
        
        return result['success'] and result['parser'] == 'text_direct'
        
    except Exception as e:
        print(f"❌ Parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multimodal_pipeline_integration():
    """Test the complete pipeline integration."""
    print("\\n🔄 Testing multimodal pipeline integration...")
    
    try:
        from multimodal_processing.multimodal_pipeline import get_multimodal_pipeline
        
        # Get pipeline
        pipeline = get_multimodal_pipeline()
        print("✅ Pipeline loaded")
        
        # Create test document
        import tempfile
        test_content = """CYSE493 Spring Brochure 2024-25
Cybersecurity Engineering Program

This comprehensive course covers advanced cybersecurity topics including:
- Network Security Architecture
- Threat Intelligence and Analysis  
- Incident Response Procedures
- Cryptographic Protocols
- Risk Assessment Methodologies

Students will gain practical experience through hands-on labs and real-world scenarios.
The course prepares graduates for leadership roles in cybersecurity engineering.

Prerequisites include CYSE301 and CYSE350, plus programming experience.
Assessment includes midterm exam, final project, lab assignments, and participation.

Instructor: Dr. Sarah Johnson
Contact: s.johnson@university.edu
Schedule: Tuesdays and Thursdays, 2:00-3:30 PM"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name
        
        print(f"📄 Created test document: {temp_file_path}")
        
        # Process through pipeline
        result = pipeline.process_document(temp_file_path)
        
        print(f"📊 Pipeline Results:")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Document ID: {result.get('document_id', 'N/A')}")
        
        if 'consolidated_knowledge' in result:
            knowledge = result['consolidated_knowledge']
            print(f"   Summary length: {len(knowledge.get('summary', ''))}")
            print(f"   Key concepts: {len(knowledge.get('key_concepts', []))}")
        
        # Cleanup
        os.unlink(temp_file_path)
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main fix function."""
    print("🔧 SAM Docling Integration Fix")
    print("=" * 50)
    
    # Step 1: Fix NumPy compatibility
    numpy_fixed = fix_numpy_compatibility()
    
    # Step 2: Update parser for better text handling
    parser_updated = update_docling_parser_for_text_files()
    
    # Step 3: Test improved parser
    parser_test = test_improved_parser()
    
    # Step 4: Test pipeline integration
    pipeline_test = test_multimodal_pipeline_integration()
    
    print("\\n" + "=" * 50)
    print("🎯 DOCLING FIX SUMMARY")
    print("=" * 50)
    
    if numpy_fixed and parser_updated and parser_test and pipeline_test:
        print("🎉 ALL FIXES SUCCESSFUL!")
        print("✅ NumPy compatibility fixed")
        print("✅ Parser updated for text files")
        print("✅ Text file parsing working")
        print("✅ Pipeline integration working")
        print("\\n🚀 Benefits:")
        print("- Text files (.txt, .md) processed directly")
        print("- PDF files still use Docling when valid")
        print("- Better error handling and fallbacks")
        print("- Improved content extraction")
        print("\\n💡 Next steps:")
        print("1. Upload the CYSE493 Spring Brochure (any format)")
        print("2. Ask SAM to summarize the document")
        print("3. Verify relevance scores are now > 0.00")
        print("4. Test document Q&A functionality")
    else:
        print("❌ Some fixes failed:")
        if not numpy_fixed:
            print("   - NumPy compatibility fix failed")
        if not parser_updated:
            print("   - Parser update failed")
        if not parser_test:
            print("   - Parser test failed")
        if not pipeline_test:
            print("   - Pipeline integration failed")
    
    return numpy_fixed and parser_updated and parser_test and pipeline_test

if __name__ == "__main__":
    main()
