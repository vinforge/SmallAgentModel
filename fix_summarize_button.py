#!/usr/bin/env python3
"""
Fix Summarize Button for e1539_wl-330g.pdf
==========================================

Fix the document retrieval issue by ensuring the summarize button
can access uploaded documents properly.
"""

import sys
import os
sys.path.append('.')

def fix_memory_search_attribute_error():
    """Fix the MemorySearchResult attribute error."""
    print("🔧 Fixing Memory Search Attribute Error")
    print("-" * 45)
    
    try:
        from memory.memory_vectorstore import get_memory_store
        
        memory_store = get_memory_store()
        results = memory_store.search_memories("e1539_wl-330g.pdf", max_results=3, min_similarity=0.1)
        
        if results:
            result = results[0]
            print(f"✅ Found {len(results)} results")
            
            # Check what attributes the result actually has
            attrs = [attr for attr in dir(result) if not attr.startswith('_')]
            print(f"📊 Available attributes: {attrs}")
            
            # Try different attribute names
            content_attrs = ['content', 'text', 'page_content', 'document', 'data']
            metadata_attrs = ['metadata', 'meta', 'info']
            
            content = None
            metadata = None
            
            for attr in content_attrs:
                if hasattr(result, attr):
                    content = getattr(result, attr)
                    print(f"✅ Content found in attribute: {attr}")
                    break
            
            for attr in metadata_attrs:
                if hasattr(result, attr):
                    metadata = getattr(result, attr)
                    print(f"✅ Metadata found in attribute: {attr}")
                    break
            
            if content:
                print(f"📄 Content preview: {str(content)[:200]}...")
                return True, content, metadata
            else:
                print("❌ No content attribute found")
                return False, None, None
        else:
            print("❌ No results found")
            return False, None, None
            
    except Exception as e:
        print(f"❌ Memory search fix failed: {e}")
        return False, None, None

def test_direct_document_access():
    """Test direct access to the document content."""
    print("\n📄 Testing Direct Document Access")
    print("-" * 40)
    
    try:
        # Try to read the PDF directly and extract text
        pdf_path = "e1539_wl-330g.pdf"
        
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return False
        
        # Try to extract text using PyPDF2
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                print(f"✅ PDF loaded with {len(pdf_reader.pages)} pages")
                
                # Extract text from first few pages
                text = ""
                for i, page in enumerate(pdf_reader.pages[:3]):  # First 3 pages
                    page_text = page.extract_text()
                    text += page_text + "\n"
                
                if text.strip():
                    print(f"✅ Extracted {len(text)} characters of text")
                    print(f"📄 Text preview: {text[:300]}...")
                    return True, text
                else:
                    print("❌ No text extracted from PDF")
                    return False, None
                    
        except ImportError:
            print("❌ PyPDF2 not available")
            return False, None
        except Exception as e:
            print(f"❌ PDF text extraction failed: {e}")
            return False, None
            
    except Exception as e:
        print(f"❌ Direct document access failed: {e}")
        return False, None

def create_manual_summary():
    """Create a manual summary using direct document access."""
    print("\n📋 Creating Manual Summary")
    print("-" * 30)
    
    success, text = test_direct_document_access()
    
    if success and text:
        # Create a simple summary
        summary = f"""
📋 **Document Summary: e1539_wl-330g.pdf**

**Document Type**: ASUS WL-330g Wireless Access Point Manual

**Content Overview**:
This appears to be a technical manual or quick installation guide for the ASUS WL-330g wireless access point device.

**Key Information Extracted**:
{text[:1000]}...

**Document Status**: ✅ Successfully accessed and processed
**Processing Method**: Direct PDF text extraction
**Content Length**: {len(text):,} characters
"""
        
        print("✅ Manual summary created")
        print(summary)
        return summary
    else:
        print("❌ Could not create manual summary")
        return None

def test_sam_response_generation():
    """Test SAM's response generation with the document content."""
    print("\n🤖 Testing SAM Response Generation")
    print("-" * 40)
    
    try:
        # Get document content
        success, text = test_direct_document_access()
        
        if not success:
            print("❌ Cannot get document content")
            return False
        
        # Create a prompt with the document content
        prompt = f"""Please provide a comprehensive summary of this ASUS WL-330g document:

{text[:2000]}...

Focus on:
- What type of device this is
- Key features and capabilities  
- Installation or setup information
- Technical specifications if available
"""
        
        print("✅ Created prompt with document content")
        print(f"📊 Prompt length: {len(prompt)} characters")
        
        # Try to use SAM's local model for response
        try:
            from models.ollama_model import OllamaModel
            
            model = OllamaModel()
            response = model.generate(prompt, temperature=0.3, max_tokens=500)
            
            if response:
                print("✅ SAM generated response")
                print(f"📄 Response: {response}")
                return True, response
            else:
                print("❌ SAM generated empty response")
                return False, None
                
        except Exception as e:
            print(f"❌ SAM response generation failed: {e}")
            return False, None
            
    except Exception as e:
        print(f"❌ SAM response test failed: {e}")
        return False, None

def main():
    """Run the summarize button fix."""
    print("🔧 SAM Summarize Button Fix")
    print("Fixing document access for e1539_wl-330g.pdf")
    print("=" * 50)
    
    # Test 1: Fix memory search
    memory_success, content, metadata = fix_memory_search_attribute_error()
    
    # Test 2: Direct document access
    direct_success, text = test_direct_document_access()
    
    # Test 3: Manual summary creation
    manual_summary = create_manual_summary()
    
    # Test 4: SAM response generation
    sam_success, sam_response = test_sam_response_generation()
    
    print("\n📊 FIX RESULTS SUMMARY")
    print("=" * 25)
    print(f"Memory search fix:     {'✅ SUCCESS' if memory_success else '❌ FAILED'}")
    print(f"Direct document access: {'✅ SUCCESS' if direct_success else '❌ FAILED'}")
    print(f"Manual summary:        {'✅ SUCCESS' if manual_summary else '❌ FAILED'}")
    print(f"SAM response:          {'✅ SUCCESS' if sam_success else '❌ FAILED'}")
    
    print("\n🎯 RECOMMENDED SOLUTION:")
    if direct_success and sam_success:
        print("✅ Use direct PDF access + SAM local model for summarization")
        print("   This bypasses the broken memory integration")
    elif direct_success:
        print("✅ Use direct PDF access + manual summary generation")
        print("   This provides immediate functionality")
    elif memory_success:
        print("✅ Fix memory search attribute handling")
        print("   The document is in memory but attribute access is broken")
    else:
        print("❌ Need to debug document storage and retrieval system")
    
    # Return the best available summary
    if sam_response:
        return sam_response
    elif manual_summary:
        return manual_summary
    elif content:
        return f"Document found in memory: {str(content)[:500]}..."
    else:
        return "Unable to access document content"

if __name__ == "__main__":
    result = main()
    print(f"\n🎯 FINAL RESULT:")
    print(f"Summary available: {'✅ YES' if result else '❌ NO'}")
    if result:
        print("This summary can be used to fix the summarize button functionality.")
