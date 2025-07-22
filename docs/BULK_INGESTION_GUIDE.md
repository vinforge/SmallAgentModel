# SAM Bulk Document Ingestion Guide - Phase 1

## 🎯 Overview

The SAM Bulk Document Ingestion Tool allows you to efficiently process entire folders of documents into SAM's knowledge base. This Phase 1 implementation provides a robust command-line interface for bulk importing documents with intelligent deduplication and state tracking.

## ✨ Key Features

### 🔍 **Smart Deduplication**
- **File Hash Tracking:** Uses SHA256 hashing to detect file changes
- **Modification Detection:** Compares file modification timestamps
- **State Persistence:** SQLite database tracks processed files
- **Skip Processed Files:** Automatically skips unchanged documents

### 📄 **Comprehensive File Support**
- **Documents:** PDF, DOCX, DOC, RTF, TXT, MD
- **Code Files:** PY, JS, HTML, CSS, JSON, XML, YAML
- **Data Files:** CSV, TSV
- **Custom Filtering:** Specify file types to process

### 📊 **Progress Tracking & Statistics**
- **Real-time Progress:** Shows current file being processed
- **Processing Statistics:** Success/failure counts and timing
- **Enrichment Scoring:** Quality assessment for each document
- **Historical Stats:** View past ingestion statistics

### 🔒 **Safe Operation**
- **Dry Run Mode:** Preview what will be processed without changes
- **Error Handling:** Graceful failure recovery
- **Logging:** Comprehensive logs for troubleshooting
- **State Recovery:** Resume interrupted ingestion sessions

## 🚀 Installation & Setup

### Prerequisites
- SAM must be installed and configured
- Python 3.8+ with required dependencies
- Sufficient disk space for state tracking database

### Quick Start
```bash
# Navigate to SAM directory
cd /path/to/SAM

# Make script executable (Unix/macOS)
chmod +x scripts/bulk_ingest.py

# Test with dry run
python scripts/bulk_ingest.py --source /path/to/documents --dry-run
```

## 📋 Usage Examples

### Basic Usage
```bash
# Process all supported files in a folder
python scripts/bulk_ingest.py --source /path/to/documents

# Dry run to preview what will be processed
python scripts/bulk_ingest.py --source /path/to/documents --dry-run

# Process only specific file types
python scripts/bulk_ingest.py --source /path/to/documents --file-types pdf,txt,md

# Enable verbose logging
python scripts/bulk_ingest.py --source /path/to/documents --verbose
```

### Statistics & Monitoring
```bash
# View ingestion statistics
python scripts/bulk_ingest.py --stats

# Check help and all options
python scripts/bulk_ingest.py --help
```

### Advanced Examples
```bash
# Process research papers only
python scripts/bulk_ingest.py --source /research/papers --file-types pdf --verbose

# Process code repositories
python scripts/bulk_ingest.py --source /code/projects --file-types py,js,md,json

# Process documentation folders
python scripts/bulk_ingest.py --source /docs --file-types md,txt,html,pdf
```

## 🖥️ Phase 2: UI Usage (Memory Control Center)

### Accessing the Interface
1. **Open Memory Control Center:** http://localhost:8501
2. **Navigate to Bulk Ingestion:** Select "📁 Bulk Ingestion" from the navigation menu
3. **Choose Operation:** Use the tabs to manage sources, scan, view statistics, or configure settings

### 📂 Source Management
```
1. Click "➕ Add New Source" to expand the form
2. Enter folder path (e.g., /Users/username/Documents)
3. Provide a friendly name (e.g., "Research Papers")
4. Select file types to process
5. Click "➕ Add Source" to save
```

### 🚀 Manual Scanning
```
Individual Source:
1. Go to "🚀 Manual Scan" tab
2. Expand the source you want to scan
3. Choose "Dry Run" for preview or uncheck for actual processing
4. Click "🚀 Scan Source"

Bulk Scanning:
1. Check "Dry Run (Preview Only)" if desired
2. Click "🚀 Scan All Sources"
3. Monitor progress bar and results
```

### 📊 Viewing Statistics
```
1. Go to "📊 Statistics" tab
2. View overview metrics (files, chunks, success rate)
3. Review recent activity timeline
4. Monitor processing performance
```

## 🎛️ Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--source` | Source folder path to ingest | `--source /path/to/docs` |
| `--dry-run` | Simulate without processing | `--dry-run` |
| `--file-types` | Comma-separated file extensions | `--file-types pdf,txt,md` |
| `--stats` | Show statistics and exit | `--stats` |
| `--verbose` | Enable verbose logging | `--verbose` |
| `--help` | Show help message | `--help` |

## 📊 State Tracking & Database

### Database Location
- **Default Path:** `data/ingestion_state.db`
- **Format:** SQLite database
- **Automatic Creation:** Created on first run

### Database Schema
```sql
CREATE TABLE processed_files (
    filepath TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    last_modified REAL NOT NULL,
    processed_at TEXT NOT NULL,
    chunks_created INTEGER DEFAULT 0,
    enrichment_score REAL DEFAULT 0.0,
    status TEXT DEFAULT 'success'
);
```

### State Management
- **Automatic Tracking:** Files are automatically marked as processed
- **Change Detection:** Modified files are reprocessed
- **Error Recovery:** Failed files can be retried
- **Statistics:** Historical processing data available

## 🔧 Supported File Types

### Document Formats
- **PDF:** `.pdf` - Portable Document Format
- **Word:** `.docx`, `.doc` - Microsoft Word documents
- **Text:** `.txt`, `.md`, `.rtf` - Plain text and markup

### Code & Development
- **Python:** `.py` - Python source code
- **JavaScript:** `.js` - JavaScript files
- **Web:** `.html`, `.htm`, `.css` - Web documents
- **Data:** `.json`, `.xml`, `.yaml`, `.yml` - Structured data
- **Tabular:** `.csv`, `.tsv` - Spreadsheet data

### Custom File Types
Use the `--file-types` option to specify custom extensions:
```bash
python scripts/bulk_ingest.py --source /path --file-types pdf,docx,py,md
```

## 📈 Processing Pipeline

### 1. **File Discovery**
- Recursively scans source folder
- Filters by supported/specified file types
- Sorts files for consistent processing order

### 2. **Deduplication Check**
- Calculates SHA256 hash of file content
- Compares with stored hash and modification time
- Skips unchanged files automatically

### 3. **Document Processing**
- Processes through SAM's multimodal pipeline
- Extracts content, metadata, and structure
- Generates embeddings and enrichment scores

### 4. **Memory Storage**
- Stores document summary in memory system
- Creates searchable content blocks
- Adds metadata and tags for retrieval

### 5. **State Update**
- Records processing results in database
- Updates statistics and metrics
- Logs success/failure status

## 📊 Output & Results

### Processing Summary
```
🎉 Bulk Ingestion Summary:
   📄 Processed: 15
   ⏭️ Skipped: 3
   ❌ Failed: 1
   📊 Total found: 19
```

### Statistics View
```
📊 Bulk Ingestion Statistics:
   Total files processed: 127
   Total chunks created: 1,543
   Average enrichment score: 0.73
   Successful: 124
   Failed: 3
```

### Log Output
```
2025-06-08 20:16:32 - INFO - 📄 Processing: research_paper.pdf
2025-06-08 20:16:35 - INFO - ✅ Successfully processed research_paper.pdf: 
                              12 content blocks, 15 memory chunks, score: 0.85
```

## 🛠️ Troubleshooting

### Common Issues

#### "No supported files found"
- **Cause:** No files match the supported extensions
- **Solution:** Check file types with `--file-types` or verify folder contents

#### "Failed to initialize SAM components"
- **Cause:** SAM dependencies not properly installed
- **Solution:** Ensure SAM is properly installed and configured

#### "Permission denied" errors
- **Cause:** Insufficient file system permissions
- **Solution:** Check read permissions on source folder and write permissions for state database

#### Processing failures
- **Cause:** Corrupted files or unsupported content
- **Solution:** Check logs for specific error messages, use `--verbose` for details

### Debug Mode
```bash
# Enable verbose logging for troubleshooting
python scripts/bulk_ingest.py --source /path/to/docs --verbose --dry-run
```

### Log Files
- **Location:** `logs/bulk_ingest.log`
- **Format:** Timestamped entries with log levels
- **Rotation:** Appends to existing log file

## 🎉 Phase 2: UI Integration (COMPLETE!)

### Memory Control Center Interface ✅
- **Native Integration:** Bulk Ingestion tab in Memory Control Center
- **Source Management:** Add, edit, remove, and configure document sources
- **Manual Scanning:** On-demand scan operations with progress tracking
- **Statistics Dashboard:** Comprehensive analytics and performance metrics

### Access Phase 2 Features
🌐 **Memory Control Center:** http://localhost:8501
📁 **Bulk Ingestion Tab:** Select "📁 Bulk Ingestion" from navigation

### Phase 2 Features
- ✅ **📂 Source Management:** Complete CRUD operations for document sources
- ✅ **🚀 Manual Scan:** Individual and bulk scanning with progress tracking
- ✅ **📊 Statistics:** Rich analytics dashboard with processing metrics
- ✅ **⚙️ Settings:** Configuration management and preferences

## 🔮 Phase 3: Live Monitoring (Future)

### Planned Features
- Background file system watching
- Automatic processing of new/modified files
- Real-time updates to knowledge base
- Event-driven ingestion pipeline

## 📚 Integration with SAM

### Memory System
- Documents stored as searchable memories
- Automatic tagging and categorization
- Enrichment scoring for quality assessment
- Integration with SAM's Q&A system

### Search & Retrieval
- Processed documents immediately searchable
- Semantic similarity matching
- Source attribution and citations
- Enhanced search with metadata filtering

### Quality Assurance
- Enrichment scoring for content quality
- Processing statistics and metrics
- Error tracking and recovery
- State persistence for reliability

---

## 🎉 Phase 1 Complete!

The SAM Bulk Document Ingestion Tool Phase 1 provides a solid foundation for efficient document processing with:

- ✅ **Robust CLI Interface** with comprehensive options
- ✅ **Smart Deduplication** to avoid reprocessing
- ✅ **State Tracking** with SQLite database
- ✅ **Error Handling** and recovery mechanisms
- ✅ **Integration** with SAM's processing pipeline
- ✅ **Statistics & Monitoring** for operational insights

**Ready for Phase 2:** UI integration in the Memory Control Center!
