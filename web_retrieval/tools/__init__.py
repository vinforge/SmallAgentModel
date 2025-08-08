# SAM Web Retrieval Tools
# Specialized tools for intelligent web content extraction

from .cocoindex_tool import CocoIndexTool
from .search_api_tool import SearchAPITool
from .news_api_tool import NewsAPITool
from .rss_reader_tool import RSSReaderTool
from .url_content_extractor import URLContentExtractor
from .firecrawl_tool import FirecrawlTool

# Task 27: Research Discovery Tools
try:
    from sam.web_retrieval.tools.arxiv_tool import ArxivSearchTool, get_arxiv_tool
    _ARXIV_AVAILABLE = True
except ImportError:
    _ARXIV_AVAILABLE = False

__all__ = [
    'CocoIndexTool',
    'SearchAPITool',
    'NewsAPITool',
    'RSSReaderTool',
    'URLContentExtractor',
    'FirecrawlTool'
]

# Add ArxivSearchTool if available
if _ARXIV_AVAILABLE:
    __all__.extend(['ArxivSearchTool', 'get_arxiv_tool'])
