#!/usr/bin/env python3
"""
API Key Manager for SAM Memory Control Center

This module provides a user-friendly interface for managing API keys
for various web retrieval and external service integrations.

Features:
- Secure API key configuration
- Service status monitoring
- Usage statistics and quotas
- Test connectivity functionality
- Configuration validation

Author: SAM Development Team
Version: 1.0.0
"""

import streamlit as st
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import sys

# Add SAM to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager
from web_retrieval.tools.cocoindex_tool import CocoIndexTool
from web_retrieval.tools.search_api_tool import SearchAPITool
from web_retrieval.tools.news_api_tool import NewsAPITool
from web_retrieval.tools.firecrawl_tool import FirecrawlTool

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manages API keys for SAM's external service integrations."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        
    def render(self):
        """Render the API key management interface."""
        st.title("🔑 API Key Manager")
        st.markdown("Configure API keys for enhanced web search and external services")
        
        # Service overview
        self._render_service_overview()
        
        # API key configuration sections
        col1, col2 = st.columns(2)

        with col1:
            self._render_web_search_config()
            self._render_firecrawl_config()

        with col2:
            self._render_news_services_config()
        
        # Advanced settings
        self._render_advanced_settings()
        
        # Test connectivity section
        self._render_connectivity_tests()
        
        # Save configuration
        self._render_save_section()
    
    def _render_service_overview(self):
        """Render service status overview."""
        st.subheader("📊 Service Status Overview")
        
        # Create status cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            serper_status = "✅ Active" if self.config.serper_api_key else "⚠️ Free Mode"
            st.metric(
                "🔍 Serper Search",
                serper_status,
                help="Google-powered search API for enhanced results"
            )
        
        with col2:
            newsapi_status = "✅ Active" if self.config.newsapi_api_key else "📰 RSS Only"
            st.metric(
                "📰 News API",
                newsapi_status,
                help="Real-time news aggregation service"
            )
        
        with col3:
            cocoindex_status = "🧠 Intelligent" if self.config.web_retrieval_provider == "cocoindex" else "🔧 Legacy"
            st.metric(
                "🧠 CocoIndex",
                cocoindex_status,
                help="Intelligent web content extraction and indexing"
            )
        
        with col4:
            firecrawl_status = "🔥 Active" if getattr(self.config, 'firecrawl_api_key', None) else "🕷️ Basic Mode"
            st.metric(
                "🔥 Firecrawl",
                firecrawl_status,
                help="Advanced web crawling with anti-bot mechanisms"
            )

        with col5:
            provider = self.config.cocoindex_search_provider.title()
            st.metric(
                "🌐 Search Provider",
                provider,
                help="Current search backend being used"
            )
        
        st.markdown("---")
    
    def _render_web_search_config(self):
        """Render web search API configuration."""
        st.subheader("🔍 Web Search Configuration")
        
        # Serper API configuration
        with st.expander("🚀 Serper API (Recommended)", expanded=True):
            st.markdown("""
            **Serper provides Google-powered search results with:**
            - 2,500 free searches per month
            - High-quality, comprehensive results
            - Fast response times
            - Enhanced relevance ranking
            
            [Get your free API key at serper.dev →](https://serper.dev)
            """)
            
            current_serper_key = self.config.serper_api_key
            serper_key = st.text_input(
                "Serper API Key",
                value=current_serper_key,
                type="password",
                help="Enter your Serper API key (starts with 'sk-')",
                key="serper_api_key_input"
            )
            
            if serper_key and serper_key != current_serper_key:
                st.session_state.config_changed = True
                st.session_state.new_serper_key = serper_key
            
            # Show current usage info if key is configured
            if current_serper_key:
                st.success("✅ Serper API key configured")
                st.caption("💡 Your searches will use Google's search index for best results")
            else:
                st.info("ℹ️ Without Serper, searches will use free DuckDuckGo (still fully functional)")
        
        # Search provider selection
        st.markdown("**Search Provider Selection:**")
        current_provider = self.config.cocoindex_search_provider
        search_provider = st.selectbox(
            "Default Search Provider",
            options=["duckduckgo", "serper"],
            index=0 if current_provider == "duckduckgo" else 1,
            help="Choose your preferred search backend",
            key="search_provider_select"
        )
        
        if search_provider != current_provider:
            st.session_state.config_changed = True
            st.session_state.new_search_provider = search_provider
        
        # Search depth configuration
        st.markdown("**Search Depth:**")
        current_pages = self.config.cocoindex_num_pages
        num_pages = st.slider(
            "Pages to Process",
            min_value=1,
            max_value=10,
            value=current_pages,
            help="More pages = better coverage but slower searches",
            key="num_pages_slider"
        )
        
        if num_pages != current_pages:
            st.session_state.config_changed = True
            st.session_state.new_num_pages = num_pages

    def _render_firecrawl_config(self):
        """Render Firecrawl API configuration."""
        st.subheader("🔥 Firecrawl Configuration")

        # Firecrawl API configuration
        with st.expander("🔥 Firecrawl API (Advanced Crawling)", expanded=True):
            st.markdown("""
            **Firecrawl provides advanced web crawling with:**
            - Anti-bot mechanisms and proxy rotation
            - JavaScript rendering for dynamic content
            - Interactive content extraction (forms, logins)
            - Batch processing for multiple URLs
            - PDF and document extraction
            - Full website crawling capabilities

            [Get your API key at firecrawl.dev →](https://firecrawl.dev)
            """)

            current_firecrawl_key = getattr(self.config, 'firecrawl_api_key', '')
            firecrawl_key = st.text_input(
                "Firecrawl API Key",
                value=current_firecrawl_key,
                type="password",
                help="Enter your Firecrawl API key (starts with 'fc-')",
                key="firecrawl_api_key_input"
            )

            if firecrawl_key and firecrawl_key != current_firecrawl_key:
                st.session_state.config_changed = True
                st.session_state.new_firecrawl_key = firecrawl_key

            # Show current usage info if key is configured
            if current_firecrawl_key:
                st.success("✅ Firecrawl API key configured")
                st.caption("💡 Complex sites and interactive content will use Firecrawl")
            else:
                st.info("ℹ️ Without Firecrawl, complex sites may be harder to access")

        # Firecrawl settings
        st.markdown("**Firecrawl Settings:**")
        current_timeout = getattr(self.config, 'firecrawl_timeout', 30)
        firecrawl_timeout = st.slider(
            "Request Timeout (seconds)",
            min_value=10,
            max_value=120,
            value=current_timeout,
            help="Timeout for Firecrawl requests",
            key="firecrawl_timeout_slider"
        )

        if firecrawl_timeout != current_timeout:
            st.session_state.config_changed = True
            st.session_state.new_firecrawl_timeout = firecrawl_timeout

    def _render_news_services_config(self):
        """Render news services API configuration."""
        st.subheader("📰 News Services Configuration")
        
        # NewsAPI configuration
        with st.expander("📰 NewsAPI (Optional)", expanded=True):
            st.markdown("""
            **NewsAPI provides real-time news with:**
            - 1,000 free requests per month
            - Real-time news from 80,000+ sources
            - Category and source filtering
            - Historical news access
            
            [Get your free API key at newsapi.org →](https://newsapi.org)
            """)
            
            current_newsapi_key = self.config.newsapi_api_key
            newsapi_key = st.text_input(
                "NewsAPI Key",
                value=current_newsapi_key,
                type="password",
                help="Enter your NewsAPI key",
                key="newsapi_key_input"
            )
            
            if newsapi_key and newsapi_key != current_newsapi_key:
                st.session_state.config_changed = True
                st.session_state.new_newsapi_key = newsapi_key
            
            if current_newsapi_key:
                st.success("✅ NewsAPI key configured")
                st.caption("💡 News queries will use real-time NewsAPI data")
            else:
                st.info("ℹ️ Without NewsAPI, news queries will use RSS feeds (still functional)")
        
        # RSS fallback info
        with st.expander("📡 RSS Feeds (Always Available)", expanded=False):
            st.markdown("""
            **RSS feeds provide reliable news access:**
            - Always free and available
            - Major news sources (BBC, NYT, CNN)
            - Topic-specific feeds
            - No API limits or quotas
            
            RSS feeds are used automatically when NewsAPI is unavailable.
            """)
            
            st.success("✅ RSS feeds always available as fallback")
    
    def _render_advanced_settings(self):
        """Render advanced configuration settings."""
        st.subheader("⚙️ Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Web retrieval provider
            current_provider = self.config.web_retrieval_provider
            web_provider = st.selectbox(
                "Web Retrieval Engine",
                options=["cocoindex", "legacy"],
                index=0 if current_provider == "cocoindex" else 1,
                help="Choose between intelligent CocoIndex or legacy tools",
                key="web_provider_select"
            )
            
            if web_provider != current_provider:
                st.session_state.config_changed = True
                st.session_state.new_web_provider = web_provider
        
        with col2:
            # Request timeout
            # Ensure the current value doesn't exceed the UI maximum
            current_timeout = min(self.config.request_timeout_seconds, 300)
            timeout = st.number_input(
                "Request Timeout (seconds)",
                min_value=10,
                max_value=300,
                value=current_timeout,
                help="Timeout for web requests (10-300 seconds)",
                key="timeout_input"
            )
            
            if timeout != self.config.request_timeout_seconds:
                st.session_state.config_changed = True
                st.session_state.new_timeout = timeout
    
    def _render_connectivity_tests(self):
        """Render connectivity testing section."""
        st.subheader("🧪 Test Connectivity")
        st.markdown("Test your API keys and service connectivity")
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🔍 Test Serper", key="test_serper_btn"):
                self._test_serper_connection()

        with col2:
            if st.button("📰 Test NewsAPI", key="test_newsapi_btn"):
                self._test_newsapi_connection()

        with col3:
            if st.button("🔥 Test Firecrawl", key="test_firecrawl_btn"):
                self._test_firecrawl_connection()

        with col4:
            if st.button("🧠 Test CocoIndex", key="test_cocoindex_btn"):
                self._test_cocoindex_connection()
    
    def _render_save_section(self):
        """Render configuration save section."""
        st.markdown("---")
        
        # Show if changes are pending
        if st.session_state.get('config_changed', False):
            st.warning("⚠️ You have unsaved changes")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Save Configuration", type="primary", key="save_config_btn"):
                    self._save_configuration()
            
            with col2:
                if st.button("🔄 Reset Changes", key="reset_changes_btn"):
                    self._reset_changes()
            
            with col3:
                if st.button("📋 Preview Changes", key="preview_changes_btn"):
                    self._preview_changes()
        else:
            st.success("✅ Configuration is up to date")
            
            if st.button("🔄 Reload Configuration", key="reload_config_btn"):
                self._reload_configuration()
    
    def _test_serper_connection(self):
        """Test Serper API connection."""
        try:
            api_key = st.session_state.get('new_serper_key', self.config.serper_api_key)
            
            if not api_key:
                st.error("❌ No Serper API key configured")
                return
            
            with st.spinner("Testing Serper connection..."):
                search_tool = SearchAPITool(api_key=api_key)
                result = search_tool.search("test query", num_results=1)
                
                if result['success']:
                    st.success("✅ Serper API connection successful!")
                    st.caption(f"Retrieved {result.get('total_results', 0)} results")
                else:
                    st.error(f"❌ Serper API test failed: {result.get('error', 'Unknown error')}")
                    
        except Exception as e:
            st.error(f"❌ Serper test error: {str(e)}")
    
    def _test_newsapi_connection(self):
        """Test NewsAPI connection."""
        try:
            api_key = st.session_state.get('new_newsapi_key', self.config.newsapi_api_key)
            
            if not api_key:
                st.error("❌ No NewsAPI key configured")
                return
            
            with st.spinner("Testing NewsAPI connection..."):
                news_tool = NewsAPITool(api_key=api_key)
                result = news_tool.get_news("test", num_articles=1)
                
                if result['success']:
                    st.success("✅ NewsAPI connection successful!")
                    st.caption(f"Retrieved {len(result.get('articles', []))} articles")
                else:
                    st.error(f"❌ NewsAPI test failed: {result.get('error', 'Unknown error')}")
                    
        except Exception as e:
            st.error(f"❌ NewsAPI test error: {str(e)}")

    def _test_firecrawl_connection(self):
        """Test Firecrawl API connection."""
        try:
            api_key = st.session_state.get('new_firecrawl_key', getattr(self.config, 'firecrawl_api_key', ''))

            if not api_key:
                st.error("❌ No Firecrawl API key configured")
                return

            with st.spinner("Testing Firecrawl connection..."):
                firecrawl_tool = FirecrawlTool(api_key=api_key)

                if firecrawl_tool.firecrawl_available:
                    st.success("✅ Firecrawl is available and ready!")
                    st.caption("🔥 Advanced web crawling capabilities enabled")
                else:
                    st.error("❌ Firecrawl is not available - check API key or install firecrawl-py")

        except Exception as e:
            st.error(f"❌ Firecrawl test error: {str(e)}")

    def _test_cocoindex_connection(self):
        """Test CocoIndex functionality."""
        try:
            with st.spinner("Testing CocoIndex..."):
                search_provider = st.session_state.get('new_search_provider', self.config.cocoindex_search_provider)
                api_key = st.session_state.get('new_serper_key', self.config.serper_api_key) if search_provider == "serper" else None
                
                tool = CocoIndexTool(
                    api_key=api_key,
                    search_provider=search_provider,
                    num_pages=2
                )
                
                if tool.cocoindex_available:
                    st.success("✅ CocoIndex is available and ready!")
                    st.caption(f"Using {search_provider} as search provider")
                else:
                    st.error("❌ CocoIndex is not available - may need installation")
                    
        except Exception as e:
            st.error(f"❌ CocoIndex test error: {str(e)}")
    
    def _save_configuration(self):
        """Save the current configuration changes."""
        try:
            # Update configuration with new values
            if hasattr(st.session_state, 'new_serper_key'):
                self.config.serper_api_key = st.session_state.new_serper_key
            
            if hasattr(st.session_state, 'new_newsapi_key'):
                self.config.newsapi_api_key = st.session_state.new_newsapi_key

            if hasattr(st.session_state, 'new_firecrawl_key'):
                self.config.firecrawl_api_key = st.session_state.new_firecrawl_key

            if hasattr(st.session_state, 'new_firecrawl_timeout'):
                self.config.firecrawl_timeout = st.session_state.new_firecrawl_timeout

            if hasattr(st.session_state, 'new_search_provider'):
                self.config.cocoindex_search_provider = st.session_state.new_search_provider
            
            if hasattr(st.session_state, 'new_num_pages'):
                self.config.cocoindex_num_pages = st.session_state.new_num_pages
            
            if hasattr(st.session_state, 'new_web_provider'):
                self.config.web_retrieval_provider = st.session_state.new_web_provider
            
            if hasattr(st.session_state, 'new_timeout'):
                self.config.request_timeout_seconds = st.session_state.new_timeout
            
            # Save to file
            self.config_manager.save_config(self.config)
            
            # Clear change flags
            st.session_state.config_changed = False
            for key in list(st.session_state.keys()):
                if key.startswith('new_'):
                    del st.session_state[key]
            
            st.success("✅ Configuration saved successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to save configuration: {str(e)}")
    
    def _reset_changes(self):
        """Reset all pending changes."""
        st.session_state.config_changed = False
        for key in list(st.session_state.keys()):
            if key.startswith('new_'):
                del st.session_state[key]
        st.rerun()
    
    def _preview_changes(self):
        """Preview pending configuration changes."""
        st.subheader("📋 Pending Changes")
        
        changes = {}
        if hasattr(st.session_state, 'new_serper_key'):
            changes['Serper API Key'] = "Updated" if st.session_state.new_serper_key else "Removed"
        
        if hasattr(st.session_state, 'new_newsapi_key'):
            changes['NewsAPI Key'] = "Updated" if st.session_state.new_newsapi_key else "Removed"

        if hasattr(st.session_state, 'new_firecrawl_key'):
            changes['Firecrawl API Key'] = "Updated" if st.session_state.new_firecrawl_key else "Removed"

        if hasattr(st.session_state, 'new_firecrawl_timeout'):
            changes['Firecrawl Timeout'] = f"{st.session_state.new_firecrawl_timeout}s"

        if hasattr(st.session_state, 'new_search_provider'):
            changes['Search Provider'] = st.session_state.new_search_provider
        
        if hasattr(st.session_state, 'new_num_pages'):
            changes['Pages to Process'] = st.session_state.new_num_pages
        
        if hasattr(st.session_state, 'new_web_provider'):
            changes['Web Retrieval Engine'] = st.session_state.new_web_provider
        
        if changes:
            for setting, value in changes.items():
                st.write(f"• **{setting}**: {value}")
        else:
            st.info("No changes to preview")
    
    def _reload_configuration(self):
        """Reload configuration from file."""
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        st.success("✅ Configuration reloaded from file")
        st.rerun()

def render_api_key_manager():
    """Render the API key manager interface."""
    manager = APIKeyManager()
    manager.render()
