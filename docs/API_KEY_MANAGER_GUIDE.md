# 🔑 API Key Manager - User Guide

## Overview

The API Key Manager is a user-friendly interface in SAM's Memory Control Center for configuring and managing API keys for enhanced web search and external service integrations.

## Accessing the API Key Manager

1. **Launch Memory Control Center**:
   ```bash
   python launch_memory_ui.py
   ```
   Or visit: http://localhost:8501

2. **Navigate to API Key Manager**:
   - In the sidebar, select "🔑 API Key Manager" from the dropdown menu

## Features

### 📊 Service Status Overview

The dashboard shows real-time status of all configured services:

- **🔍 Serper Search**: Shows "Active" or "Free Mode"
- **📰 News API**: Shows "Active" or "RSS Only"  
- **🧠 CocoIndex**: Shows "Intelligent" or "Legacy"
- **🌐 Search Provider**: Current backend (Serper/DuckDuckGo)

### 🔍 Web Search Configuration

#### Serper API (Recommended)
- **Benefits**: Google-powered results, 2,500 free searches/month
- **Setup**: Get free API key at [serper.dev](https://serper.dev)
- **Configuration**: Enter API key in the secure password field

#### Search Provider Selection
- **Serper**: High-quality Google results (requires API key)
- **DuckDuckGo**: Free, privacy-focused search (no API key needed)

#### Search Depth Control
- **Pages to Process**: 1-10 pages (more = better coverage, slower searches)
- **Recommended**: 5 pages for optimal balance

### 📰 News Services Configuration

#### NewsAPI (Optional)
- **Benefits**: Real-time news from 80,000+ sources
- **Setup**: Get free API key at [newsapi.org](https://newsapi.org)
- **Fallback**: RSS feeds used automatically if not configured

#### RSS Feeds (Always Available)
- **Always free and functional**
- **Major news sources included**
- **No configuration required**

### ⚙️ Advanced Settings

- **Web Retrieval Engine**: Choose between CocoIndex (intelligent) or Legacy tools
- **Request Timeout**: Configure timeout for web requests (10-120 seconds)

### 🧪 Connectivity Testing

Test your API keys and service connectivity:

- **🔍 Test Serper**: Verify Serper API key and connection
- **📰 Test NewsAPI**: Verify NewsAPI key and connection  
- **🧠 Test CocoIndex**: Verify CocoIndex functionality

## Step-by-Step Setup Guide

### 1. Get Your API Keys (Optional but Recommended)

#### For Enhanced Search (Serper):
1. Visit [serper.dev](https://serper.dev)
2. Sign up for free account
3. Copy your API key (starts with `sk-`)

#### For Real-time News (NewsAPI):
1. Visit [newsapi.org](https://newsapi.org)
2. Sign up for free account
3. Copy your API key

### 2. Configure in SAM

1. **Open Memory Control Center** (http://localhost:8501)
2. **Select "🔑 API Key Manager"** from navigation
3. **Enter API Keys**:
   - Paste Serper key in "Serper API Key" field
   - Paste NewsAPI key in "NewsAPI Key" field
4. **Configure Settings**:
   - Set search provider to "serper" for best results
   - Adjust pages to process (5 recommended)
5. **Test Connectivity**:
   - Click "🔍 Test Serper" to verify
   - Click "📰 Test NewsAPI" to verify
6. **Save Configuration**:
   - Click "💾 Save Configuration"

### 3. Verify Setup

After saving, you should see:
- ✅ Green status indicators in Service Overview
- ✅ "Configuration saved successfully!" message
- ✅ Enhanced search results in SAM's web queries

## Configuration Options

### Free Mode (No API Keys)
- **Search**: DuckDuckGo (unlimited, privacy-focused)
- **News**: RSS feeds from major sources
- **Intelligence**: Full CocoIndex processing
- **Status**: Fully functional, good quality

### Enhanced Mode (With API Keys)
- **Search**: Google-powered via Serper (2,500/month free)
- **News**: Real-time NewsAPI (1,000/month free)
- **Intelligence**: Full CocoIndex processing
- **Status**: Optimal quality and coverage

## Troubleshooting

### Common Issues

#### "API Key Test Failed"
- **Check**: Verify API key is correct (copy/paste carefully)
- **Check**: Ensure API key has remaining quota
- **Check**: Test internet connectivity

#### "CocoIndex Not Available"
- **Solution**: CocoIndex will auto-install on first use
- **Alternative**: System falls back to legacy tools automatically

#### "Configuration Not Saving"
- **Check**: Ensure you clicked "💾 Save Configuration"
- **Check**: Look for error messages in the interface
- **Try**: Reload the page and try again

### Getting Help

1. **Test Connectivity**: Use built-in test buttons to diagnose issues
2. **Check Logs**: Look for error messages in the interface
3. **Reset Configuration**: Use "🔄 Reset Changes" if needed
4. **Reload**: Use "🔄 Reload Configuration" to refresh from file

## Security Notes

- **API keys are stored securely** in SAM's configuration files
- **Keys are masked** in the interface (shown as password fields)
- **No keys are logged** or transmitted except to their respective services
- **Local storage only** - keys never leave your system

## Benefits of Configuration

### Without API Keys (Free Mode)
- ✅ Unlimited searches via DuckDuckGo
- ✅ Privacy-focused search
- ✅ RSS news feeds
- ✅ Full CocoIndex intelligence
- ✅ Zero cost

### With API Keys (Enhanced Mode)
- 🚀 Google-quality search results
- 🚀 Real-time news from 80,000+ sources
- 🚀 Faster response times
- 🚀 Better relevance ranking
- 🚀 More comprehensive coverage

## Integration with SAM

The API Key Manager integrates seamlessly with:

- **Web Search**: Enhanced results in chat queries
- **News Queries**: Real-time news when asking about current events
- **Content Vetting**: Better source material for analysis
- **Memory Consolidation**: Higher quality web knowledge integration

## Updates and Maintenance

- **Automatic Fallbacks**: System gracefully handles API failures
- **Usage Monitoring**: Track API usage through service dashboards
- **Key Rotation**: Update keys anytime through the interface
- **Configuration Backup**: Settings are preserved across SAM updates

---

**Need Help?** The API Key Manager includes built-in testing, status monitoring, and helpful tooltips to guide you through the configuration process.

**Ready to Start?** Launch the Memory Control Center and navigate to "🔑 API Key Manager" to begin enhancing your SAM experience!
