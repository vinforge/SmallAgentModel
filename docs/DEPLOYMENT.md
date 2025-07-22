# SAM Deployment Guide

## 🚀 Quick Start

### Local Development

1. **Clone and Setup**
   ```bash
   git clone https://github.com/forge-1825/SAM.git
   cd SAM
   pip install -r requirements.txt
   ```

2. **Start SAM**
   ```bash
   python start_sam.py
   ```

3. **Access Interfaces**
   - Chat Interface: http://localhost:5001
   - Memory Control Center: http://localhost:8501

### Docker Deployment

1. **Using Docker Compose (Recommended)**
   ```bash
   # Copy environment template
   cp .env.template .env
   
   # Edit .env with your settings
   nano .env
   
   # Start services
   docker-compose up -d
   ```

2. **Using Docker Only**
   ```bash
   # Build image
   docker build -t sam:latest .
   
   # Run container
   docker run -d \
     --name sam \
     -p 5001:5001 \
     -p 8501:8501 \
     -v sam_config:/app/config \
     -v sam_memory:/app/memory_store \
     sam:latest
   ```

## 📋 Configuration

### Environment Variables

Copy `.env.template` to `.env` and customize:

```bash
# Agent mode: solo or collaborative
SAM_AGENT_MODE=solo

# Memory backend: simple, faiss, chroma
SAM_MEMORY_BACKEND=simple

# Server ports
SAM_CHAT_PORT=5001
SAM_MEMORY_UI_PORT=8501

# Model settings
SAM_MODEL_PROVIDER=ollama
SAM_MODEL_NAME=hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M
```

### Configuration Management

Use the CLI tools for configuration management:

```bash
# Show current configuration
python cli_tools.py config show

# Update a setting
python cli_tools.py config set --key agent_mode --value collaborative

# Validate configuration
python cli_tools.py config validate

# Export configuration
python cli_tools.py config export config_backup.json

# Create template
python cli_tools.py config template production_config.json --type production
```

## 🐳 Docker Configuration

### Docker Compose Services

- **sam**: Main SAM application
- **ollama**: Local LLM service
- **redis**: (Optional) Caching layer
- **postgres**: (Optional) Advanced memory storage

### Volumes

- `sam_config`: Configuration files
- `sam_memory`: Memory storage
- `sam_logs`: Application logs
- `sam_backups`: Memory snapshots

### Health Checks

All services include health checks:
- SAM: `http://localhost:5001/health`
- Ollama: `http://localhost:11434/api/tags`

## 🔧 System Service (Linux)

### Systemd Service

1. **Install Service**
   ```bash
   sudo cp sam.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable sam.service
   ```

2. **Start Service**
   ```bash
   sudo systemctl start sam.service
   sudo systemctl status sam.service
   ```

3. **View Logs**
   ```bash
   sudo journalctl -u sam.service -f
   ```

## 📊 Monitoring

### Health Endpoints

- **Health Check**: `GET /health`
  - Returns overall system health
  - HTTP 200: Healthy
  - HTTP 503: Critical issues

- **Detailed Status**: `GET /status`
  - Comprehensive system information
  - Memory statistics
  - Configuration summary
  - Component health

### CLI Monitoring

```bash
# System status
python cli_tools.py system status

# System metrics
python cli_tools.py system metrics --hours 24

# Memory statistics
python cli_tools.py memory stats
```

## 💾 Backup & Recovery

### Memory Snapshots

```bash
# Create snapshot
python cli_tools.py memory snapshot --name "before_update"

# List snapshots
python cli_tools.py memory list-snapshots

# Restore snapshot
python cli_tools.py memory restore backup_snapshot.zip

# Export to JSON
python cli_tools.py memory export-json memory_export.json
```

### Automated Backups

Set up automated backups with cron:

```bash
# Add to crontab
0 2 * * * cd /opt/sam && python cli_tools.py memory snapshot --name "daily_$(date +\%Y\%m\%d)"
```

## 🔐 Security

### Authentication

Enable authentication in production:

```bash
# Generate secret key
python -c "import secrets; print(secrets.token_hex(32))"

# Update configuration
python cli_tools.py config set --key enable_auth --value true
python cli_tools.py config set --key auth_secret_key --value "your_secret_key"
```

### Collaboration Mode

For multi-agent collaboration:

1. **Generate Collaboration Key**
   ```bash
   python -c "
   from config.agent_mode import get_mode_controller
   controller = get_mode_controller()
   key = controller.generate_collaboration_key()
   print(f'Collaboration key: {key}')
   "
   ```

2. **Switch to Collaborative Mode**
   ```bash
   python cli_tools.py config set --key agent_mode --value collaborative
   ```

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :5001
   
   # Change port in configuration
   python cli_tools.py config set --key chat_port --value 5002
   ```

2. **Memory Issues**
   ```bash
   # Check memory statistics
   python cli_tools.py memory stats
   
   # Create snapshot before cleanup
   python cli_tools.py memory snapshot --name "before_cleanup"
   ```

3. **Model Connection Issues**
   ```bash
   # Check Ollama status
   curl http://localhost:11434/api/tags
   
   # Update model URL
   python cli_tools.py config set --key model_api_url --value "http://your-ollama-host:11434"
   ```

### Log Files

- Application logs: `logs/sam.log`
- Launcher logs: `logs/sam_launcher.log`
- Web UI logs: `web_ui/curiosity.log`

### Health Checks

```bash
# Check system health
curl http://localhost:5001/health

# Get detailed status
curl http://localhost:5001/status | jq
```

## 📈 Performance Tuning

### Memory Backend Selection

- **Simple**: Best for development and small datasets
- **FAISS**: Better performance for large memory stores
- **Chroma**: Advanced features and persistence

### Resource Limits

Configure resource limits in Docker:

```yaml
services:
  sam:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'
```

### Concurrent Requests

Adjust for your load:

```bash
python cli_tools.py config set --key max_concurrent_requests --value 50
```

## 🔄 Updates

### Application Updates

1. **Backup Current State**
   ```bash
   python cli_tools.py memory snapshot --name "pre_update_$(date +%Y%m%d)"
   python cli_tools.py config export config_backup.json
   ```

2. **Update Code**
   ```bash
   git pull origin main
   pip install -r requirements.txt
   ```

3. **Restart Services**
   ```bash
   # Docker Compose
   docker-compose down && docker-compose up -d
   
   # Systemd
   sudo systemctl restart sam.service
   ```

### Configuration Updates

Use the CLI tools to safely update configuration:

```bash
# Validate before applying
python cli_tools.py config validate

# Apply updates
python cli_tools.py config import new_config.json --merge
```

## 📞 Support

### Getting Help

1. **Check Logs**: Review application logs for errors
2. **Health Status**: Use health endpoints to diagnose issues
3. **CLI Tools**: Use built-in diagnostic commands
4. **Documentation**: Refer to component-specific documentation

### Reporting Issues

Include the following information:
- System status output
- Relevant log entries
- Configuration summary
- Steps to reproduce

```bash
# Collect diagnostic information
python cli_tools.py system status > diagnostic_info.txt
python cli_tools.py config show >> diagnostic_info.txt
```
