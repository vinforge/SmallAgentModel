# SAM (Small Agent Model) Makefile
# Sprint 13: Deployment and Development Automation

.PHONY: help install install-all install-dpo install-dev start stop status clean test docker-build docker-run backup restore

# Default target
help:
	@echo "🤖 SAM (Small Agent Model) - Available Commands"
	@echo "================================================"
	@echo "Development:"
	@echo "  install       - Install core/runtime dependencies"
	@echo "  install-all   - Install core + DPO add-ons"
	@echo "  install-dpo   - Install DPO add-ons (plus core)"
	@echo "  install-dev   - Install dev tools (pytest, black)"
	@echo "  start         - Start SAM locally"
	@echo "  stop          - Stop SAM processes"
	@echo "  status        - Show system status"
	@echo "  test          - Run tests"
	@echo "  clean         - Clean temporary files"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker Compose"
	@echo "  docker-stop  - Stop Docker containers"
	@echo "  docker-logs  - View Docker logs"
	@echo ""
	@echo "Configuration:"
	@echo "  config-show  - Show current configuration"
	@echo "  config-validate - Validate configuration"
	@echo "  config-export - Export configuration"
	@echo ""
	@echo "Memory Management:"
	@echo "  backup      - Create memory snapshot"
	@echo "  restore     - Restore from snapshot"
	@echo "  memory-stats - Show memory statistics"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-prod - Deploy to production"
	@echo "  deploy-dev  - Deploy to development"

# Development Commands
install:
	@echo "📦 Installing SAM dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"


install-all:
	@echo "📦 Installing SAM core + DPO dependencies..."
	pip install -r requirements_all.txt
	@echo "✅ All dependencies installed"

install-dpo:
	@echo "📦 Installing SAM DPO add-ons..."
	pip install -r requirements.txt -r requirements_dpo.txt
	@echo "✅ DPO add-ons installed"

install-dev:
	@echo "🛠 Installing development tools..."
	pip install -r requirements_dev.txt
	@echo "✅ Dev tools installed"

start:
	@echo "🚀 Starting SAM..."
	python start_sam.py

stop:
	@echo "🛑 Stopping SAM processes..."
	pkill -f "start_sam.py" || true
	pkill -f "launch_web_ui.py" || true
	pkill -f "streamlit run ui/memory_app.py" || true
	@echo "✅ SAM processes stopped"

status:
	@echo "🔍 SAM System Status:"
	python cli_tools.py system status

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v
	@echo "✅ Tests completed"

clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	@echo "✅ Cleanup completed"

# Docker Commands
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t sam:latest .
	@echo "✅ Docker image built"

docker-run:
	@echo "🐳 Starting SAM with Docker Compose..."
	docker-compose up -d
	@echo "✅ SAM started in Docker"
	@echo "🌐 Chat Interface: http://localhost:5001"
	@echo "🧠 Memory Control: http://localhost:8501"

docker-stop:
	@echo "🐳 Stopping Docker containers..."
	docker-compose down
	@echo "✅ Docker containers stopped"

docker-logs:
	@echo "📋 Docker logs:"
	docker-compose logs -f

# Configuration Commands
config-show:
	@echo "📋 Current Configuration:"
	python cli_tools.py config show

config-validate:
	@echo "🔍 Validating Configuration:"
	python cli_tools.py config validate

config-export:
	@echo "📤 Exporting Configuration:"
	python cli_tools.py config export config_backup_$(shell date +%Y%m%d_%H%M%S).json
	@echo "✅ Configuration exported"

# Memory Management Commands
backup:
	@echo "💾 Creating memory snapshot..."
	python cli_tools.py memory snapshot --name "manual_backup_$(shell date +%Y%m%d_%H%M%S)"
	@echo "✅ Memory snapshot created"

restore:
	@echo "📥 Available snapshots:"
	python cli_tools.py memory list-snapshots
	@echo "Use: python cli_tools.py memory restore <snapshot_file>"

memory-stats:
	@echo "🧠 Memory Statistics:"
	python cli_tools.py memory stats

# Deployment Commands
deploy-prod:
	@echo "🚀 Deploying to Production..."
	@echo "1. Creating backup..."
	$(MAKE) backup
	@echo "2. Validating configuration..."
	python cli_tools.py config validate
	@echo "3. Building Docker image..."
	docker build -t sam:prod .
	@echo "4. Starting production services..."
	docker-compose -f docker-compose.prod.yml up -d
	@echo "✅ Production deployment completed"

deploy-dev:
	@echo "🛠️ Deploying to Development..."
	@echo "1. Installing dependencies..."
	$(MAKE) install
	@echo "2. Validating configuration..."
	python cli_tools.py config validate
	@echo "3. Starting development server..."
	$(MAKE) start

# Health and Monitoring
health:
	@echo "🏥 Health Check:"
	curl -s http://localhost:5001/health | python -m json.tool || echo "❌ Health check failed"

monitor:
	@echo "📊 System Monitoring:"
	python cli_tools.py system metrics --hours 1

# Setup Commands
setup-dev:
	@echo "🛠️ Setting up development environment..."
	cp .env.template .env
	mkdir -p logs config memory_store backups
	$(MAKE) install
	@echo "✅ Development environment ready"
	@echo "📝 Edit .env file with your settings"
	@echo "🚀 Run 'make start' to begin"

setup-prod:
	@echo "🏭 Setting up production environment..."
	cp .env.template .env
	mkdir -p logs config memory_store backups
	@echo "✅ Production environment ready"
	@echo "📝 Edit .env file with production settings"
	@echo "🐳 Run 'make docker-run' to start"

# Maintenance Commands
update:
	@echo "🔄 Updating SAM..."
	git pull origin main
	$(MAKE) install
	$(MAKE) config-validate
	@echo "✅ Update completed"

reset:
	@echo "⚠️ Resetting SAM (this will clear all data)..."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	$(MAKE) stop
	rm -rf memory_store/* logs/* backups/*
	python cli_tools.py onboarding reset
	@echo "✅ SAM reset completed"

# Quick Start
quick-start: setup-dev start
	@echo "🎉 SAM is now running!"
	@echo "🌐 Chat Interface: http://localhost:5001"
	@echo "🧠 Memory Control: http://localhost:8501"
