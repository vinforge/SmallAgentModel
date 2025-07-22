#!/bin/bash
# SAM Docker Management Script
# Comprehensive management tool for SAM Docker deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="sam"

# Helper functions
print_header() {
    echo -e "${BLUE}=================================="
    echo -e "🐳 SAM Docker Management"
    echo -e "==================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Docker and Docker Compose are installed
check_dependencies() {
    print_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

# Build SAM Docker image
build() {
    print_info "Building SAM Docker image..."
    docker-compose -p $PROJECT_NAME build --no-cache
    print_success "SAM Docker image built successfully"
}

# Start SAM services
start() {
    print_info "Starting SAM services..."
    docker-compose -p $PROJECT_NAME up -d
    
    print_info "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    if docker-compose -p $PROJECT_NAME ps | grep -q "Up"; then
        print_success "SAM services started successfully"
        show_urls
    else
        print_error "Failed to start SAM services"
        docker-compose -p $PROJECT_NAME logs
        exit 1
    fi
}

# Stop SAM services
stop() {
    print_info "Stopping SAM services..."
    docker-compose -p $PROJECT_NAME down
    print_success "SAM services stopped"
}

# Restart SAM services
restart() {
    print_info "Restarting SAM services..."
    stop
    start
}

# Show service status
status() {
    print_info "SAM Services Status:"
    docker-compose -p $PROJECT_NAME ps
    
    echo ""
    print_info "Container Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" \
        $(docker-compose -p $PROJECT_NAME ps -q) 2>/dev/null || echo "No containers running"
}

# Show logs
logs() {
    local service=${1:-""}
    if [ -n "$service" ]; then
        print_info "Showing logs for $service..."
        docker-compose -p $PROJECT_NAME logs -f "$service"
    else
        print_info "Showing logs for all services..."
        docker-compose -p $PROJECT_NAME logs -f
    fi
}

# Show access URLs
show_urls() {
    echo ""
    print_info "SAM is now accessible at:"
    echo -e "${GREEN}🌟 Main SAM Interface:     http://localhost:8502${NC}"
    echo -e "${GREEN}🧠 Memory Control Center: http://localhost:8501${NC}"
    echo -e "${GREEN}🎯 Setup/Welcome Page:    http://localhost:8503${NC}"
    echo -e "${GREEN}📊 Health Check:          http://localhost:8502/health${NC}"
    echo ""
}

# Backup SAM data
backup() {
    local backup_name="sam_backup_$(date +%Y%m%d_%H%M%S)"
    local backup_dir="./backups/$backup_name"
    
    print_info "Creating backup: $backup_name"
    mkdir -p "$backup_dir"
    
    # Backup volumes
    docker run --rm -v sam_data:/data -v sam_memory:/memory -v sam_security:/security \
        -v "$(pwd)/backups/$backup_name":/backup alpine \
        sh -c "cp -r /data /backup/ && cp -r /memory /backup/ && cp -r /security /backup/"
    
    # Create backup info
    cat > "$backup_dir/backup_info.json" << EOF
{
    "backup_name": "$backup_name",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "sam_version": "$(git describe --tags --always 2>/dev/null || echo 'unknown')",
    "docker_image": "$(docker images --format '{{.Repository}}:{{.Tag}}' | grep sam | head -1)"
}
EOF
    
    print_success "Backup created: $backup_dir"
}

# Restore SAM data
restore() {
    local backup_path="$1"
    
    if [ -z "$backup_path" ]; then
        print_error "Please specify backup path: ./manage_sam.sh restore <backup_path>"
        exit 1
    fi
    
    if [ ! -d "$backup_path" ]; then
        print_error "Backup directory not found: $backup_path"
        exit 1
    fi
    
    print_warning "This will overwrite current SAM data. Are you sure? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_info "Restore cancelled"
        exit 0
    fi
    
    print_info "Stopping SAM services..."
    stop
    
    print_info "Restoring data from: $backup_path"
    docker run --rm -v sam_data:/data -v sam_memory:/memory -v sam_security:/security \
        -v "$(realpath $backup_path)":/backup alpine \
        sh -c "rm -rf /data/* /memory/* /security/* && cp -r /backup/data/* /data/ && cp -r /backup/memory/* /memory/ && cp -r /backup/security/* /security/"
    
    print_info "Starting SAM services..."
    start
    
    print_success "Restore completed"
}

# Update SAM
update() {
    print_info "Updating SAM..."
    
    # Pull latest code (if in git repo)
    if [ -d ".git" ]; then
        git pull origin main
    fi
    
    # Rebuild and restart
    build
    restart
    
    print_success "SAM updated successfully"
}

# Clean up Docker resources
cleanup() {
    print_warning "This will remove all SAM containers, images, and unused Docker resources. Continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_info "Cleanup cancelled"
        exit 0
    fi
    
    print_info "Stopping and removing SAM containers..."
    docker-compose -p $PROJECT_NAME down -v --remove-orphans
    
    print_info "Removing SAM images..."
    docker images | grep sam | awk '{print $3}' | xargs -r docker rmi -f
    
    print_info "Cleaning up unused Docker resources..."
    docker system prune -f
    
    print_success "Cleanup completed"
}

# Show help
show_help() {
    print_header
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build          Build SAM Docker image"
    echo "  start          Start SAM services"
    echo "  stop           Stop SAM services"
    echo "  restart        Restart SAM services"
    echo "  status         Show service status"
    echo "  logs [service] Show logs (optionally for specific service)"
    echo "  urls           Show access URLs"
    echo "  backup         Create data backup"
    echo "  restore <path> Restore data from backup"
    echo "  update         Update SAM (pull code, rebuild, restart)"
    echo "  cleanup        Remove all SAM containers and images"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start SAM"
    echo "  $0 logs sam-app            # Show logs for main app"
    echo "  $0 restore ./backups/sam_backup_20240101_120000"
    echo ""
}

# Main script logic
main() {
    case "${1:-help}" in
        build)
            check_dependencies
            build
            ;;
        start)
            check_dependencies
            start
            ;;
        stop)
            stop
            ;;
        restart)
            restart
            ;;
        status)
            status
            ;;
        logs)
            logs "$2"
            ;;
        urls)
            show_urls
            ;;
        backup)
            backup
            ;;
        restore)
            restore "$2"
            ;;
        update)
            check_dependencies
            update
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
