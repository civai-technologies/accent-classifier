# Docker Setup for Accent Classifier

This document describes how to run the Accent Classifier using Docker Compose with ingress port 3104.

## Quick Start

### Prerequisites
- Docker (version 20.0+)
- Docker Compose (version 1.28+)

### Start the Application

1. **Quick Start Script** (Recommended):
   ```bash
   ./start-docker.sh
   ```

2. **Manual Start**:
   ```bash
   docker-compose up --build -d
   ```

### Access the Application

- **Web Interface**: http://localhost:3104
- **Health Check**: http://localhost:3104/health

## Docker Compose Configuration

The `docker-compose.yml` file provides:

- **Port Mapping**: External port 3104 → Internal port 5000
- **Auto-restart**: Service restarts automatically if it crashes
- **Health Checks**: Built-in health monitoring
- **Volume Mapping**: Persistent storage for uploads and temporary files
- **Environment**: Production-ready configuration

## Available Commands

### Basic Operations
```bash
# Start services
docker-compose up -d

# Start with rebuild
docker-compose up --build -d

# Stop services
docker-compose down

# Restart services
docker-compose restart

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f accent-classifier
```

### Development Commands
```bash
# Rebuild only
docker-compose build

# Remove containers and volumes
docker-compose down -v

# Shell access to running container
docker-compose exec accent-classifier bash
```

## Directory Structure

```
accent-classifier/
├── docker-compose.yml      # Main compose configuration
├── Dockerfile             # Container build instructions
├── start-docker.sh        # Convenient start script
├── frontend/              # Flask web application
├── src/                   # Core processing modules
├── models/                # ML models
├── test_samples/          # Test audio/video files
├── uploads/               # User uploaded files (volume)
└── temp/                  # Temporary processing files (volume)
```

## Environment Variables

The following environment variables are configured in docker-compose.yml:

| Variable | Value | Description |
|----------|-------|-------------|
| PORT | 5000 | Internal Flask port |
| HOST | 0.0.0.0 | Flask host binding |
| FLASK_ENV | production | Flask environment |
| FLASK_DEBUG | False | Debug mode (disabled in production) |

## Volumes

Two volumes are configured for data persistence:

- `./uploads` → `/app/uploads` - User uploaded audio files
- `./temp` → `/app/temp` - Temporary processing files

## Health Checks

The container includes built-in health monitoring:

- **Endpoint**: `/health`
- **Interval**: Every 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 3 attempts
- **Start Period**: 40 seconds

## Troubleshooting

### Service Won't Start
```bash
# Check logs
docker-compose logs accent-classifier

# Rebuild and start
docker-compose down
docker-compose up --build -d
```

### Port Already in Use
```bash
# Find process using port 3104
lsof -i :3104

# Kill the process (replace PID)
kill -9 <PID>
```

### Permission Issues
```bash
# Fix directory permissions
mkdir -p uploads temp
chmod 755 uploads temp
```

### Clean Restart
```bash
# Complete cleanup and restart
docker-compose down -v
docker system prune -f
docker-compose up --build -d
```

## Performance Optimization

### Container Resources
The container is optimized for:
- Memory usage: ~512MB-1GB
- CPU usage: 1-2 cores
- Storage: 2-5GB for models and samples

### Build Optimization
- Multi-stage builds for smaller images
- Layer caching for faster rebuilds
- `.dockerignore` to exclude unnecessary files

## Security Considerations

- Container runs as non-root user
- No unnecessary ports exposed
- Environment variables for sensitive configuration
- Volume mounting for data persistence only

## Production Deployment

For production deployment, consider:

1. **Reverse Proxy**: Use nginx or traefik
2. **SSL/TLS**: Configure HTTPS certificates
3. **Monitoring**: Add logging and metrics collection
4. **Scaling**: Use docker swarm or kubernetes
5. **Backup**: Regular backup of models and data volumes 