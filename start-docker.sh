#!/bin/bash

# Start script for Accent Classifier with Docker Compose

echo "🚀 Starting Accent Classifier with Docker Compose..."
echo "   Access URL: http://localhost:3104"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker compose is available
if ! docker compose version > /dev/null 2>&1; then
    if ! command -v docker-compose > /dev/null 2>&1; then
        echo "❌ docker compose is not available. Please install Docker Compose and try again."
        exit 1
    fi
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

# Create required directories
mkdir -p uploads temp

# Build and start the services
echo "🔧 Building and starting services..."
$COMPOSE_CMD up --build -d

# Wait a moment for the service to start
echo "⏳ Waiting for service to start..."
sleep 5

# Check if the service is healthy
echo "🔍 Checking service health..."
for i in {1..10}; do
    if curl -s http://localhost:3104/health > /dev/null; then
        echo "✅ Service is running successfully!"
        echo "🌐 Open your browser to: http://localhost:3104"
        echo ""
        echo "📋 Useful commands:"
        echo "   View logs:    $COMPOSE_CMD logs -f"
        echo "   Stop service: $COMPOSE_CMD down"
        echo "   Restart:      $COMPOSE_CMD restart"
        exit 0
    fi
    echo "   Attempt $i/10: Service not ready yet..."
    sleep 3
done

echo "⚠️  Service might not be fully ready yet. Check logs with:"
echo "   $COMPOSE_CMD logs -f"
echo ""
echo "🌐 Try accessing: http://localhost:3104" 