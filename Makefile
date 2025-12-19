. PHONY: help dev test clean logs health

help:
	@echo "Enterprise Agent Platform - Commands"
	@echo ""
	@echo "  make dev     - Start development environment"
	@echo "  make test    - Run tests with coverage"
	@echo "  make clean   - Clean environment"
	@echo "  make logs    - View application logs"
	@echo "  make health  - Check service health"

dev:
	@echo "🚀 Starting development environment..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "📚 API Docs: http://localhost:8000/docs"
	@echo "📊 Grafana: http://localhost:3000 (admin/admin)"
	@echo "🔍 Jaeger: http://localhost:16686"

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✅ Tests complete!  Report:  htmlcov/index.html"

clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

logs:
	docker-compose logs -f app

health:
	@echo "🏥 Checking service health..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "❌ Service not responding"
