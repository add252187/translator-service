# Makefile for translator-service

.PHONY: help install dev test lint format clean docker-build docker-up docker-down run migrate

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=app --cov-report=term-missing

lint: ## Run linting
	flake8 app/ tests/
	mypy app/ --ignore-missing-imports
	black --check app/ tests/

format: ## Format code
	black app/ tests/
	isort app/ tests/

clean: ## Clean temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.db" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

docker-build: ## Build Docker image
	docker-compose build

docker-up: ## Start services with Docker Compose
	docker-compose up -d

docker-down: ## Stop Docker Compose services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f translator-service

run: ## Run the application locally
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

migrate: ## Run database migrations
	alembic upgrade head

db-reset: ## Reset database
	alembic downgrade base
	alembic upgrade head

check-env: ## Check if required environment variables are set
	@echo "Checking environment variables..."
	@test -n "$$TWILIO_ACCOUNT_SID" || (echo "TWILIO_ACCOUNT_SID not set" && exit 1)
	@test -n "$$TWILIO_AUTH_TOKEN" || (echo "TWILIO_AUTH_TOKEN not set" && exit 1)
	@echo "Environment variables OK"

setup: install ## Complete setup for development
	cp .env.example .env
	@echo "Setup complete! Edit .env file with your API keys"

prod-deploy: ## Deploy to production
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

monitor: ## Show system metrics
	@echo "Active calls:"
	@curl -s http://localhost:8000/calls | python -m json.tool
	@echo "\nSystem metrics:"
	@curl -s http://localhost:8000/metrics | python -m json.tool

health: ## Check service health
	@curl -s http://localhost:8000/health | python -m json.tool
