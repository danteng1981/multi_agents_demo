# Quick Start Guide

Welcome to the Multi-Agents Demo! This guide will help you get up and running quickly.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Startup](#startup)
- [Verification](#verification)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
- **pip**: Python package installer (usually comes with Python)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **Virtual Environment** (recommended): `venv` or `virtualenv`

### Step 1: Clone the Repository

```bash
git clone https://github.com/danteng1981/multi_agents_demo.git
cd multi_agents_demo
```

### Step 2: Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# If you're developing, install dev dependencies
pip install -r requirements-dev.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory of the project:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Agent Configuration
MAX_AGENTS=5
AGENT_TIMEOUT=30
LOG_LEVEL=INFO

# Database Configuration (if applicable)
DATABASE_URL=sqlite:///./multi_agents.db

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False
```

### Configuration File

Alternatively, you can use the `config.yaml` file for more detailed configuration:

```yaml
agents:
  max_concurrent: 5
  timeout: 30
  retry_attempts: 3

models:
  default: "gpt-4"
  fallback: "gpt-3.5-turbo"
  
logging:
  level: "INFO"
  format: "json"
  file: "logs/multi_agents.log"
```

## Startup

### Basic Startup

Start the multi-agent system with default settings:

```bash
python main.py
```

### Advanced Startup Options

Start with custom configuration:

```bash
# Specify a custom config file
python main.py --config config/custom_config.yaml

# Enable debug mode
python main.py --debug

# Specify number of agents
python main.py --agents 3

# Run in verbose mode
python main.py --verbose
```

### Running as a Service

For production environments, you can run the application as a service:

```bash
# Using systemd (Linux)
sudo systemctl start multi-agents-demo

# Using Docker
docker-compose up -d
```

### Docker Setup

If you prefer Docker:

```bash
# Build the Docker image
docker build -t multi-agents-demo .

# Run the container
docker run -d \
  --name multi-agents-demo \
  -p 8000:8000 \
  --env-file .env \
  multi-agents-demo
```

## Verification

### Check Installation

Verify that the installation was successful:

```bash
# Check Python version
python --version

# Verify dependencies
pip list | grep -i "agent\|openai\|anthropic"

# Run tests
pytest tests/
```

### Health Check

Once the application is running, verify it's working correctly:

```bash
# Check if the service is running
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "agents": 5, "version": "1.0.0"}
```

### Verify Agents

Check that agents are initialized properly:

```bash
# List all agents
curl http://localhost:8000/api/agents

# Check agent status
python -c "from multi_agents import AgentManager; manager = AgentManager(); print(manager.status())"
```

### Log Verification

Check the logs to ensure everything is running smoothly:

```bash
# View logs
tail -f logs/multi_agents.log

# Check for errors
grep ERROR logs/multi_agents.log
```

## Usage Examples

### Example 1: Basic Agent Interaction

```python
from multi_agents import AgentManager, Task

# Initialize the agent manager
manager = AgentManager()

# Create a simple task
task = Task(
    name="data_analysis",
    description="Analyze the sales data from Q4 2024",
    priority="high"
)

# Execute the task
result = manager.execute_task(task)
print(f"Result: {result}")
```

### Example 2: Multi-Agent Collaboration

```python
from multi_agents import AgentManager, CollaborativeTask

# Initialize manager with multiple agents
manager = AgentManager(num_agents=3)

# Create a collaborative task
task = CollaborativeTask(
    name="research_project",
    description="Research market trends and create a report",
    subtasks=[
        {"name": "data_collection", "agent_type": "researcher"},
        {"name": "data_analysis", "agent_type": "analyst"},
        {"name": "report_generation", "agent_type": "writer"}
    ]
)

# Execute collaboratively
result = manager.execute_collaborative(task)
print(f"Final Report: {result.output}")
```

### Example 3: Using the API

```python
import requests

# Submit a task via API
response = requests.post(
    "http://localhost:8000/api/tasks",
    json={
        "name": "sentiment_analysis",
        "description": "Analyze customer feedback sentiment",
        "data": {
            "feedback": ["Great product!", "Needs improvement", "Excellent service!"]
        }
    }
)

task_id = response.json()["task_id"]
print(f"Task submitted: {task_id}")

# Check task status
status = requests.get(f"http://localhost:8000/api/tasks/{task_id}")
print(f"Status: {status.json()}")
```

### Example 4: Custom Agent Creation

```python
from multi_agents import BaseAgent, AgentCapability

class CustomAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="CustomAnalyzer",
            capabilities=[
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.VISUALIZATION
            ]
        )
    
    def process(self, data):
        # Custom processing logic
        analysis = self.analyze_data(data)
        return {
            "summary": analysis,
            "insights": self.generate_insights(analysis)
        }

# Register and use the custom agent
manager = AgentManager()
manager.register_agent(CustomAnalysisAgent())
```

### Example 5: Batch Processing

```bash
# Process multiple tasks from a file
python cli.py batch --input tasks.json --output results.json

# Monitor batch progress
python cli.py batch-status --job-id abc123
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'multi_agents'`

**Solution**:
```bash
# Ensure you're in the correct directory
cd multi_agents_demo

# Reinstall dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

#### Issue 2: API Key Errors

**Problem**: `AuthenticationError: Invalid API key`

**Solution**:
```bash
# Verify your .env file exists
ls -la .env

# Check API key is set
echo $OPENAI_API_KEY

# Reload environment variables
source .env  # or restart your terminal
```

#### Issue 3: Port Already in Use

**Problem**: `OSError: [Errno 48] Address already in use`

**Solution**:
```bash
# Find and kill the process using the port
lsof -i :8000
kill -9 <PID>

# Or use a different port
python main.py --port 8001
```

#### Issue 4: Agent Timeout

**Problem**: Agents taking too long to respond

**Solution**:
```python
# Increase timeout in config
manager = AgentManager(timeout=60)  # 60 seconds

# Or in .env file
AGENT_TIMEOUT=60
```

#### Issue 5: Memory Issues

**Problem**: High memory usage with multiple agents

**Solution**:
```bash
# Reduce number of concurrent agents
python main.py --agents 2

# Enable memory profiling
pip install memory_profiler
python -m memory_profiler main.py
```

### Debugging Tips

#### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via command line
python main.py --log-level DEBUG
```

#### Check System Resources

```bash
# Monitor CPU and memory usage
top -p $(pgrep -f multi_agents)

# Check disk space
df -h

# View resource usage in Python
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"
```

#### Database Issues

```bash
# Reset database
python manage.py reset-db

# Run migrations
python manage.py migrate

# Check database connection
python -c "from multi_agents.db import check_connection; check_connection()"
```

### Getting Help

If you're still experiencing issues:

1. **Check the logs**: Look at `logs/multi_agents.log` for detailed error messages
2. **Review documentation**: See the full [documentation](../README.md)
3. **Search issues**: Check [GitHub Issues](https://github.com/danteng1981/multi_agents_demo/issues)
4. **Ask for help**: Open a new issue with:
   - Your Python version
   - Operating system
   - Complete error message
   - Steps to reproduce

### Performance Optimization

If you're experiencing slow performance:

```yaml
# Optimize config.yaml
performance:
  cache_enabled: true
  max_workers: 4
  batch_size: 10
  connection_pool_size: 20
```

```bash
# Enable caching
export ENABLE_CACHE=true

# Use faster models for testing
export DEFAULT_MODEL=gpt-3.5-turbo
```

### Clean Restart

For a completely fresh start:

```bash
# Remove virtual environment
deactivate
rm -rf venv

# Clear cache and logs
rm -rf __pycache__ logs/*.log .cache

# Reinstall everything
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Next Steps

Now that you're up and running:

1. Explore the [API Documentation](API.md)
2. Review [Architecture Overview](ARCHITECTURE.md)
3. Check out [Advanced Examples](EXAMPLES.md)
4. Read the [Contributing Guide](../CONTRIBUTING.md)

## Feedback

Found an issue with this guide? Please [open an issue](https://github.com/danteng1981/multi_agents_demo/issues/new) or submit a pull request!

---

**Last Updated**: 2025-12-19  
**Version**: 1.0.0
