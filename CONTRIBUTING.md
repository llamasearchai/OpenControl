# Contributing to OpenControl

Thank you for your interest in contributing to OpenControl! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. We expect all contributors to:

- Be respectful and considerate in all interactions
- Focus on constructive feedback and collaboration
- Respect different viewpoints and experiences
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM (recommended)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/opencontrol.git
   cd opencontrol
   ```

2. **Set up Development Environment**
   ```bash
   # Using uv (recommended)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   
   # Or using pip
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Verify Installation**
   ```bash
   python test_system.py
   ```

## Contributing Process

### 1. Choose an Issue

- Look for issues labeled `good first issue` for beginners
- Check existing issues and discussions before creating new ones
- For major changes, open an issue first to discuss the approach

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Make Changes

- Follow the code standards outlined below
- Write tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new multimodal encoder architecture"
```

Use conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding tests
- `refactor:` for code refactoring
- `perf:` for performance improvements

## Code Standards

### Python Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:
```bash
# Format code
black opencontrol/ tests/
isort opencontrol/ tests/

# Check linting
flake8 opencontrol/ tests/

# Type checking
mypy opencontrol/
```

### Code Quality Guidelines

1. **Type Hints**: Use type hints for all function signatures
   ```python
   def process_data(data: torch.Tensor, config: Config) -> Dict[str, Any]:
       ...
   ```

2. **Docstrings**: Use Google-style docstrings
   ```python
   def compute_attention(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
       """Compute multi-head attention.
       
       Args:
           query: Query tensor of shape (batch, seq_len, d_model)
           key: Key tensor of shape (batch, seq_len, d_model)
           
       Returns:
           Attention output tensor of shape (batch, seq_len, d_model)
       """
   ```

3. **Error Handling**: Use appropriate exception handling
   ```python
   try:
       result = risky_operation()
   except SpecificError as e:
       logger.error(f"Operation failed: {e}")
       raise
   ```

4. **Logging**: Use structured logging
   ```python
   logger.info("Starting training", extra={
       "epoch": epoch,
       "batch_size": batch_size,
       "learning_rate": lr
   })
   ```

### Architecture Guidelines

1. **Modularity**: Keep components loosely coupled
2. **Configuration**: Use dataclasses for configuration
3. **Testing**: Write unit tests for all new functionality
4. **Performance**: Consider memory and computational efficiency
5. **Documentation**: Document complex algorithms and design decisions

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_world_model.py

# Run with coverage
pytest --cov=opencontrol

# Run performance benchmarks
pytest --benchmark-only
```

### Writing Tests

1. **Unit Tests**: Test individual components
   ```python
   def test_attention_mechanism():
       attention = MultiModalAttention(512, 8)
       x = torch.randn(2, 10, 512)
       output, weights = attention(x)
       assert output.shape == x.shape
   ```

2. **Integration Tests**: Test component interactions
   ```python
   @pytest.mark.asyncio
   async def test_training_pipeline():
       trainer = create_trainer()
       await trainer.train_one_epoch()
       assert trainer.metrics['loss'] < initial_loss
   ```

3. **Performance Tests**: Test computational requirements
   ```python
   @pytest.mark.benchmark
   def test_inference_speed(benchmark):
       result = benchmark(model.forward, input_data)
       assert result is not None
   ```

## Documentation

### Code Documentation

- Use clear, descriptive variable and function names
- Add docstrings to all public functions and classes
- Include type hints for better IDE support
- Document complex algorithms with inline comments

### API Documentation

- Update docstrings when changing function signatures
- Include examples in docstrings for complex functions
- Document configuration options and their effects

### User Documentation

- Update README.md for user-facing changes
- Add examples for new features
- Update installation instructions if dependencies change

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**
   ```bash
   pytest
   ```

2. **Check code quality**
   ```bash
   black --check opencontrol/ tests/
   flake8 opencontrol/ tests/
   mypy opencontrol/
   ```

3. **Update documentation**
   - Add docstrings for new functions
   - Update README if needed
   - Add examples for new features

### Pull Request Template

When creating a pull request, include:

- **Description**: Clear description of changes
- **Motivation**: Why this change is needed
- **Testing**: How the changes were tested
- **Breaking Changes**: Any backward compatibility issues
- **Related Issues**: Link to related issues

### Review Process

1. **Automated Checks**: CI will run tests and quality checks
2. **Code Review**: Maintainers will review the code
3. **Feedback**: Address any feedback or requested changes
4. **Approval**: Once approved, the PR will be merged

### Merge Requirements

- All CI checks must pass
- At least one maintainer approval
- No unresolved conversations
- Up-to-date with main branch

## Development Tips

### Performance Optimization

- Profile code before optimizing
- Use appropriate data types (float16 vs float32)
- Consider memory usage in large-scale training
- Leverage GPU acceleration where possible

### Debugging

- Use descriptive error messages
- Add logging to track execution flow
- Use debugger breakpoints for complex issues
- Test with small datasets first

### Best Practices

- Write self-documenting code
- Keep functions small and focused
- Use meaningful variable names
- Avoid premature optimization
- Test edge cases and error conditions

## Getting Help

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact nikjois@llamasearch.ai for urgent matters

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Project documentation

Thank you for contributing to OpenControl! 