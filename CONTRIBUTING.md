# Contributing to EMOTIA

Thank you for your interest in contributing to the EMOTIA project. We welcome contributions from the community and are grateful for your help in making this project better.

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:
- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Help create a positive community

## How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs
- Provide detailed steps to reproduce the issue
- Include relevant system information and error messages
- Check if the issue has already been reported

### Suggesting Features
- Use the GitHub issue tracker for feature requests
- Clearly describe the proposed feature and its benefits
- Consider if the feature aligns with the project's goals
- Be open to discussion and feedback

### Contributing Code

1. **Fork the Repository**
   - Create a fork of the repository on GitHub
   - Clone your fork locally

2. **Set Up Development Environment**
   ```bash
   git clone https://github.com/your-username/emotia.git
   cd emotia
   pip install -r requirements.txt
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**
   - Write clear, concise commit messages
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

5. **Run Tests**
   ```bash
   pytest backend/tests/ -v
   ```

6. **Submit a Pull Request**
   - Push your changes to your fork
   - Create a pull request with a clear description
   - Reference any related issues

## Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use Black for code formatting
- Use Flake8 for linting
- Use MyPy for type checking

### Testing
- Write unit tests for new functionality
- Maintain 90%+ test coverage
- Run the full test suite before submitting
- Test both positive and negative scenarios

### Documentation
- Update docstrings for new functions
- Add comments for complex logic
- Update README.md for significant changes
- Document API changes

### Security
- Run security scans before submitting
- Avoid committing sensitive information
- Use secure coding practices
- Report security issues through proper channels

## Commit Guidelines

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in imperative mood
- Keep the first line under 50 characters
- Provide additional context in the body if needed

### Examples
```
Fix memory leak in video processing
Add support for WebRTC streaming
Update documentation for API endpoints
```

## Pull Request Process

### Before Submitting
- Ensure all tests pass
- Update documentation
- Add appropriate labels
- Request review from maintainers

### During Review
- Address reviewer feedback promptly
- Make requested changes
- Keep the conversation constructive
- Be open to suggestions

### After Approval
- Maintainers will merge the pull request
- Your contribution will be acknowledged
- You may be asked to help with future related changes

## Areas for Contribution

### High Priority
- Bug fixes and security patches
- Performance improvements
- Documentation improvements
- Test coverage expansion

### Medium Priority
- New features (with prior discussion)
- Code refactoring
- Tooling improvements
- Example applications

### Low Priority
- Minor UI improvements
- Additional language support
- Community tools and integrations

## Recognition

Contributors will be:
- Listed in the project contributors file
- Acknowledged in release notes
- Recognized for significant contributions
- Invited to join the core team for major contributions

## Getting Help

If you need help:
- Check the documentation first
- Search existing issues and discussions
- Ask questions in GitHub discussions
- Contact the maintainers directly

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).