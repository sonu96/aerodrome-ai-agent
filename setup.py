"""
Setup script for Aerodrome AI Agent

Development setup script for easy installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="aerodrome-ai-agent",
    version="1.0.0",
    description="Autonomous AI-powered DeFi portfolio manager for Aerodrome Finance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Aerodrome AI Agent Team",
    author_email="admin@aerodrome-agent.ai",
    url="https://github.com/aerodrome-ai/agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-mock>=3.12.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
            "pre-commit>=3.6.0",
            "coverage>=7.3.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
            "mkdocstrings[python]>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aerodrome-agent=aerodrome_ai_agent.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["defi", "aerodrome", "ai", "trading", "base", "blockchain"],
    project_urls={
        "Documentation": "https://docs.aerodrome-agent.ai",
        "Source": "https://github.com/aerodrome-ai/agent",
        "Tracker": "https://github.com/aerodrome-ai/agent/issues",
    },
)