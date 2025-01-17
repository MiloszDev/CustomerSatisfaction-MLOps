from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="CustomerStisfactionPrediction",
    version="0.1.0",
    author="Miłosz Zawolik",
    author_email="milosz.zawolik@gmail.com",
    description="A short description of the project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/project-name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0",
        "tensorflow>=2.6.0",
        "mlflow>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "train_model=scripts.run_training:main",
            "evaluate_model=scripts.run_evaluation:main",
        ],
    },
    project_urls={
        "Documentation": "https://github.com/yourusername/project-name/wiki",
        "Bug Tracker": "https://github.com/yourusername/project-name/issues",
    },
)
