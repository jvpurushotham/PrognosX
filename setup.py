from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="prognosx",
    version="0.1.0",
    description="Industrial Predictive Maintenance & Remaining Useful Life Intelligence",
    author="Purushotham J V",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.10",
)
