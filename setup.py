"""Package installation setup."""

from setuptools import setup, find_packages

setup(
    name="uniflowmatch",
    version="0.1.0",
    description="UniFlowMatch Project",
    author="AirLab",
    license="BSD Clause-3",
    packages=find_packages(exclude=['test']),  # Directly specify the package
    include_package_data=True,
)
