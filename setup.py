"""Package installation setup."""

from setuptools import setup

setup(
    name="uniflowmatch",
    version="0.0.0",
    description="UniFlowMatch Project",
    author="AirLab",
    license="BSD Clause-3",
    packages=["uniflowmatch"],  # Directly specify the package
    package_dir={
        "uniflowmatch": "uniflowmatch",  # Map uniflowmatch package
    },
    include_package_data=True,
)
