from setuptools import setup, find_packages


with open("requirements.txt", "r") as file:
    requirements = file.readlines()

setup(
    name="NaRLA",
    version="0.1dev",
    license="None",
    packages=find_packages(),
    install_requires=requirements,
    description="Neurons as Reinforcement Learning Agents",
    url="https://github.com/Multi-Agent-Networks/NaRLA",
    author="Jordan Ott",
    author_email="jordanott365@gmail.com",
    long_description=open("README.md").read(),
)
