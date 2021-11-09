from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy"]

setup(
    name="PyQAC",
    version="0.0.1",
    author="MajesticWizard",
    author_email="lizardwizardetofake@gmail.com",
    description="A package for quantum computing",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/MajesticWizard/PyQAC",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)