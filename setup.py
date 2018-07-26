import versioneer
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='geometric',
    description='Geometry optimization for quantum chemistry',
    url='https://github.com/leeping/geomeTRIC',
    author='Lee-Ping Wang, Chenchen Song',
    packages=['geometric'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={'console_scripts': [
        'geometric-optimize = geometric.optimize:main',
    ]},
    install_requires=[
        'numpy>=1.11',
        'networkx',
        'six',
    ],
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass())
