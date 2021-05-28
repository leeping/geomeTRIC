from setuptools import find_packages, setup

import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='geometric',
    description='Geometry optimization for quantum chemistry',
    url='https://github.com/leeping/geomeTRIC',
    author='Lee-Ping Wang, Chenchen Song',
    packages=find_packages(),
    package_data={'': ['*.ini']},
    include_package_data=True,
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
    tests_require=[
        'pytest',
        'pytest-cov',
    ],
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    zip_safe=True,
)
