import sys
import setuptools

setuptools.setup(
    name="archer",
    version='0.1.0',
    url="https://github.com/andreazanette/OfflineArcher.git",
    author=("Andrea Zanette"),
    description="Research code for Offline ArCHer (Actor Critic Framework with Hierarchical Structures)",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='ArCHer',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=open("requirements.txt", "r").read().split(),
    include_package_data=True,
    python_requires='>=3.7',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)