import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chronoclust", # Replace with your own username
    version="0.2.4",
    author="Givanna Putri",
    author_email="ghar1821@uni.sydney.edu.au",
    description="A clustering algorithm that will perform clustering on each of a time-series of discrete datasets, and explicitly track the evolution of clusters over time.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/ghar1821/Chronoclust",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7'
    ],
    python_requires='>=3.6',
)
