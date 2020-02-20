import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
  name="chronoclust", # Replace with your own username
  version="0.2.3",
  author="Givanna Putri",
  author_email="ghar1821@uni.sydney.edu.au",
  description="A clustering algorithm that will perform clustering on each of a time-series of discrete datasets, and explicitly track the evolution of clusters over time.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://https://github.com/ghar1821/Chronoclust",
  packages=setuptools.find_packages(),
  classifiers=[
    'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',   # Again, pick a license
    'Programming Language :: Python :: 3.7'
  ],
  python_requires='>=3.6',
)
