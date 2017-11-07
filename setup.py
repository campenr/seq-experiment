"""setup.py for mothur_py."""

from setuptools import setup, find_packages

# # Get the long description from the README.md, converted to .rst if possible
# try:
#     import pypandoc
#     long_description = pypandoc.convert('README.md', 'rst')
#     print('using README.rst for long description')
# except(IOError, ImportError):
#     long_description = open('README.md').read()
#     print('using README.md for long description')

setup(
    name="seq-experiment",
    version="0.1.0",
    author="Richard Campen",
    author_email="richard@campen.co",
    license="Modified BSD License",

    # classifiers=[
    # ],
    #
    # keywords="",
    packages=find_packages(),
    include_package_data=True
)
