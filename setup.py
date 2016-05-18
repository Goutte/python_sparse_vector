from distutils.core import setup
import os

# python setup.py sdist upload

version = '0.4'
github_url = 'https://github.com/Goutte/python_sparse_vector'
paj = 'Goutte'
paj_email = 'antoine@goutenoir.com'

# Pypi does not support markdown (the cheeseshop strikes again)
long_description = ''
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
    long_description = long_description.replace("\r", "")
except (ImportError, OSError):
    print("Pandoc not found. Long description conversion failure.")
    import io
    with io.open('README.md', encoding="utf-8") as f:
        long_description = f.read()


setup(
    name='sparse_vector',
    py_modules=['sparse_vector'],
    version=version,
    description='A sparse vector in pure python, based on numpy.',
    author=paj,
    author_email=paj_email,
    maintainer=paj,
    maintainer_email=paj_email,
    url=github_url,
    download_url='{}/tarball/{}'.format(github_url, version),
    keywords=['sparse', 'vector', 'list', 'container', 'iterable'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    long_description=long_description,
    # install_requires('future', 'six'),
    license='MIT'
)
