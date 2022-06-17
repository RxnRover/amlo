from setuptools import setup

setup(
    # version=open("VERSION.txt", 'r').read()
    install_requires=open("requirements.txt", 'r').read().split('\n')
)
