from distutils.core import setup

setup(
    name='Universal Boosting Variational Inference',
    version='0.1',
    packages=['ubvi',],
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires = ['numpy', 'scipy', 'autograd'],
)
