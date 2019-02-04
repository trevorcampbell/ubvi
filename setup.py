from distutils.core import setup

setup(
    name='FisherGP',
    version='1.0',
    packages=['fishergp',],
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires = ['numpy', 'scipy', 'GPy', 'autograd'],
)
