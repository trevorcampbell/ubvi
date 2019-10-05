from setuptools import setup, find_packages

setup(
    name='ubvi',
    url='https://github.com/trevorcampbell/ubvi',
    author='Trevor Campbell',
    author_email='trevor@stat.ubc.ca',
    version='0.2',
    packages=find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires = ['numpy', 'scipy', 'autograd'],
)
