from distutils.core import setup

setup(
    name='ubvi',
    url='https://github.com/trevorcampbell/ubvi',
    author='Trevor Campbell',
    author_email='trevor@stat.ubc.ca',
    version='0.1',
    packages=['ubvi',],
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires = ['numpy', 'scipy', 'autograd'],
)
