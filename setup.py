from setuptools import setup, find_packages

setup(
    name='dense',
    version='0.0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    url='https://github.com/luyug/Dense',
    license='Apache 2.0',
    author='Luyu Gao',
    author_email='luyug@cs.cmu.edu',
    description='A toolkit for learning and running deep dense retrieval models.'
)
