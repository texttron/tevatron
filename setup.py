from setuptools import setup, find_packages

setup(
    name='tevatron',
    version='0.0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    url='https://github.com/texttron/tevatron',
    license='Apache 2.0',
    author='Luyu Gao',
    author_email='luyug@cs.cmu.edu',
    description='Tevatron: A toolkit for learning and running deep dense retrieval models.',
    python_requires='>=3.7',
    install_requires=[
        "transformers>=4.3.0,<=4.9.2",
        "datasets>=1.1.3"
    ]
)
