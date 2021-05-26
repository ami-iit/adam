from setuptools import setup
from setuptools import find_packages

config = {
    'name' : 'ADAM',
    'description' : 'Automatic Differentiation for rigid body AlgorithMs',
    'install_requires' : [
        'numpy',
        'scipy',
        'casadi',
        'matplotlib',
        'prettytable',
        'urdf_parser_py'
    ],
    'python_requires' : '>=3.6',
    'packages' : find_packages(),
}

setup(**config)
