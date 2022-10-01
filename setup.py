import os.path
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__),fname)).read()

setup(name='mosa',
      version='0.3.5',
      description="Multi-objective Simulated Annealing (MOSA) implementation in pure Python.",
      long_description_content_type='text/markdown',
      long_description=read('Readme.MD'),
      author='Roberto Gomes de Aguiar Veiga',
      url="https://github.com/rgaveiga/mosa",
      packages=['mosa'])


