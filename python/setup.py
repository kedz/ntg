from setuptools import setup

packages = [
    'ntp', 
    'ntp.dataio', 
    'ntp.dataio.field_reader', 
    'ntp.dataio.file_reader', 
    'ntp.modules',
    'ntp.models',
    'ntp.criterion',
    'ntp.optimizer',
    'ntp.datasets',
    'ntp.trainer',
]

setup(name='ntp',
      version='0.1',
      description='Neural text processing library using pytorch.',
      author='Chris Kedzie',
      author_email='kedzie@cs.columbia.edu',
      packages=packages,
      install_requires=[
          "numpy",
          "scipy",
          "sklearn",
          "pandas",
          "nltk"]  
)
