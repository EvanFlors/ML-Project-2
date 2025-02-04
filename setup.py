""" 
  The setup.py file is essential part of packaging and
  distributing Python projects. It is used by setuptools
  (or distutils in older Python versions) to define the configuration
  of your project, such as its metadata and dependencies.
"""

from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
  """
    This function returns a list of requirements
  """
  requirement_lst: List[str] = []
  try:
    with open('requirements.txt') as f:
      lines = f.readlines()
      for line in lines:
        requirement = line.strip()
        
        if requirement and requirement != '-e .':
          requirement_lst.append(requirement)
        
  except FileNotFoundError:
    print("requirements.txt file not found")
    
  return requirement_lst

setup(
    name='NetworkSecurity',
    version='0.0.1',
    description='Machine Learning Project for Network Security',
    author='Evan Flores',
    packages=find_packages(),
    install_requires=get_requirements()
)