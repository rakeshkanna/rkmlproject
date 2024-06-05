from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(fileName:str)->List[str]:
    '''
    Function to return list of requirements
    '''
    requirements = []
    with open(fileName) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements



setup(
    name="mlProject",
    version="0.0.1",
    author="Rakesh Khanna",
    author_email="rakesh.khanna@live.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)