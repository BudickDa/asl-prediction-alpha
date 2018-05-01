from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name='ec_forecast',
    version='0.1',
    author = 'Daniel Budick',
    author_email = 'daniel@budick.eu',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='RNN forecaster/LSTM forecaster',
    requires=[]
)