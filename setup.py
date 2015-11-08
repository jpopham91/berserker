from distutils.core import setup

setup(
    name='berserker',
    version='0.1',
    description='Model ensembling framework',
    packages=['berserker', 'berserker.estimators'],
    url='github.com/jpopham91/berserker',
    license='MIT',
    author='Jake Popham',
    author_email='jxp6414@gmail.com',
    install_requires=['sklearn', 'numpy', 'pandas'],
)
