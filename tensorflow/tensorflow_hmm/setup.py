from setuptools import setup


setup(
    name='tensorflow_hmm',
    version='0.2.5',
    description='Tensorflow and numpy implementations of the HMM viterbi and '
                'forward/backward algorithms',
    url='http://github.com/dwiel/tensorflow_hmm',
    author='Zach Dwiel',
    author_email='zdwiel@gmail.com',
    license='Apache',
    packages=['tensorflow_hmm'],
    install_requires=[
        'numpy',
        'pytest',
        'tensorflow',
    ],
)
