

## Notebooks    [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/gopala-kr/notebooks/master)




> Collection of practical handson jupyter notebooks on bigdata/ml/dl/rl/cv/nlp/ds/python-scientific-tools/viz-lib/various command lines. All this stuff is collected from github. I will soon reorganize this to reflect recency.

Play with these notebooks by launching a live server on [binder](https://mybinder.org) or google-colabs(https://research.google.com/colaboratory/unregistered.html) [to speedup training on GPU's)


----------------

* [deep-learning](https://github.com/gopala-kr/notebooks/tree/master/deep-learning)
    * [tensorflow](https://github.com/gopala-kr/notebooks/tree/master/tensorflow)
    * [theano](#theano-tutorials)
    
    * [keras](#keras-tutorials)
    * [caffe](#deep-learning-misc)
    * [DL notebooks](#deep-learning-with-python-notebooks)
* [Deep-Learning-Boot-Camp](https://github.com/gopala-kr/notebooks/tree/master/Deep-Learning-Boot-Camp)
* [DeepLearningForNLPInPytorch](https://github.com/gopala-kr/notebooks/tree/master/DeepLearningForNLPInPytorch)
* [DeepLearningFrameworks](https://github.com/gopala-kr/notebooks/tree/master/DeepLearningFrameworks)
* [DeepNLP-models-Pytorch](https://github.com/gopala-kr/notebooks/tree/master/DeepNLP-models-Pytorch)
* [deep-learning-keras-tensorflow](https://github.com/gopala-kr/notebooks/tree/master/deep-learning-keras-tensorflow)
* [deep-learning-with-python-notebooks](https://github.com/gopala-kr/notebooks/tree/master/deep-learning-with-python-notebooks)
* [scikit-learn](#scikit-learn)
     * [scikit learn videos](#scikit-learn-videos)
     * [scikit-learn-official-examples](https://github.com/gopala-kr/notebooks/tree/master/scikit-learn-official-examples)
* [statistical-inference-scipy](#statistical-inference-scipy)
* [pandas](https://github.com/gopala-kr/notebooks/tree/master/pandas)
* [matplotlib](https://github.com/gopala-kr/notebooks/tree/master/matplotlib)
* [numpy](https://github.com/gopala-kr/notebooks/tree/master/numpy)
* [python-data](https://github.com/gopala-kr/notebooks/tree/master/python-data)
* [kaggle-and-business-analyses](https://github.com/gopala-kr/notebooks/tree/master/kaggle)
* [spark](https://github.com/gopala-kr/notebooks/tree/master/spark)
* [mapreduce-python](https://github.com/gopala-kr/notebooks/tree/master/mapreduce)
* [amazon web services](https://github.com/gopala-kr/notebooks/tree/master/aws)
* [command lines](https://github.com/gopala-kr/notebooks/tree/master/commands)
* [DA and ML projects](https://github.com/gopala-kr/notebooks/tree/master/Data-Analysis-and-Machine-Learning-Projects)
* [PPBMH](https://github.com/gopala-kr/notebooks/tree/master/PPBMH)
* [Python Data Science Handbook](https://github.com/gopala-kr/notebooks/tree/master/PythonDataScienceHandbook)
* [Handson ML](https://github.com/gopala-kr/notebooks/tree/master/handson-ml)
* [ipython notebooks](https://github.com/gopala-kr/notebooks/tree/master/ipython-notebooks)
* [numerical linear algebra](https://github.com/gopala-kr/notebooks/tree/master/numerical-linear-algebra)
* [Python scientific stack](#scientific-python-lectures)
* [python machine learning book](https://github.com/gopala-kr/notebooks/tree/master/python-machine-learning-book)
* [Most viewed notebooks](http://nb.bianp.net/sort/views/)
* [Interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)
* [Data Visualization for All - gitbook](https://www.datavizforall.org/)
* [Online-Courses](https://github.com/gopala-kr/notebooks/tree/master/Online-Courses)
* python-machine-learning-book [[1st edition](https://github.com/gopala-kr/notebooks/tree/master/python-machine-learning-book)][[2nd edition](https://github.com/gopala-kr/notebooks/tree/master/python-machine-learning-book-2nd-edition)]
* [reinforcement-learning](https://github.com/gopala-kr/notebooks/tree/master/reinforcement-learning)
* [Data-Analysis](https://github.com/gopala-kr/notebooks/tree/master/Data-Analysis)
* [tests](https://github.com/gopala-kr/notebooks/tree/master/tests)
* [introtodeeplearning_labs_MIT](https://github.com/aamini/introtodeeplearning_labs)
* [newbooks](https://github.com/gopala-kr/notebooks/tree/master/0-newbooks)
* [algorithms_in_ipython_notebooks](https://github.com/gopala-kr/notebooks/tree/master/algorithms_in_ipython_notebooks)
* [PyTorch-Tutorial](https://github.com/gopala-kr/notebooks/tree/master/PyTorch-Tutorial)
* [mltrain-nips-2017](https://github.com/gopala-kr/notebooks/tree/master/mltrain-nips-2017)
* [pydata-notebooks](https://github.com/gopala-kr/notebooks/tree/master/pydata-notebooks)
* [python_reference](https://github.com/gopala-kr/notebooks/tree/master/python_reference)

---------------------



## Installation

First, you will need to install [git](https://git-scm.com/), if you don't have it already.

Next, clone this repository by opening a terminal and typing the following commands:

    $ cd $HOME  # or any other development directory you prefer
    $ git clone https://github.com/gopala-kr/notebooks.git
    $ cd notebooks

If you want to go through chapter 16 on Reinforcement Learning, you will need to [install OpenAI gym](https://gym.openai.com/docs) and its dependencies for Atari simulations.

If you are familiar with Python and you know how to install Python libraries, go ahead and install the libraries listed in `requirements.txt` and jump to the [Starting Jupyter](#starting-jupyter) section. If you need detailed instructions, please read on.

### Python & Required Libraries

Of course, you obviously need Python. Python 2 is already preinstalled on most systems nowadays, and sometimes even Python 3. You can check which version(s) you have by typing the following commands:

    $ python --version   # for Python 2
    $ python3 --version  # for Python 3

Any Python 3 version should be fine, preferably ≥3.5. If you don't have Python 3, I recommend installing it (Python ≥2.6 should work, but it is deprecated so Python 3 is preferable). To do so, you have several options: on Windows or MacOSX, you can just download it from [python.org](https://www.python.org/downloads/). On MacOSX, you can alternatively use [MacPorts](https://www.macports.org/) or [Homebrew](https://brew.sh/). If you are using Python 3.6 on MacOSX, you need to run the following command to install the `certifi` package of certificates because Python 3.6 on MacOSX has no certificates to validate SSL connections (see this [StackOverflow question](https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)):

    $ /Applications/Python\ 3.6/Install\ Certificates.command

On Linux, unless you know what you are doing, you should use your system's packaging system. For example, on Debian or Ubuntu, type:

    $ sudo apt-get update
    $ sudo apt-get install python3

Another option is to download and install [Anaconda](https://www.continuum.io/downloads). This is a package that includes both Python and many scientific libraries. You should prefer the Python 3 version.

If you choose to use Anaconda, read the next section, or else jump to the [Using pip](#using-pip) section.

### Using Anaconda
When using Anaconda, you can optionally create an isolated Python environment dedicated to this project. This is recommended as it makes it possible to have a different environment for each project (e.g. one for this project), with potentially different libraries and library versions:

    $ conda create -n notebooks python=3.5 anaconda
    $ source activate notebooks

This creates a fresh Python 3.5 environment called `notebooks` (you can change the name if you want to), and it activates it. This environment contains all the scientific libraries that come with Anaconda. This includes all the libraries we will need (NumPy, Matplotlib, Pandas, Jupyter and a few others), except for TensorFlow, so let's install it:

    $ conda install -n notebooks -c conda-forge tensorflow=1.4.0

This installs TensorFlow 1.4.0 in the `notebooks` environment (fetching it from the `conda-forge` repository). If you chose not to create an `notebooks` environment, then just remove the `-n notebooks` option.

Next, you can optionally install Jupyter extensions. These are useful to have nice tables of contents in the notebooks, but they are not required.

    $ conda install -n notebooks -c conda-forge jupyter_contrib_nbextensions

You are all set! Next, jump to the [Starting Jupyter](#starting-jupyter) section.

### Using pip 
If you are not using Anaconda, you need to install several scientific Python libraries that are necessary for this project, in particular NumPy, Matplotlib, Pandas, Jupyter and TensorFlow (and a few others). For this, you can either use Python's integrated packaging system, pip, or you may prefer to use your system's own packaging system (if available, e.g. on Linux, or on MacOSX when using MacPorts or Homebrew). The advantage of using pip is that it is easy to create multiple isolated Python environments with different libraries and different library versions (e.g. one environment for each project). The advantage of using your system's packaging system is that there is less risk of having conflicts between your Python libraries and your system's other packages. Since I have many projects with different library requirements, I prefer to use pip with isolated environments.

These are the commands you need to type in a terminal if you want to use pip to install the required libraries. Note: in all the following commands, if you chose to use Python 2 rather than Python 3, you must replace `pip3` with `pip`, and `python3` with `python`.

First you need to make sure you have the latest version of pip installed:

    $ pip3 install --user --upgrade pip

The `--user` option will install the latest version of pip only for the current user. If you prefer to install it system wide (i.e. for all users), you must have administrator rights (e.g. use `sudo pip3` instead of `pip3` on Linux), and you should remove the `--user` option. The same is true of the command below that uses the `--user` option.

Next, you can optionally create an isolated environment. This is recommended as it makes it possible to have a different environment for each project (e.g. one for this project), with potentially very different libraries, and different versions:

    $ pip3 install --user --upgrade virtualenv
    $ virtualenv -p `which python3` env

This creates a new directory called `env` in the current directory, containing an isolated Python environment based on Python 3. If you installed multiple versions of Python 3 on your system, you can replace `` `which python3` `` with the path to the Python executable you prefer to use.

Now you must activate this environment. You will need to run this command every time you want to use this environment.

    $ source ./env/bin/activate

Next, use pip to install the required python packages. If you are not using virtualenv, you should add the `--user` option (alternatively you could install the libraries system-wide, but this will probably require administrator rights, e.g. using `sudo pip3` instead of `pip3` on Linux).

    $ pip3 install --upgrade -r requirements.txt

Great! You're all set, you just need to start Jupyter now.

### Starting Jupyter
If you want to use the Jupyter extensions (optional, they are mainly useful to have nice tables of contents), you first need to install them:

    $ jupyter contrib nbextension install --user

Then you can activate an extension, such as the Table of Contents (2) extension:

    $ jupyter nbextension enable toc2/main

Okay! You can now start Jupyter, simply type:

    $ jupyter notebook

This should open up your browser, and you should see Jupyter's tree view, with the contents of the current directory. If your browser does not open automatically, visit [localhost:8888](http://localhost:8888/tree). Click on `index.ipynb` to get started!

Note: you can also visit [http://localhost:8888/nbextensions](http://localhost:8888/nbextensions) to activate and configure Jupyter extensions.

Congrats! You are ready to learn Machine Learning, hands on! happy learning!!


------------

**reference**

* [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](http://www.amazon.com/Python-Data-Analysis-Wrangling-IPython/dp/1449319793) by Wes McKinney
* [PyCon 2015 Scikit-learn Tutorial](https://github.com/jakevdp/sklearn_pycon2015) by Jake VanderPlas
* [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) by Jake VanderPlas
* [Parallel Machine Learning with scikit-learn and IPython](https://github.com/ogrisel/parallel_ml_tutorial) by Olivier Grisel
* [Statistical Interference Using Computational Methods in Python](https://github.com/AllenDowney/CompStats) by Allen Downey
* [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples) by Aymeric Damien
* [TensorFlow Tutorials](https://github.com/pkmital/tensorflow_tutorials) by Parag K Mital
* [TensorFlow Tutorials](https://github.com/nlintz/TensorFlow-Tutorials) by Nathan Lintz
* [TensorFlow Tutorials](https://github.com/alrojo/tensorflow-tutorial) by Alexander R Johansen
* [TensorFlow Book](https://github.com/BinRoot/TensorFlow-Book) by Nishant Shukla
* [Summer School 2015](https://github.com/mila-udem/summerschool2015) by mila-udem
* [Keras tutorials](https://github.com/leriomaggio/deep-learning-keras-tensorflow) by Valerio Maggio
* [Kaggle](https://www.kaggle.com/)
* [Yhat Blog](http://blog.yhat.com/)
* [data-science-ipython-notebooks](https://github.com/donnemartin/data-science-ipython-notebooks)
* [awesome-datascience](https://github.com/bulutyazilim/awesome-datascience)
* [deep-learning-with-python-notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks)
* [handson-ml ](https://github.com/ageron/handson-ml)
* [dive-into-machine-learning](http://hangtwenty.github.io/dive-into-machine-learning/)
* [ipython-notebooks](https://github.com/jdwittenauer/ipython-notebooks)
* [scikit-learn-videos](https://github.com/justmarkham/scikit-learn-videos)


------------------------
