
    # pip environment setup
    !pip install watermark
    !pip install nltk
    !pip install theano
    !pip install mxnet
    !pip install chainer
    !pip install seaborn
    !pip install keras
    !pip install tensorflow-gpu
    !pip install scikit-image
    !pip install tqdm
    !pip install torch
    !pip install tflearn
    !pip install h5py
    !pip install gensim
    !pip install bokeh
    !pip install prettytable
      
    #load watermark
    %load_ext watermark
    %watermark -a 'Gopala KR' -u -d -v -p watermark,numpy,pandas,matplotlib,nltk,sklearn,tensorflow,theano,mxnet,chainer,seaborn,keras,tflearn,bokeh,gensim
