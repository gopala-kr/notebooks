
    #environment setup
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
      
    #load watermark
    %load_ext watermark
    %watermark -a 'Gopala KR' -u -d -v -p watermark,numpy,pandas,matplotlib,nltk,sklearn,tensorflow,theano,mxnet,chainer,seaborn,keras
