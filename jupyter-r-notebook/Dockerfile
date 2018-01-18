FROM jupyter/notebook

MAINTAINER Saagie

ENV JAVA_HOME /usr/lib/jvm/java-7-openjdk-amd64
ENV HADOOP_CMD /usr/bin/hadoop
ENV SPARK_HOME /opt/spark

# R install
RUN sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
RUN echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" >> /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends r-base r-base-dev && apt-get clean

# Install Java
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-7-jdk && rm -rf /var/lib/apt/lists/* && apt-get clean

# Configure Java for R
RUN R CMD javareconf
RUN echo 'install.packages(c("rJava"),repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R && Rscript /tmp/packages.R

# Impala Jars
RUN mkdir /usr/lib/impala && mkdir /usr/lib/impala/lib && cd /usr/lib/impala/lib && \
curl -O https://downloads.cloudera.com/impala-jdbc/impala-jdbc-0.5-2.zip && unzip -j impala-jdbc-0.5-2.zip && rm impala-jdbc-0.5-2.zip

# Utilities for R Jupyter Kernel
RUN echo 'install.packages(c("base64enc","evaluate","IRdisplay","jsonlite","uuid","digest"), \
repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R \
   && Rscript /tmp/packages.R

RUN apt-get update && apt-get install -y --no-install-recommends libzmq3-dev && apt-get clean

# Database Libraries
RUN echo 'install.packages(c("RODBC","elastic","mongolite","rmongobd","RMySQL","RPostgreSQL","RJDBC","rredis","RCassandra","RHive","RNeo4j","RImpala"),repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R && Rscript /tmp/packages.R

# R HDFS Library
RUN echo 'install.packages("https://github.com/RevolutionAnalytics/rhdfs/blob/master/build/rhdfs_1.0.8.tar.gz?raw=true", repos = NULL, type = "source")' > /tmp/packages.R && Rscript /tmp/packages.R

# Machine Learning Libraries
RUN echo 'install.packages(c("dplyr","shiny","foreach","microbenchmark","parallel","runit","arules","arulesSequences","neuralnet","RSNNS","AUC","sprint","recommenderlab","acepack","addinexamples","clv","cubature","dtw","Formula","git2r","googleVis","gridExtra","gsubfn","hash","Hmisc","ifultools","latticeExtra","locpol","longitudinalData","lubridate","miniUI","misc3d","mvtsplot","np","openssl","packrat","pdc","PKI","rsconnect","splus2R","sqldf","TaoTeProgramming","TraMineR","TSclust","withr","wmtsa"), repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R && Rscript /tmp/packages.R

# Install R Jupyter Kernel
RUN echo 'install.packages(c("repr", "IRdisplay", "crayon", "pbdZMQ", "devtools"),repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R && Rscript /tmp/packages.R
RUN echo 'devtools::install_github("IRkernel/IRkernel")' > /tmp/packages.R && Rscript /tmp/packages.R

# Install R kernel
RUN echo 'IRkernel::installspec()' > /tmp/temp.R && Rscript /tmp/temp.R

#Run the notebook
CMD jupyter notebook \
    --ip=* \
    --MappingKernelManager.time_to_dead=10 \
    --MappingKernelManager.first_beat=3 \
    --notebook-dir=/notebooks-dir/
