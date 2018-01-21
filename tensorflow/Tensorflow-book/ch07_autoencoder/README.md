# Chapter 7

<p align="center"><a href="http://tensorflowbook.com" target="_blank"><img src="http://i.imgur.com/sBoP8cg.png"/></a></p>


Have you ever identified a song from a person just humming a melody? It might be easy for you, but I’m comically tone-deaf when it comes to music. Humming, by itself, is an approximation of its corresponding song. An even better approximation could be singing. Include some instrumentals, and sometimes a cover of a song sounds indistinguishable from the original.

Instead of songs, in this chapter, we will approximate functions. Functions are a very general notion of relations between inputs and outputs. In machine learning, we typically want to find the function that relates inputs to outputs. Finding the best possible function is difficult, but approximating the function is much easier.

Conveniently, artificial neural networks are a model in machine learning that can approximate any function. Given training data, we want to build a neural network model that best approximates the implicit function that might have generated the data.

After introducing neural networks in section 8.1, we’ll learn how to use them to encode data into a smaller representation in section 8.2, using a network structure called an autoencoder. 

- **Concept 1**: Autoencoder
- **Concept 2**: Applying an autoencoder to images
- **Concept 3**: Denoising autoencoder