# Chapter 3

<p align="center"><a href="http://tensorflowbook.com" target="_blank"><img src="http://i.imgur.com/F2FOdon.png"/></a></p>

Remember science courses back in grade school? It might have been a while ago, or who knows - maybe you’re in grade school now starting your journey in machine learning early. Either way, whether you took biology, chemistry, or physics, a common technique to analyze data is to plot how changing one variable affects the other.

Imagine plotting the correlation between rainfall frequency and agriculture production. You may observe that an increase in rainfall produces an increase in agriculture rate. Fitting a line to these data points enables you to make predictions about the agriculture rate under different rain conditions. If you discover the underlying function from a few data points, then that learned function empowers you to make predictions about the values of unseen data.

Regression is a study of how to best fit a curve to summarize your data. It is one of the most powerful and well-studied types of supervised learning algorithms. In regression, we try to understand the data points by discovering the curve that might have generated them. In doing so, we seek an explanation for why the given data is scattered the way it is. The best fit curve gives us a model for explaining how the dataset might have been produced.

This chapter will show you how to formulate a real world problem to use regression. As you’ll see, TensorFlow is just the right tool that endows us with some of the most powerful predictors.

- **Concept 1**: Linear regression
- **Concept 2**: Polynomial regression
- **Concept 3**: Regularization

---

* Listing 1-2: `simple_model.py`
* Listing 3: `polynomial_model.py`
* Listing 4-5: `regularization.py`
* Listing 6: `data_reader.py`
