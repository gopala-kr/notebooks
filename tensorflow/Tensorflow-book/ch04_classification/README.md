# Chapter 4

<p align="center"><a href="http://tensorflowbook.com" target="_blank"><img src="http://i.imgur.com/8pYWN0k.png"/></a></p>

Imagine an advertisement agency collecting information about user interactions to decide what type of ad to show. That’s not so uncommon. Google, Twitter, Facebook, and other big tech giants that rely on ads have creepy-good personal profiles of their users to help deliver personalized ads. A user who’s recently searched for gaming keyboards or graphics cards is probably more likely to click ads about the latest and greatest video games.

It may be difficult to cater a specially crafted advertisement for each individual, so grouping users into categories is a common technique. For example, a user may be categorized as a “gamer” to receive relevant video game related ads.

Machine learning has been the go-to tool to accomplish such as task. At the most fundamental level, machine learning practitioners want to build a tool to help them understand data. Being able to label data items into separate categories is an excellent way to characterize it for specific needs.

The previous chapter dealt with regression, which was about fitting a curve to data. If you recall, the best-fit curve is a function that takes as input a data item and assigns it a number. Creating a machine learning model that instead assigns discrete labels to its inputs is called classification. It is a supervised learning algorithm for dealing with discrete output. (Each discrete value is called a class.) The input is typically a feature vector, and the output is a class. If there are only two class labels (for example, True/False, On/Off, Yes/No), then we call this learning algorithm a binary classifier. Otherwise, it’s called a multiclass classifier.

- **Concept 1**: Linear regression for classification
- **Concept 2**: Logistic regression
- **Concept 3**: 2D Logistic regression
- **Concept 4**: Softmax classification

---

* Listing 1-3: `linear_1d.py`
* Listing 4: `logistic_1d.py`
* Listing 5: `logistic_2d.py`
* Listing 6-10: `softmax.py`
