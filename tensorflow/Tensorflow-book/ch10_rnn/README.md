# Chapter 10

<p align="center"><a href="http://tensorflowbook.com" target="_blank"><img src="http://i.imgur.com/IZbRx4E.png"/></a></p>

Back in school, I remember the sigh of relief when one of my midterm exams was made up of only true-or-false questions. I can’t be the only one that assumed half the answers would be “true” and the other half would be “false.”

I figured out answers to a most of the questions, and left the rest to random guessing. Actually, I did something clever, a strategy that you might have employed as well. After counting my number of “true” answers, I realized a disproportionate amount of “false” answers were lacking. So, a majority of my guesses were “false” to balance the distribution. 

It worked. I sure felt sly in the moment. What exactly is this feeling of craftiness that makes us feel so confident in our decisions, and how can we give a neural network the same power? 

One answer to this question is to use context to answer questions. Contextual cues are important signals that can also improve the performance of machine learning algorithms. For example, imagine you want to examine an English sentence and tag the part of speech of each word. The naive approach is to individually classify each word as a “noun,”, “adjective,”, and so on, without acknowledging its neighboring words. Consider trying that technique on the words in this sentence. The word “trying” was used as a verb, but depending on the context, you can also use it as an adjective, making parts-of-speech tagging a very trying problem. 

A better approach would consider the context. To bestow neural networks with contextual cues, we’ll study an architecture called a recurrent neural network. Instead of natural language data, we’ll be dealing with continuous timeseries data, similar to stock-market prices, as covered in previous chapters. By the end of the chapter, you’ll be able to model the patterns in timeseries data to make predictions about future value

- **Concept 1**: Loading timeseries data
- **Concept 2**: Recurrent neural networks
- **Concept 3**: Applying RNN to real-world data for timeseries prediction