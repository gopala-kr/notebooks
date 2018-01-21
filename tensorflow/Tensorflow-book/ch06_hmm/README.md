# Chapter 6

<p align="center"><a href="http://tensorflowbook.com" target="_blank"><img src="http://i.imgur.com/yhpbDGv.png"/></a></p>

If a rocket blows up, someone’s probably getting fired, so rocket scientists and engineers must be able to make confident decisions about all components and configurations. They do so by physical simulations and mathematical deduction from first principles. You, too, have solved science problems with pure logical thinking. Consider Boyle’s law: pressure and volume of a gas are inversely related under a fixed temperature. You can make insightful inferences from these simple laws that have been discovered about the world. Recently, machine learning has started to play the role of an important side-kick to deductive reasoning.

“Rocket science” and “machine learning” aren’t phrases that usually appear together.  But nowadays, modeling real-world sensor readings using intelligent data-driven algorithms is more approachable in the aerospace industry. Also, the use of machine learning techniques is flourishing in the healthcare and automotive industries. But why?

Part of the reason for this influx can be attributed to better understanding of interpretable models, which are machine learning models where the learned parameters have clear interpretations. If a rocket blows up, for example, an interpretable model might help trace the root cause.

This chapter is about exposing the hidden explanations behind observations. Consider a puppet-master pulling strings to make a puppet appear alive. Analyzing only the motions of the puppet might lead to over-complicated conclusions about how it’s possible for an inanimate object to move. Once you notice the attached strings, you’ll realize that a puppet-master is the best explanation for the life-like motions. 

- **Concept 1**: Forward algorithm
- **Concept 2**: Viterbi decode

--

* Listing 1-6: `forward.py`
* Listing 7-11: `hmm.py`
