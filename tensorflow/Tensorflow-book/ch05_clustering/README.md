# Chapter 5

<p align="center"><a href="http://tensorflowbook.com" target="_blank"><img src="http://i.imgur.com/OMR8tkf.png"/></a></p>

Suppose there’s a collection of not-pirated-totally-legal mp3s on your hard drive. All your songs are crowded in one massive folder. But it might help to automatically group together similar songs and organize them into categories like “country,” “rap,” “rock,” and so on. This act of assigning an item to a group (such as an mp3 to a playlist) in an unsupervised fashion is called clustering.

The previous chapter on classification assumes you’re given a training dataset of correctly labeled data. Unfortunately, we don’t always have that luxury when we collect data in the real-world. For example, suppose we would like to divide up a large amount of music into interesting playlists. How could we possibly group together songs if we don’t have direct access to their metadata?

Spotify, SoundCloud, Google Music, Pandora, and many other music streaming services try to solve this problem to recommend similar songs to customers. Their approach includes a mixture of various machine learning techniques, but clustering is often at the heart of the solution.

The overall idea of clustering is that two items in the same cluster are “closer” to each other than items that belong to separate clusters. That is the general definition, leaving the interpretation of “closeness” open. For example, perhaps cheetahs and leopards belong in the same cluster, whereas elephants belong to another when closeness is measured by how similar two species are in the hierarchy of biological classification (family, genus, and species).

You can image there are many clustering algorithms out there. In this chapter we’ll focus on two types, namely k-means and self-organizing map. These approaches are completely unsupervised, meaning they fit a model without ground-truth examples.

- **Concept 1**: Clustering
- **Concept 2**: Segmentation
- **Concept 3**: Self-organizing map

---

* Listing 1-4: `audio_clustering.py`
* Listing 5-6: `audio_segmentation.py`
* Listing 7-12: `som.py`
