 # Dynamic Maximal Independent Set on a Matroid

This project aims to understand the complexity of finding a maximal independent set on a matroid when the underlying ground 
set E undergoes changes, i.e., elements are either inserted or deleted from the ground set. 
There are two main approaches to attack this problem. The first one is based on the idea of maintaining a random permutation of the ground set, while the second one runs the greedy algorithms from scratch and exploits the stability of matroids. 
The goal is to empirically evaluate the performance of these two approaches, and understand whether randomness helps in improving the running time for this problem.
