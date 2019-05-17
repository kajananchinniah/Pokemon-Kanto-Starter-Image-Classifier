# Pokemon-Kanto-Starter-Image-Classifier

Image clasifier implemented using PyTorch. 

Originally I had wanted to classifiy all 151 Pokemon, but it was out of my scope (computation power wise + time).
~90% accuracy was achieved on my validation set.

Dataset I used: https://www.kaggle.com/thedagger/pokemon-generation-one/

Note: I only used the files that had the four starters (Bulbasaur, Charmander, Pikachu and Squirtle)

Later, I used a pretrained densenet network but retrained the classifier on the original dataset from above (all 151 Kanto Pokemon). 

Through this process, I was able to get 73% accuracy on my validation set. 
