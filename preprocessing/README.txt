In preprocessing_PCA.py script file.

What I am doing is that firstly extract RUL label. And then clustering the environmental data.

Lastly I do the PCA in each environmental dataset and retain 99% variance in the original dataset.


Advantage:
Largely reduce the dimension of features to 10 or 11 with most variance remaining.

Disadvantage:
Doing PCA may decrease the model performance in a sense of information loss.