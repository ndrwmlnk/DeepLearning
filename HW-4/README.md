# DeepLearning

1. use a different dataset (e.g. driver's license questionnaire, pandemic guide, etc. of 2-5 pages)
2. train two models: with a threshold for tokenizing words, for example, tokenize only words that occur more than 1 time in the text, and without this threshold and compare the results
3. find latent representation for each word (token)
4. write a function that returns top 5 synonyms for a given input word (closest Euclidean distance in latent representation). Words must be from the dictionary (tokens).