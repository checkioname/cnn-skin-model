# About this repo


All the code done is made so that the results can be replicabe on any machine (need some hardware adjustments)
Because of this characterisc I decided to separate this repo in two parts


### First part -> data engineering 
So the first part of this repo contains code that can be used as a CLI for:
- Augmenting data (brightness, flip, resize, rotate, denoise)
- removing data background
- generating a csv with labels from a dir


The intention of the CLI is to setup the basics of your data simple CLI commands so you can run your training 

### Second part -> model engineering
The second part is all of the python code necessary to train the model
This code is also responsible for:
- defining the model
- training the model
- monitoring model metrics
- load Data with a custom DataLoader
- in memory pre processing (need to revisit it)
- generating indexes for k-fold training