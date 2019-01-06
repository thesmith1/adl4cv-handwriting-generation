# adl4cv-handwriting-generation
Repository for Project in Advanced Deep Learning for Computer Vision at TUM

## Installation

To create a new virtual environment with the required dependencies:

```conda env create -f adl4cv-env.yml```

To log into Tensorboard and check the training procedure live, an environment with Tensorflow (and Tensorboard)
is required. From that environment:

```tensorboard --logdir model/runs```


## File Structure
The folder `model/` contains the GAN classes and the models of all discriminators and generators.

The folder `data_management/` contains the class CharacterDataset.