# adl4cv-handwriting-generation
Repository for Project in Advanced Deep Learning for Computer Vision at TUM

## Installation

To create a new virtual environment with the required dependencies:

```conda env create -f adl4cv-env.yml```

To run the Crayon logging server:

```sudo docker run -d -p 8888:8888 -p 8889:8889 alband/crayon```
## File Structure
The folder `model/` contains the GAN classes and the models of all discriminators and generators.

The folder `data_management/` contains the class CharacterDataset.