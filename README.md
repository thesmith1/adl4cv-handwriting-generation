# adl4cv-handwriting-generation
Repository for Project in Advanced Deep Learning for Computer Vision at TUM

## Project Structure
The folder `model` contains the GAN classes and the models of all discriminators and generators.
It is also the base folder to store Tensorboard runs.

The folder `preprocessing` contains the code required to process the raw crops of characters and
the label files.

The folder `data_management` contains the class CharacterDataset.

The folder `utils` contains common definitions and frequently used functions, the crop utility
(required to crop the scanned pages of raw handwritten text) and the script used to compute the
content of the dataset.

The folder `stitching` contains the methods implementing our stitching technique.

The folder `webapp` contains the front (Angular 2) and the backend (Flask) of our web application.

In the end, in the root folder there are four scripts:
- `train.py` and `train_wasserstein.py`: the main training procedure
- `test.py`: this allows to test the generation of single characters
- `compute_FID_score.py`: this script quantitatively evaluates a trained model by means of the Fréchet Inception Distance.

## Installation

To create a new virtual environment with the required dependencies:

```conda env create -f adl4cv-env.yml```

To log into Tensorboard and check the training procedure live, an environment with Tensorflow (and Tensorboard)
is required. From that environment:

```tensorboard --logdir model/runs```

The webapp is based on Flask (Python 3 module) for the backend and on Angular 2 for the frontend. The modules necessary for the
backend are already included in `adl4cv-env.yml`. In order to setup Angular 2, simply run the bash script
`webapp/front/setup_frontend.sh`: this sets up NodeJS and the NPM package manager and installs all the components
needed by the app.

## Usage
Once installed the virtual environment, use the script `train.py` (or `train_wasserstein.py`) to train a conditional DCGAN
model (or conditional Wasserstein DCGAN).

In order to test the generation of _single characters_, use the script `test.py`, that produces results in Tensorboard.

In order to test the generation of entire sentences, the webapp can be used.
Launch the script `backend_server.py` to serve the backend. This script must be launched from its folder, not from the root of the project.
From the folder `webapp/front` launch `ng serve` on the command line to serve the frontend. This will compile the frontend
and will make it accessible on the browser at the address `localhost:4200`.

Alternatively, one can use the script `stitching/stitching.py` to see the results of the stitching only.
