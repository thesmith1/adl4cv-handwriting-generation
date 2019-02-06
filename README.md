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

The webapp is based on Flask (Python 3 module) for the backend and on Angular 2 for the frontend. The modules necessary for the
backend are already included in `adl4cv-env.yml`. In order to setup Angular 2, simply run the bash script
`webapp/front/setup_frontend.sh`: this sets up NodeJS and the NPM package manager and installs all the components
needed by the app.

## Usage

### Test
One model is already available inside the submission, for testing purposes.
In order to test the generation of _single characters_, use the script `test.py`, that produces results in Tensorboard.
To check such results in Tensorboard (available at localhost:6006):

```tensorboard --logdir model/runs```

In order to test the generation of _entire sentences_, the webapp can be used.
Launch the script `backend_server.py` to serve the backend. This script must be launched from its folder, not from the root of the project.
From the folder `webapp/front` launch `ng serve` on the command line to serve the frontend. This will compile the frontend
and will make it accessible on the browser at the address `localhost:4200`.
Alternatively, one can use the script `stitching/stitching.py` to see the results of the stitching only.

### Training

To perform a new training, the dataset is required. It is available at this address: https://drive.google.com/file/d/1wFOkU1hssaZWpzkJWPtvat8hPz0k2pEu/view?usp=sharing

Once big.zip has been downloaded, copy it and extract it in the "data" folder. The now unzipped folder "big" must contain a folder (the already processed images) and two files (the Inception features required to evaluate the FID score, and the label file for the dataset).

Once installed the virtual environment, use the script `train.py` (or `train_wasserstein.py`) to train a conditional DCGAN
model (or conditional Wasserstein DCGAN).

To log into Tensorboard and check the training procedure live, it is again possible to run:

```tensorboard --logdir model/runs```

With the inception features file at hand, computing the FID score of a model is made faster, by specifying the option "-p" followed by the inception features file path, when launching the `compute_FID_score.py` script.
