# NowcastPNN

This repository contains all code required for reproducing the results corresponding to the thesis paper titled "Attention-Based Probabilistic Neural Networks for Nowcasting of Dengue Fever in Brazil". All data necessary is available upon request. Any contributions to this repository and the NowcastPNN model are welcome.

## Table of Contents

- [Data](#data)
- [Outputs](#outputs)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Data

The dataset used in this project is sourced from the Brazilian Health Ministry (Ministério da saúde) and its associated DATASUS service. For this analysis, data on Dengue incidences in São Paulo from years 2013 to 2020 were used.

## Outputs

The outputs folder contains the results of the analyses. The subdirectory [figures](./outputs/figures/) contains all images used in the thesis.

## Installation

To run this project locally, follow these steps:

1. Clone the repository: git clone https://github.com/silaskoemen/NowcastPNN.git
2. Navigate to the project directory: cd NowcastPNN
3. Install required packages (preferably in a virtual environment to avoid conflicts): pip install -r requirements.txt OR pip3 install -r requirements.txt

## Usage

model predictions has dictionaries with PIs for all models in [model_predictions](./date/model_predictions/)
[src](./src/) contains all source code to run the models and analyses with notebooks available for detailed analysis corresponding to different sections of the paper. Most importantly, [NowcastPNN.py](./src/NowcastPNN.py) contains the code for the NowcastPNN models, both with and without embedding of weekdays.
The folder [weights](./src/weights/) contains the trained model weights for a variety of situations, including trained on randomly split versus sequentially split data, as well as weights for analysis investigating the number of training data points or past units needed. The suffix `-rec` here stands for trained on older data to be evaluated on the most recent observations and suffix `-dow/` stands for a model using the embedding block for the weekday.
Files [Epinowcast.R](./src/Epinowcast.R) and [RIVM.R](./src/RIVM.R) contain the code to form predictions from the benchmark models.
The model can thus be used directly by loading the desired weights. If you want to retrain the models, e.g. with different hyperparameter settings, just follow the outline of the jupyter notebooks. All plots created will be saved to the outputs folder directly.

## Contributing

Contributions are welcome. Please feel free to submit a pull request or open an issue if you encounter problems.

## License

This project is licensed under the MIT and CC-40 Licenses. See the `License.txt` file for details.
      
