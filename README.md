# Shakespeare Text Generation Project

## Overview
This project aims to train a neural network model to generate text in the style of Shakespeare. It utilizes a character-level RNN to understand the patterns in Shakespeare's writings and generate text that mimics his style. This README covers the setup and running instructions for both the Jupyter Notebook and the standalone Python script versions of the project.

## Project Structure

- `train.ipynb`: Jupyter Notebook containing the step-by-step process of building, training, and generating text with the model.
- `train.py`: Python script that encapsulates the model training and text generation logic for command-line execution.
- `tinyshakespeare.txt`: Dataset file containing the text of Shakespeare's works. (Note: This file needs to be downloaded separately.)
- `README.md`: This file, providing an overview and instructions for the project.

## Prerequisites

- Python 3.6 or above
- NumPy
- (Optional for Jupyter Notebook) JupyterLab or Jupyter Notebook

## Setup Instructions

1. **Clone the Repository**: Clone this repository to your local machine to get started.

   ```
   git clone <repository-url>
   ```

2. **Environment Setup**: It's recommended to create a virtual environment for this project to manage dependencies easily.

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:

   ```
   pip install numpy jupyterlab
   ```

4. **Download the Dataset**: The Shakespeare dataset can be downloaded from [this link](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). Place it in the same directory as the notebook and script.

## Running the Project

### Jupyter Notebook

1. Launch JupyterLab or Jupyter Notebook:

   ```
   jupyter lab
   ```
   or
   ```
   jupyter notebook
   ```

2. Navigate to the `train.ipynb` file and open it.

3. Run the cells in sequence by pressing `Shift + Enter` for each cell. Note: Cells 3 and 5 are essential as they define neural network layers and utilities.

### Python Script

1. Ensure the `tinyshakespeare.txt` dataset is in the same directory as `train.py`.

2. Run the script:

   ```
   python train.py
   ```



## License

This project is licensed under the [MIT](LICENSE.md).
