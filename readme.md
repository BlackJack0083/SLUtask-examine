# Joint Intent Recognition and Slot Filling Algorithm Tests

This repository contains implementations and tests for several fundamental models applied to the Joint Intent Recognition (IR) and Slot Filling (SF) tasks.

## Tasks

* **Intent Recognition (IR):** Classifying the overall goal or intent of an utterance (e.g., "book a flight").
* **Slot Filling (SF):** Identifying and extracting specific pieces of information (slots) within the utterance that are relevant to the intent (e.g., "New York" as destination, "tomorrow" as date).

## Implemented Models

This project includes implementations for testing the following model architectures:

* Convolutional Neural Network (CNN)
* Long Short-Term Memory (LSTM)
* Gated Recurrent Unit (GRU)
* Result-based Bi-Feedback Network ([RBFN](https://arxiv.org/abs/1905.03969)) 
* BERT-based models (specifically, using a frozen BERT encoder with custom downstream layers) 
* Transformer(come soon)

## Getting Started

### Dataset

The dataset used in this project is sourced from the [AGIF repository](https://github.com/LooperXX/AGIF). Please refer to their repository for details on how to obtain and preprocess the raw data.

### Pre-trained Models

For the BERT-based experiments, the [bert-base-cased](https://huggingface.co/bert-base-cased) model from Hugging Face Transformers is used. If you are running experiments offline, ensure you have this model downloaded locally in the `bert-base-cased` directory at the root of this project.

### Configuration

Hyperparameters and training settings can be adjusted by modifying the `configs/config.yaml` file.

## Usage

1.  **Data Preprocessing:**
    Prepare the dataset by running the preprocessing script. Ensure you have placed the raw data in the expected location (`data/` folder as per your `config.yaml`).

    ```bash
    python data_processing.py
    ```

2.  **Model Training:**
    Train a model using the main script. The model type is specified via command-line argument or the default in `main.py`.

    ```bash
    # Train the default model (e.g., BERT as per your config)
    python main.py

    # Or train a specific model type
    python main.py --model_type lstm
    # Available types: cnn, lstm, gru, rbfn, bert
    ```

The training process will output loss and evaluation metrics, and the trained model checkpoint will be saved to the `checkpoints/` directory.