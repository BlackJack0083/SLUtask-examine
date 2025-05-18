#!/bin/bash

# This script runs hyperparameter search for each model type.

# Define the list of model types to search
MODEL_TYPES=("cnn" "lstm" "gru" "rbfn" "bertfreeze" "bert" "transformer")
# MODEL_TYPES=("cnn" )

# Define the number of epochs to run for each search combination
SEARCH_EPOCHS=50

# Define the base directory to save results
SAVE_BASE_DIR="checkpoints/hyperparameter_search"

echo "Starting hyperparameter search for all model types..."
echo "Search epochs per combination: ${SEARCH_EPOCHS}"
echo "Results will be saved in: ${SAVE_BASE_DIR}"
echo "---"

# Loop through each model type
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    echo ""
    echo "=================================================="
    echo "Running search for model type: ${MODEL_TYPE}"
    echo "=================================================="
    echo ""

    # Run the python search script for the current model type
    # Redirect stdout and stderr to log files for each model type
    # Optional: Add & to run in background if desired, but sequential is simpler for logs
    python hyperparameter_search.py \
        --model_type "${MODEL_TYPE}" \
        --search_epochs "${SEARCH_EPOCHS}" \
        --save_dir "${SAVE_BASE_DIR}"

    # Check the exit status of the python script
    if [ $? -eq 0 ]; then
        echo "Successfully finished search for ${MODEL_TYPE}"
    else
        echo "Error occurred during search for ${MODEL_TYPE}"
        # Optional: break or continue based on desired behavior
    fi

    echo "---"

done

echo "All model types search finished."