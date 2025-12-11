# # MIDI-Music-Classification-Genre-and-Mood-using-REMI-z

This project aims to classify MIDI music by genre using a pre-trained Transformer model. It employs a hierarchical aggregation approach to effectively process MIDI data, tokenized into the REMI-z format, for multi-label genre classification.

## Setup and Installation

To get started with this project, follow these steps to set up your environment and install the necessary libraries.

1.  **Clone the REMI-z repository:**

    ```bash
    !git clone https://github.com/Sonata165/REMI-z.git
    %cd REMI-z
    !pip install -r Requirements.txt
    !pip install -e .
    %cd ..
    ```

2.  **Install Hugging Face Transformers and Accelerate:**

    ```bash
    !pip install transformers accelerate
    ```

3.  **Install other necessary libraries:**

    ```bash
    !pip install datasets miditoolkit pandas tqdm lightning
    ```

## Dataset Preparation

The project uses the MidiCaps dataset, which contains MIDI file locations and associated genre labels.

1.  **Download LMD Full dataset:**

    ```bash
    !wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
    !tar -xzf lmd_full.tar.gz
    ```

2.  **Load MidiCaps dataset from Hugging Face:**

    ```python
    from datasets import load_dataset
    ds = load_dataset("amaai-lab/MidiCaps")
    df = ds['train'].to_pandas()
    ```

3.  **Extract relevant columns and map genres to numerical indices:**

    The `df` DataFrame is processed to extract `location` and `genre` information. A `CLASS_TO_INDEX` mapping is created for multi-label classification, derived from a stratified sample of the original dataset. The final `df` is a stratified sample of the original MidiCaps dataset, ensuring representation across genres.

## Data Preprocessing and Chunking Explanation

To manage the varying lengths of MIDI files and fit them into the Transformer model's fixed input size, a crucial preprocessing step involves chunking the musical sequences. This process is handled by a custom `process_midi_to_segments` function:

*   MIDI files are first converted into `MultiTrack` objects using the REMI-z library.
*   Empty bars are filtered out from the `MultiTrack` object to ensure only meaningful musical content is processed.
*   The filtered `MultiTrack` is then segmented into fixed-size chunks, specifically `BARS_PER_SEGMENT` bars (e.g., 8-bar segments). This standardization helps in creating consistent input lengths for the model.
*   Each chunk is converted into its REMI-z string representation, which is a sequence of tokens.
*   Each REMI-z string is tokenized with `MAX_CHUNK_LENGTH` to ensure uniform input size for the Transformer. Tokens are also padded to this length.
*   A `MIDIMultiLabelDataset` is used to efficiently load and prepare data, and a `midi_custom_collate_fn` handles batching, padding both the number of chunks and the token length within each batch to accommodate variable-length MIDI files.

## Model Architecture Details

This project leverages a pre-trained `LongshenOu/m2m_pt` tokenizer and `GPT2LMHeadModel` from Hugging Face Transformers. The architecture consists of:

1.  **Feature Extractor**: The `GPT2Model` (the `model.transformer` part) is extracted from the pre-trained model and its parameters are frozen. This allows it to serve as a fixed feature extractor, providing rich contextual embeddings for MIDI sequences.
2.  **Custom Classification Head (`ChunkAggregationHead`)**: A custom PyTorch module is implemented for multi-label classification. It takes the hidden states from the feature extractor (one embedding per segment), applies an attention mechanism to weigh these segment embeddings, aggregates them into a single document-level embedding, and finally passes this through a dropout layer and a linear layer to produce logits for `NUM_CLASSES` genres.
    *   **Attention Mechanism**: Employs a hierarchical attention mechanism (`chunk_attention`) to learn the importance of different segments within a MIDI document, allowing the model to focus on the most relevant parts for genre classification.
3.  **Combined Model (`GPT2ForHierarchicalClassification`)**: This wrapper class integrates the frozen `feature_extractor` and the trainable `ChunkAggregationHead`. During the forward pass, it reshapes the input to process individual chunks through the feature extractor, then aggregates the resulting chunk embeddings via the `aggregation_head` to produce the final genre classification logits.

The `EMBEDDING_DIM` of the `GPT2Model` is `768`, and the `NUM_CLASSES` is `42` (corresponding to the unique genres identified in the dataset).

## Training with PyTorch Lightning

The model is fine-tuned using PyTorch Lightning for efficient and organized training. The training process involves:

1.  **`GenreClassificationData` DataModule**: This custom DataModule handles the creation of `MIDIMultiLabelDataset` instances for training, validation, and test sets. It also manages data splitting (e.g., `val_split=0.1`, `test_split=0.1`) and provides `DataLoader`s with the custom `midi_custom_collate_fn` to correctly batch the variable-length and variable-chunk MIDI data.

2.  **`GenreClassifier` LightningModule**: This module wraps our `GPT2ForHierarchicalClassification` model and defines the training, validation, and test steps. It uses `torch.nn.BCEWithLogitsLoss` as the loss function, appropriate for multi-label classification. Metrics like `train_loss`, `val_loss`, `val_acc`, `test_loss`, and `test_acc` are logged.

3.  **Optimizer**: `torch.optim.AdamW` is used, configured to optimize only the trainable parameters of the `aggregation_head` (the custom classification head), with a learning rate of `1e-4`.

4.  **Trainer**: A PyTorch Lightning `Trainer` is instantiated with `max_epochs=1` (for demonstration purposes, though more epochs would typically be used for convergence) and configured to use a GPU if available.

The training loop is managed by the Lightning `Trainer.fit()` method, which handles moving data to the device, zeroing gradients, forward passes, loss calculation, backward passes, and optimizer steps automatically.

## Model Evaluation Results

After training for 1 epoch, the model was evaluated on the test set. The results are as follows:

*   **`test_loss`**: 0.2378
*   **`test_acc`**: 0.9099

These results demonstrate a promising initial performance for genre classification. The hierarchical aggregation approach, combined with efficient data handling, successfully resolved memory issues that could arise from processing large MIDI files, allowing for effective training on the available GPU resources.

## Model Saving and Inference Instructions

After training, the model's state dictionary is saved to allow for future use without retraining. The `predict_genre` function demonstrates how to load the saved model and perform inference on new MIDI files.

1.  **Saving the Model**:

    The trained model's state dictionary is saved to a `.pth` file:

    ```python
    import os
    import torch

    CHECKPOINT_DIR = "model_checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_save_path = os.path.join(CHECKPOINT_DIR, "genre_classifier_model.pth")

    torch.save(lightning_model.model_with_head.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    ```

2.  **Loading the Model and Inference**:

    To load the model and make predictions on a new MIDI file, you can use the `predict_genre` function. This function processes a given MIDI file, applies the model, and returns the top predicted genres with their probabilities.

    ```python
    def predict_genre(midi_file_path, model, tokenizer, bars_per_segment, class_to_idx, num_classes, max_token_length, device, threshold=0.5, top_k=2):
        # Ensure model is in evaluation mode
        model.eval()

        # Prepare the MIDI data for inference
        try:
            tokenized_segments = process_midi_to_segments(midi_file_path, bars_per_segment, tokenizer, max_token_length)
        except ValueError as e:
            print(f"Error processing MIDI file: {e}")
            return []
        if not tokenized_segments:
            return []

        # Pad segments for this single document
        max_segment_len = max(len(seg) for seg in tokenized_segments)
        padded_segments = []
        for segment in tokenized_segments:
            padding_needed = max_segment_len - len(segment)
            padded_segment = segment + [tokenizer.pad_token_id] * padding_needed
            padded_segments.append(padded_segment)

        # Convert to tensors and add batch dimension (Batch size = 1 for single inference)
        input_ids = torch.tensor(padded_segments, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).int().to(device)

        # Perform inference
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits).squeeze(0) # Remove batch dimension

        # Convert probabilities to genre labels
        predicted_genres = []
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        for i, prob in enumerate(probabilities):
            if prob > threshold:
                predicted_genres.append((idx_to_class[i], prob.item()))

        # Sort by probability descending
        predicted_genres.sort(key=lambda x: x[1], reverse=True)

        # Return only the top_k genres
        return predicted_genres[:top_k]

    # Example usage
    sample_midi_path = "/content/lmd_full/2/2a604c93ed4ee9d82d388da694e89e7c.mid" # Replace with your MIDI file

    # Re-instantiate the model structure
    loaded_model = GPT2ForHierarchicalClassification(feature_extractor, classifier_head).to(device)
    loaded_model.load_state_dict(torch.load(model_save_path))
    loaded_model.eval()

    predicted_genres = predict_genre(
        midi_file_path=sample_midi_path,
        model=loaded_model,
        tokenizer=tokenizer,
        bars_per_segment=BARS_PER_SEGMENT,
        class_to_idx=CLASS_TO_INDEX,
        num_classes=NUM_CLASSES,
        max_token_length=MAX_CHUNK_LENGTH,
        device=device,
        threshold=0.01, # Example threshold
        top_k=3 # Example top K predictions
    )

    print(f"Predicted Genres for {sample_midi_path}:")
    for genre, prob in predicted_genres:
        print(f"  - {genre}: {prob:.4f}")
    ```

## Future Work/Improvements

*   **Longer Training**: Train the model for more epochs to observe potential improvements in convergence and overall performance.
*   **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and dropout probabilities to find optimal training configurations.
*   **Explore Aggregation Strategies**: Investigate alternative methods for aggregating chunk embeddings (e.g., attention over instruments, other pooling strategies).
*   **Dataset Expansion**: Incorporate a larger and more diverse dataset for training to enhance the model's generalization capabilities.
*   **Evaluation Metrics**: Implement additional evaluation metrics suitable for multi-label classification (e.g., F1-score, Jaccard index, Hamming loss) to provide a more comprehensive assessment of model performance.
*   **Real-time Inference**: Optimize the `predict_genre` function and the overall model for faster, potentially real-time, genre prediction.

---
