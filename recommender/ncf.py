"""
Neural Collaborative Filtering (NCF) architecture using TensorFlow / Keras.
Implements the Multi-Layer Perceptron (MLP) collaborative filtering model
learning non-linear user-item interaction features.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
from typing import List, Tuple, Dict, Optional
from config import NCF_LATENT_DIM, NCF_LEARNING_RATE


def build_ncf_model(
    num_users: int,
    num_books: int,
    latent_dim: int = NCF_LATENT_DIM,
    learning_rate: float = NCF_LEARNING_RATE,
) -> tf.keras.Model:
    """
    Builds the Neural Collaborative Filtering (NCF) MLP architecture.

    Architecture:
        User Input -> User Embedding
        Book Input -> Book Embedding
        Flatten both -> Concatenate
        Dense(64, ReLU)
        Dense(32, ReLU)
        Dense(16, ReLU)
        Dense(1, Sigmoid) -> Preference Score in [0, 1]
    """
    # 1. Inputs
    user_input = layers.Input(shape=(1,), name="user_id", dtype=tf.int32)
    book_input = layers.Input(shape=(1,), name="book_id", dtype=tf.int32)

    # 2. Embedding Layers
    user_embedding = layers.Embedding(
        input_dim=num_users,
        output_dim=latent_dim,
        name="user_embedding",
        embeddings_initializer="he_normal",
    )(user_input)

    book_embedding = layers.Embedding(
        input_dim=num_books,
        output_dim=latent_dim,
        name="book_embedding",
        embeddings_initializer="he_normal",
    )(book_input)

    # 3. Flatten Embeddings
    user_vec = layers.Flatten(name="flatten_user")(user_embedding)
    book_vec = layers.Flatten(name="flatten_book")(book_embedding)

    # 4. Concatenation
    concat = layers.Concatenate(name="concat_embeddings")([user_vec, book_vec])

    # 5. Multi-Layer Perceptron (MLP)
    x = layers.Dense(64, activation="relu", name="dense_64")(concat)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    x = layers.Dense(32, activation="relu", name="dense_32")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    x = layers.Dense(16, activation="relu", name="dense_16")(x)

    # 6. Output Preference Score in [0, 1]
    output = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs=[user_input, book_input], outputs=output, name="NCF_MLP_Recommender")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model


def predict_user_book_scores(
    model: tf.keras.Model,
    user_idx: int,
    book_indices: List[int],
    batch_size: int = 512,
) -> np.ndarray:
    """
    Batch predict preference scores for a given user across multiple candidate book indices.
    Returns a 1D numpy array of scores in [0, 1].
    """
    if not book_indices:
        return np.array([], dtype=np.float32)

    num_candidates = len(book_indices)
    user_array = np.full(num_candidates, user_idx, dtype=np.int32)
    book_array = np.array(book_indices, dtype=np.int32)

    predictions = model.predict(
        [user_array, book_array],
        batch_size=batch_size,
        verbose=0,
    )
    return predictions.flatten()
