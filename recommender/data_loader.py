"""
Data loader module for Book-Crossing dataset files.
Handles latin-1 encoding, semicolon separators, missing values, and corrupted rows safely.
"""
import pandas as pd
from pathlib import Path
from config import DATA_DIR


def load_raw_books(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load BX-Books.csv with safe error handling and standard column names."""
    file_path = data_dir / "BX-Books.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Books file not found at {file_path}")

    df = pd.read_csv(
        file_path,
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False,
        dtype={
            "ISBN": str,
            "Book-Title": str,
            "Book-Author": str,
            "Year-Of-Publication": str,
            "Publisher": str,
            "Image-URL-S": str,
            "Image-URL-M": str,
            "Image-URL-L": str,
        },
    )

    # Standardize column names
    rename_dict = {
        "Book-Title": "Title",
        "Book-Author": "Author",
        "Year-Of-Publication": "Year",
        "Image-URL-S": "Image_URL_S",
        "Image-URL-M": "Image_URL_M",
        "Image-URL-L": "Image_URL_L",
    }
    df.rename(columns=rename_dict, inplace=True)
    return df


def load_raw_users(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load BX-Users.csv safely."""
    file_path = data_dir / "BX-Users.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Users file not found at {file_path}")

    df = pd.read_csv(
        file_path,
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False,
        dtype={"User-ID": str, "Location": str, "Age": str},
    )
    df.rename(columns={"User-ID": "User_ID"}, inplace=True)
    return df


def load_raw_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load BX-Book-Ratings.csv safely."""
    file_path = data_dir / "BX-Book-Ratings.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Ratings file not found at {file_path}")

    df = pd.read_csv(
        file_path,
        sep=";",
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False,
        dtype={"User-ID": str, "ISBN": str, "Book-Rating": str},
    )
    df.rename(
        columns={"User-ID": "User_ID", "Book-Rating": "Rating"},
        inplace=True,
    )
    return df
