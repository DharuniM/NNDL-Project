"""
Preprocessing and feature engineering module for Book-Crossing data.
Integrates enriched metadata (genres, subjects, descriptions) with tabular interaction datasets.
Constructs rich structured semantic documents for Sentence-BERT embedding.
"""
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from config import MIN_BOOK_RATINGS, MIN_USER_RATINGS, MAX_CATALOG_BOOKS, DATA_DIR


def clean_isbn(isbn_val) -> str:
    """Clean and standardize ISBN strings."""
    if pd.isna(isbn_val):
        return ""
    cleaned = str(isbn_val).strip().upper()
    cleaned = re.sub(r"[^0-9X]", "", cleaned)
    return cleaned


def clean_year(year_val) -> int:
    """Clean and validate publication year."""
    try:
        y = int(str(year_val).strip())
        if 1800 <= y <= 2026:
            return y
        return 0
    except (ValueError, TypeError):
        return 0


def build_semantic_document(row: pd.Series) -> str:
    """
    Constructs a rich structured semantic document from available metadata:
    Title, Author, Genres, Subjects, Description, Publisher, and Year.
    Does not invent fabricated text when metadata is missing.
    """
    parts = [
        f"Title: {row.get('Title', 'Unknown Title')}",
        f"Author: {row.get('Author', 'Unknown Author')}",
    ]

    genres = str(row.get("Genres", "")).strip()
    if genres and genres not in ["nan", "None", ""]:
        parts.append(f"Genres: {genres}")

    subjects = str(row.get("Subjects", "")).strip()
    if subjects and subjects not in ["nan", "None", ""]:
        parts.append(f"Subjects: {subjects}")

    description = str(row.get("Description", "")).strip()
    if description and description not in ["nan", "None", ""]:
        parts.append(f"Description: {description}")

    publisher = str(row.get("Publisher", "")).strip()
    if publisher and publisher not in ["nan", "None", ""]:
        parts.append(f"Publisher: {publisher}")

    year = row.get("Year", 0)
    try:
        year_int = int(year)
        if year_int > 0:
            parts.append(f"Year: {year_int}")
    except (ValueError, TypeError):
        pass

    return "\n".join(parts)


def clean_books_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw books dataframe."""
    df = df.copy()
    df["ISBN"] = df["ISBN"].apply(clean_isbn)
    df = df[df["ISBN"] != ""].drop_duplicates(subset=["ISBN"])

    df["Title"] = df["Title"].fillna("Unknown Title").astype(str).str.strip()
    df["Author"] = df["Author"].fillna("Unknown Author").astype(str).str.strip()
    df["Publisher"] = df["Publisher"].fillna("Unknown Publisher").astype(str).str.strip()
    df["Year"] = df["Year"].apply(clean_year)

    # Initialize empty metadata fields
    df["Genres"] = ""
    df["Subjects"] = ""
    df["Description"] = ""

    # Clean Image URLs: replace http:// with https:// for secure browser loading
    for col in ["Image_URL_S", "Image_URL_M", "Image_URL_L"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
            df[col] = df[col].apply(lambda u: u.replace("http://", "https://") if u.startswith("http://") else u)

    return df


def load_enriched_metadata(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load enriched metadata file if available."""
    json_path = data_dir / "book_metadata_enriched.json"
    if not json_path.exists():
        return pd.DataFrame()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    enriched_df = pd.DataFrame(data)
    if "ISBN" in enriched_df.columns:
        enriched_df["ISBN"] = enriched_df["ISBN"].apply(clean_isbn)
    return enriched_df


def clean_users_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw users dataframe."""
    df = df.copy()
    df["User_ID"] = df["User_ID"].astype(str).str.strip()
    df = df[df["User_ID"] != ""].drop_duplicates(subset=["User_ID"])
    df["Location"] = df["Location"].fillna("Unknown").astype(str).str.strip()

    def parse_age(a):
        try:
            val = float(a)
            if 5 <= val <= 105:
                return int(val)
            return None
        except (ValueError, TypeError):
            return None

    df["Age"] = df["Age"].apply(parse_age)
    return df


def clean_ratings_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw ratings dataframe."""
    df = df.copy()
    df["User_ID"] = df["User_ID"].astype(str).str.strip()
    df["ISBN"] = df["ISBN"].apply(clean_isbn)

    def parse_rating(r):
        try:
            val = float(r)
            if 0 <= val <= 10:
                return int(val)
            return None
        except (ValueError, TypeError):
            return None

    df["Rating"] = df["Rating"].apply(parse_rating)
    df = df.dropna(subset=["Rating"])
    df["Rating"] = df["Rating"].astype(int)
    df = df[(df["User_ID"] != "") & (df["ISBN"] != "")]
    df = df.drop_duplicates(subset=["User_ID", "ISBN"])
    return df


def process_dataset(
    df_books: pd.DataFrame,
    df_users: pd.DataFrame,
    df_ratings: pd.DataFrame,
    min_book_ratings: int = MIN_BOOK_RATINGS,
    min_user_ratings: int = MIN_USER_RATINGS,
    max_catalog_books: int = MAX_CATALOG_BOOKS,
    data_dir: Path = DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict, Dict, Dict]:
    """
    Process dataset, merge enriched metadata, and build sequential ID encodings.
    """
    cleaned_books = clean_books_df(df_books)
    cleaned_users = clean_users_df(df_users)
    cleaned_ratings = clean_ratings_df(df_ratings)

    # 1. Filter ratings to books present in cleaned_books
    valid_isbns = set(cleaned_books["ISBN"])
    ratings = cleaned_ratings[cleaned_ratings["ISBN"].isin(valid_isbns)].copy()

    # Calculate book rating counts
    book_counts = ratings["ISBN"].value_counts()
    frequent_books = book_counts[book_counts >= min_book_ratings].index

    if len(frequent_books) > max_catalog_books:
        frequent_books = frequent_books[:max_catalog_books]

    ratings = ratings[ratings["ISBN"].isin(frequent_books)].copy()

    # Calculate user rating counts on the filtered books
    user_counts = ratings["User_ID"].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    ratings = ratings[ratings["User_ID"].isin(active_users)].copy()

    # Final list of books from filtered ratings
    active_isbns = set(ratings["ISBN"].unique())
    final_books = cleaned_books[cleaned_books["ISBN"].isin(active_isbns)].copy()

    # 2. Integrate Enriched Metadata Source
    enriched_df = load_enriched_metadata(data_dir)
    if not enriched_df.empty:
        # Merge enriched metadata for existing books
        for _, e_row in enriched_df.iterrows():
            e_isbn = e_row["ISBN"]
            e_title = e_row.get("Title", "")
            e_author = e_row.get("Author", "")

            # Match by ISBN or (Title + Author)
            match_mask = (final_books["ISBN"] == e_isbn) | (
                (final_books["Title"].str.lower() == str(e_title).lower()) &
                (final_books["Author"].str.lower() == str(e_author).lower())
            )

            if match_mask.any():
                final_books.loc[match_mask, "Genres"] = e_row.get("Genres", "")
                final_books.loc[match_mask, "Subjects"] = e_row.get("Subjects", "")
                final_books.loc[match_mask, "Description"] = e_row.get("Description", "")
                if e_row.get("Image_URL_L"):
                    final_books.loc[match_mask, "Image_URL_L"] = e_row.get("Image_URL_L")
            else:
                # Add new enriched reference title to catalog
                new_book_record = {
                    "ISBN": e_isbn,
                    "Title": e_title,
                    "Author": e_author,
                    "Year": e_row.get("Year", 0),
                    "Publisher": e_row.get("Publisher", "Unknown Publisher"),
                    "Image_URL_S": e_row.get("Image_URL_L", ""),
                    "Image_URL_M": e_row.get("Image_URL_L", ""),
                    "Image_URL_L": e_row.get("Image_URL_L", ""),
                    "Genres": e_row.get("Genres", ""),
                    "Subjects": e_row.get("Subjects", ""),
                    "Description": e_row.get("Description", ""),
                }
                final_books = pd.concat([final_books, pd.DataFrame([new_book_record])], ignore_index=True)

    final_books = final_books.drop_duplicates(subset=["ISBN"]).reset_index(drop=True)

    # 3. Compute book statistics
    rating_stats = (
        ratings.groupby("ISBN")
        .agg(
            Rating_Count=("Rating", "count"),
            Avg_Rating=("Rating", "mean"),
            Explicit_Count=("Rating", lambda s: (s > 0).sum()),
            Explicit_Avg=("Rating", lambda s: s[s > 0].mean() if (s > 0).sum() > 0 else s.mean()),
        )
        .reset_index()
    )

    final_books = final_books.merge(rating_stats, on="ISBN", how="left")
    final_books["Rating_Count"] = final_books["Rating_Count"].fillna(0).astype(int)
    final_books["Avg_Rating"] = final_books["Avg_Rating"].fillna(0.0).round(2)
    final_books["Explicit_Count"] = final_books["Explicit_Count"].fillna(0).astype(int)
    final_books["Explicit_Avg"] = final_books["Explicit_Avg"].fillna(0.0).round(2)

    # Construct the rich structured semantic document for every book
    final_books["Semantic_Document"] = final_books.apply(build_semantic_document, axis=1)
    final_books["Semantic_Text"] = final_books["Semantic_Document"]  # Backward compatibility

    # 4. Build sequential ID encodings
    all_isbns = final_books["ISBN"].tolist()
    isbn_to_idx = {isbn: idx for idx, isbn in enumerate(all_isbns)}
    idx_to_isbn = {idx: isbn for idx, isbn in enumerate(all_isbns)}

    unique_users = sorted(ratings["User_ID"].unique())
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    idx_to_user = {idx: uid for idx, uid in enumerate(unique_users)}

    ratings["user_idx"] = ratings["User_ID"].map(user_to_idx)
    ratings["book_idx"] = ratings["ISBN"].map(isbn_to_idx)
    ratings = ratings.dropna(subset=["user_idx", "book_idx"]).copy()
    ratings["user_idx"] = ratings["user_idx"].astype(int)
    ratings["book_idx"] = ratings["book_idx"].astype(int)

    final_users = cleaned_users[cleaned_users["User_ID"].isin(set(unique_users))].copy().reset_index(drop=True)

    return (
        final_books,
        final_users,
        ratings,
        user_to_idx,
        idx_to_user,
        isbn_to_idx,
        idx_to_isbn,
    )
