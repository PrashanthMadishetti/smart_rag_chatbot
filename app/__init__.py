# app/__init__.py
from .ingest.loaders import load_pdfs, load_txts, load_web
from .ingest.preprocess import clean, chunk


__all__ = ["load_pdfs", "load_txts", "load_web","clean","chunk"]

