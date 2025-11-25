# Quant Platform

Multi-asset quantitative research & trading system (work-in-progress).

Overview

This platform is a modular, production-oriented system for:

Equities, macro, and eventually multi-asset data ingestion.

Data validation with Pydantic schemas and domain-specific rules.

Feature computation for both equities and macro indicators.

Storing and retrieving features via an in-memory feature store.

End-to-end pipelines combining ingestion, feature engineering, and storage.

The project is structured for interview-grade rigor, modularity, and full test coverage.

Installation

Clone the repo:

git clone https://github.com/CarlosC21/quant-platform.git
cd quant-platform


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt

Usage
Run Tests
python -m pytest -v
python -m pytest --cov=src/quant_platform


All tests should pass, with coverage around 94%.

Ingest Data & Compute Features
from src.quant_platform.data.ingestion.pipelines import DataPipeline

pipeline = DataPipeline("path/to/equities.csv", "path/to/macro.csv")
feature_store = await pipeline.run()

print(feature_store.list_features())

Project Structure
src/
└─ quant_platform/
   ├─ data/
   │  ├─ feature_store/
   │  │  ├─ store.py
   │  │  └─ features.py
   │  ├─ ingestion/
   │  │  ├─ equities.py
   │  │  ├─ macro.py
   │  │  ├─ pipelines.py
   │  │  └─ base.py
   │  ├─ schemas/
   │  │  ├─ equities.py
   │  │  └─ macro.py
   │  └─ validation/
   │     └─ validators.py
   └─ __init__.py
tests/
└─ ...

Testing & Coverage

Unit tests: pytest for ingestion, validation, feature computation, and pipelines.

Async support: via pytest-asyncio.

Polars is used for fast DataFrame operations.

Pydantic v2 is used for schema validation.

Notes

Week 1 & 2 features are complete: repo structure, CI, basic pipelines, ingestion, validation, feature engineering.

All pipelines run successfully on sample data, and feature store saves/retrieves features correctly.