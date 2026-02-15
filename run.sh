#! /bin/bash
docker build --no-cache -t slide-extractor .
docker run -p 8501:8501 slide-extractor