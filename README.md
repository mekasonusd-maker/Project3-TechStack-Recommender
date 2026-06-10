# Project 3 - AI Tech Stack Recommender

## Overview
An AI-powered content-based filtering recommendation 
system that maps a user's skills to the most relevant 
tech career paths using TF-IDF vectorization and 
Cosine Similarity.

## How It Works
- User inputs minimum 3 skills
- Skills are converted to TF-IDF vectors
- Cosine Similarity matches against 15 job roles
- Top 3 career matches displayed with similarity scores

## Technologies Used
- Python 3.14
- Scikit-learn (TF-IDF + Cosine Similarity)
- Pandas
- Streamlit

## Files
- app.py — Main application and UI
- raw_skills.csv — Dataset of 15 job roles and skills

## How To Run
pip install pandas scikit-learn streamlit
streamlit run app.py

## Built By
Emmanuel — DecodeLabs Internship Batch 2026
