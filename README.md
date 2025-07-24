# AutoTagger Intelligent Question Tagging with Difficulty & Intent Detection

**AutoTagger is an AI-powered NLP system designed to enhance how technical questions are classified on platforms like **StackOverflow** or **Quora**.** 

It automatically:
- Predicts relevant **topic tags**
- Estimates **difficulty level** (Easy, Medium, Hard)
- Detects **intent** (e.g., How-to, Debugging, Concept)
- Retrieves **similar previously answered questions**

All of this is integrated into a user-friendly **Streamlit web app** that delivers real-time predictions with **confidence scores**.

---

## Project Overview

### Key Features
- **Tag Prediction** – Multi-label classification (e.g., `["Python", "NLP"]`)
- **Difficulty Estimation** – Predicts whether a question is **Easy**, **Medium**, or **Hard**
- **Intent Detection** – Understands question type like **How-to**, **Debugging**, or **Concept**
- **Similar Questions Retrieval** – Suggests related solved questions using Sentence-BERT
- **Streamlit Web App** – Clean, interactive UI for real-time feedback
- **Confidence Scores** – Displays how confident each prediction is

---

## Rationale & Market Relevance

### Why AutoTagger?

Platforms like StackOverflow face:
- Inconsistent manual tagging
- Time-consuming moderation
- Poor understanding of user question intent

### Existing Limitations
- Traditional methods rely on keyword matching
- No unified system for **tagging**, **difficulty**, and **intent**
- Lack of smart retrieval for **similar questions**

### Our Solution
AutoTagger uses **machine learning** and **LLMs** to automate:
- Accurate tagging  
- Skill-level estimation  
- Understanding question intent  
- Retrieving relevant solutions

---

## Workflow & Methodology

### 1. Dataset
- **StackOverflow**: Titles, bodies, tags, scores, views
- **Manual Labeling**: 500–1000 entries for intent and difficulty

### 2. Preprocessing
- HTML tag removal, lowercasing
- Stopword removal, tokenization via **NLTK**
- Embedding generation: **TF-IDF** or **transformer-based**

### 3. Modeling
| Task                | Approach                                      |
|---------------------|-----------------------------------------------|
| **Tag Prediction**   | TF-IDF + Logistic Regression → BERT (Sigmoid) |
| **Difficulty**       | Rule-based + RandomForest → BERT              |
| **Intent Detection** | TF-IDF + Logistic Regression → GPT-4 zero-shot      |
| **Similarity**       | Sentence-BERT + Cosine Similarity             |

### 4. Tools & Frameworks
- **NLP**: spaCy, BeautifulSoup, `re`
- **ML/Embeddings**: Scikit-learn, TensorFlow, HuggingFace, Sentence-BERT
- **Interface**: Streamlit
- **Deployment**: Streamlit Cloud or Hugging Face Spaces

### 5. Development Pipeline
1. Data collection & manual labeling
2. Preprocessing & feature extraction
3. Model training & evaluation
4. Streamlit app development & integration

---

## ERD & System Workflow

### ERD (Entity Relationship Diagram)
| Entity              | Attributes                                       |
|---------------------|--------------------------------------------------|
| `QuestionData`       | ID, title, body, score, views                    |
| `PredictedTags`      | QuestionID, tag list, confidence scores          |
| `PredictedIntent`    | QuestionID, intent label, confidence score       |
| `PredictedDifficulty`| QuestionID, level (Easy/Medium/Hard), confidence|
| `SimilarQuestions`   | QuestionID, top-k similar IDs, similarity scores |

---

### User Workflow
1. User inputs a question into the Streamlit app.
2. System preprocesses the input text.
3. AutoTagger performs:
    - Tag prediction
    - Difficulty estimation
    - Intent detection
    - Similar question retrieval
4. UI displays results with:
    - Tags and their confidence
    - Difficulty level
    - Intent category
    - Similar questions with scores

---

## Installation

```bash
git clone https://github.com/yourusername/AutoTagger.git
cd AutoTagger
pip install -r requirements.txt
streamlit run app.py
```

### Author
- **Danish Karim**


