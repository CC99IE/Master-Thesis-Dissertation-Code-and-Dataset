import os
import json

from PIL import Image, ExifTags
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from docx import Document
from PyPDF2 import PdfReader
from langdetect import detect, DetectorFactory
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from .common_warnings import *    # this sets up filters for PyPDF2

Image.MAX_IMAGE_PIXELS = None

# Ensure reproducibility for language detection
DetectorFactory.seed = 0

# Files to ignore during content analysis
IGNORED_FILES = {"Thumbs.db", ".DS_Store"}

# Default number of topics for LDA
DEFAULT_NUM_TOPICS = 5


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_text(file_path, ext):
    """
    Extract text based on file extension.
    """
    text = ''
    try:
        if ext == '.pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text() or ''
                text += page_text + '\n'

        elif ext == '.docx':
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + '\n'

        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

        elif ext in ('.eml', '.msg'):
            # Simple email body extraction
            import email
            msg = email.message_from_file(open(file_path, 'r', encoding='utf-8', errors='ignore'))
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        text += part.get_payload(decode=True).decode('utf-8', errors='ignore') + '\n'
            else:
                text = msg.get_payload(decode=True).decode('utf-8', errors='ignore')

        elif ext in ('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif') and OCR_AVAILABLE:
            # OCR on images
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)

    except Exception:
        # On any extraction error, return empty
        text = ''

    return text


def run_content_analysis(input_folder: str, output_folder: str, num_topics: int = DEFAULT_NUM_TOPICS) -> dict:
    """
    Extracts text, detects language, performs NER and topic modeling on breach files.

    Parameters:
        input_folder (str): Directory with breach files.
        output_folder (str): Directory to write outputs.
        num_topics (int): Number of LDA topics.

    Returns:
        dict: Summary of content analysis, with paths to detailed outputs.
    """
    ensure_dir(output_folder)

    texts = []
    paths = []
    records = []

    # 1. Text extraction and language detection
    for root, _, files in os.walk(input_folder):
        for fname in files:
            if fname in IGNORED_FILES:
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, input_folder)
            ext = os.path.splitext(fname)[1].lower()

            text = extract_text(full_path, ext)
            if not text.strip():
                continue  # skip files with no extractable text

            try:
                lang = detect(text)
            except Exception:
                lang = 'unknown'

            texts.append(text)
            paths.append(rel_path)
            records.append({'relative_path': rel_path, 'language': lang})

    # 2. Load spaCy model for NER (English)
    try:
        nlp = spacy.load('en_core_web_sm')
    except Exception:
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')

    # 3. Named Entity Recognition
    for rec, text in zip(records, texts):
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({'text': ent.text, 'label': ent.label_})
        rec['entities'] = entities

    # 4. Topic Modeling with LDA
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda_matrix = lda.fit_transform(tfidf)

    # Assign dominant topic to each document
    topic_assignments = lda_matrix.argmax(axis=1)
    for rec, topic in zip(records, topic_assignments):
        rec['topic'] = int(topic)

    # Collect topic keywords
    feature_names = vectorizer.get_feature_names_out()
    topics_keywords = {}
    for idx, comp in enumerate(lda.components_):
        top_indices = comp.argsort()[-10:][::-1]
        topics_keywords[idx] = [feature_names[i] for i in top_indices]

    # Write detailed records
    details_path = os.path.join(output_folder, 'content_analysis_records.json')
    with open(details_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)

    # Write summary
    # Language distribution
    lang_dist = {}
    for rec in records:
        lang_dist[rec['language']] = lang_dist.get(rec['language'], 0) + 1
    # Topic distribution
    topic_dist = {}
    for rec in records:
        topic_dist[rec['topic']] = topic_dist.get(rec['topic'], 0) + 1

    summary = {
        'total_text_files': len(records),
        'language_distribution': lang_dist,
        'topic_distribution': topic_dist,
        'topics_keywords': topics_keywords,
        'details_path': details_path
    }
    summary_path = os.path.join(output_folder, 'content_analysis_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary
