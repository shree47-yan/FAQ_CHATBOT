import tkinter as tk
from tkinter import scrolledtext
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data (first run only)
nltk.download('punkt')
nltk.download('stopwords')

# ==========================
# FAQ DATABASE
# ==========================

faq_data = {
    "What is Python?": "Python is a high-level programming language.",
    "Who developed Python?": "Python was developed by Guido van Rossum.",
    "What is AI?": "Artificial Intelligence enables computers to perform tasks that normally require human intelligence.",
    "What is Machine Learning?": "Machine Learning is a subset of Artificial Intelligence that allows computers to learn from data.",
    "What is NLP?": "Natural Language Processing helps computers understand and process human language.",
    "What is Tkinter?": "Tkinter is Python's built-in library for creating graphical user interfaces (GUI).",
    "What is a variable?": "A variable is used to store data in a program.",
    "What is a function?": "A function is a reusable block of code that performs a specific task.",
    "What is a list in Python?": "A list is an ordered collection of items.",
    "What is the use of Python?": "Python is used in web development, AI, machine learning, automation, data science, and software development."
}

# ==========================
# TEXT PREPROCESSING
# ==========================

stop_words = set(stopwords.words("english"))

def preprocess(text):
    words = word_tokenize(text.lower())

    filtered_words = []

    for word in words:
        if word.isalnum() and word not in stop_words:
            filtered_words.append(word)

    return " ".join(filtered_words)

# Prepare FAQ Questions
questions = list(faq_data.keys())

processed_questions = []

for question in questions:
    processed_questions.append(preprocess(question))

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(processed_questions)

# ==========================
# CHATBOT FUNCTION
# ==========================

def chatbot():

    user_question = entry.get().strip()

    if user_question == "":
        return

    chat_box.insert(tk.END, "You : " + user_question + "\n")

    processed_user = preprocess(user_question)

    user_vector = vectorizer.transform([processed_user])

    similarity = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarity.argmax()

    similarity_score = similarity[0][best_match_index]

    if similarity_score > 0.30:
        answer = faq_data[questions[best_match_index]]
    else:
        answer = "Sorry, I don't understand your question."

    chat_box.insert(tk.END, "Bot : " + answer + "\n\n")

    entry.delete(0, tk.END)

# ==========================
# GUI
# ==========================

root = tk.Tk()

root.title("FAQ Chatbot using NLP")

root.geometry("700x550")

title = tk.Label(
    root,
    text="FAQ Chatbot",
    font=("Arial", 20, "bold")
)

title.pack(pady=10)

chat_box = scrolledtext.ScrolledText(
    root,
    width=80,
    height=22,
    font=("Arial", 10)
)

chat_box.pack(pady=10)

entry = tk.Entry(
    root,
    width=60,
    font=("Arial", 12)
)

entry.pack(side=tk.LEFT, padx=10, pady=10)

send_button = tk.Button(
    root,
    text="Send",
    command=chatbot,
    bg="green",
    fg="white",
    font=("Arial", 12, "bold")
)

send_button.pack(side=tk.LEFT)

root.mainloop()