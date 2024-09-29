import streamlit as st
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# Preprocessing functions
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  # remove hashtags
    text = re.sub(r'RT[\s]', '', text)  # remove RT
    text = re.sub(r"http\S+", '', text)  # remove links
    text = re.sub(r'[0-9]+', '', text)  # remove numbers
    text = text.replace('\n', ' ')  # replace new lines with spaces
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = text.strip(' ')  # remove spaces from both ends
    return text

def casefoldingText(text):
    return text.lower()

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(text):
    listStopwords = set(stopwords.words('indonesian'))
    filtered = [word for word in text if word not in listStopwords]
    return filtered

def stemmingText(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(word) for word in text]

def toSentence(list_words):
    return ' '.join(list_words)

# Streamlit App
st.title('Sentiment Analysis Preprocessing')

# Input text
input_text = st.text_area("Masukkan teks untuk diproses:", "Contoh teks yang akan dianalisis...")

if st.button("Proses Teks"):
    st.write("### Hasil Preprocessing:")

    # Apply the preprocessing functions
    cleaned_text = cleaningText(input_text)
    st.write(f"**1. Cleaning Text:**\n{cleaned_text}")
    
    casefolded_text = casefoldingText(cleaned_text)
    st.write(f"**2. Casefolding (Lowercase):**\n{casefolded_text}")
    
    tokenized_text = tokenizingText(casefolded_text)
    st.write(f"**3. Tokenizing (Tokenization):**\n{tokenized_text}")
    
    filtered_text = filteringText(tokenized_text)
    st.write(f"**4. Filtering (Stopword Removal):**\n{filtered_text}")
    
    stemmed_text = stemmingText(filtered_text)
    st.write(f"**5. Stemming (Root Words):**\n{stemmed_text}")
    
    final_sentence = toSentence(stemmed_text)
    st.write(f"**6. Final Sentence:**\n{final_sentence}")

# Download necessary data for tokenization
nltk.download('punkt')  # Required for word_tokenize
nltk.download('stopwords')  # If you're using stopwords
