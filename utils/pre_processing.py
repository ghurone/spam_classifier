import email
import os
import re
from html import unescape
from multiprocessing import Pool

import spacy
from bs4 import BeautifulSoup
from tqdm import tqdm


nlp = spacy.load('en_core_web_lg')

def parse_html(html_text):
    text = re.sub('<head.*?>.*?</head>', '', html_text, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' url_id ', text, flags=re.M | re.S | re.I)
    text = re.sub('<img\s.*?>', ' img_tag ', text, flags=re.M | re.S | re.I)
    text = re.sub('<iframe\s.*?>', ' iframe_tag ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    
    return unescape(text)


def get_text(email):
    s = ''
    for part in email.walk():
        tipo = part.get_content_type()
        
        try:
            content = part.get_content()
        except: 
            content = str(part.get_payload())
        
        content = content.lower()
        
        s += parse_html(content) + ' '

    return s.strip()


def get_tokens(text: str):
    tokens = []
    text = text.replace('\n', ' ').strip()
    for token in nlp(text):
        if token.orth_ in ("img_tag", "url_id", "iframe_tag"):
            tokens.append(token.orth_)
            continue 
            
        if token.orth_.isalnum() or token.like_url or token.like_num:
            palavra = token.lemma_

            if token.like_url:
                palavra = 'url_id'
            elif token.like_num:
                palavra = 'num_id'

            tokens.append(palavra.strip())

    return ' '.join(tokens).lower()


def email_text(df):
    print("Parsing emails...")
    email_parsed = df.apply(email.message_from_bytes)
    
    print("Getting emails' text...")
    email_text = email_parsed.apply(get_text)
    
    return email_text


def text_tokenizer(email_text):
    print("Cleaning emails' text...")
    with Pool(processes = os.cpu_count() - 1) as pool:
        email_tokens = pool.map(get_tokens, tqdm(email_text, total=len(email_text)))
    
    return email_tokens