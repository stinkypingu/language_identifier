import pandas as pd
import numpy as np
import time as time

def process_data(language_tsv):
    sentences = pd.read_csv(language_tsv, delimiter='\t')    

    train_data = sentences.iloc[:10000, 2]
    train_text = " ".join(train_data.tolist()).lower()

    train_chars = set(train_text)
    return train_text, train_chars

eng_text, eng_chars = process_data("eng_sentences.tsv")
spa_text, spa_chars = process_data("spa_sentences.tsv")

all_chars = eng_chars | spa_chars

dictionary = {char: np.random.choice([1, -1], 10000) for char in all_chars}


def get_AM(text):
    AM = np.zeros(10000)
    for i in range(len(text) - 2):
        char0 = dictionary[text[i]]
        char1 = dictionary[text[i + 1]]
        char2 = dictionary[text[i + 2]]

        char1 = np.roll(char1, 1)
        char2 = np.roll(char2, 2)

        trigram = char0 * char1 * char2

        AM += trigram
    return AM


start = time.time()
eng_AM = get_AM(eng_text)
spa_AM = get_AM(spa_text)
end = time.time()
print(f"processing time: {end - start :.2f}s")


def cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

while True:
    user_text = input().lower()

    user_AM = get_AM(user_text)

    if cos_sim(user_AM, eng_AM) > cos_sim(user_AM, spa_AM):
        print('english')
    else:
        print('spanish')

    
    
