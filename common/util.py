import re
import numpy as np

def preprocess(text):
    text = text.lower()
    words = re.split('(\W+)?',text)
    
    word_to_id = {}
    id_to_word = {}
    
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size = 1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype = np.int32)
    
    for idx, word_id in enumerate(corpus):
        for i in range(1,window_size+1):
            left_idx = idx - i
            right_idx = idx + i
            
            if left_idx >= 0 :
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
                
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
                
    return co_matrix


def cos_similarity(x,y,eps=1e-8): #인수로 제로 벡터 들어올 경우 에러 방지를 위해 eps같이 작은 값 필요(일반적으로 다른 값에 흡수, 0일 경우 eps 해당 값 유지)
    nx = x / np.sqrt(np.sum(x**2)+eps) #정규화
    ny = y / np.sqrt(np.sum(y**2)+eps) #정규화
    return np.dot(nx,ny) #내적