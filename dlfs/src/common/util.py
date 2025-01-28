import numpy as np


def preprocess(text:str):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    """말뭉치를 받아 동시발생 행렬을 리턴한다.
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    
    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    """두 벡터를 받아 코사인 유사도를 계산한다.
    벡터의 크기는 L2노름을 사용하여 계산하고
    내적을 사용해 두 벡터의 각도를 사용하여 유사도를 계산.
    """
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        raise ValueError(f"{query} not found")

    print(f"query: {query}")
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)

    # 모든 단어와 query단어의 유사도를 얻는다.
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1*similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f"  {id_to_word[i]}: {similarity[i]}")
        count += 1
        if count >= top:
            return


def ppmi(C, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32) # 결과 값
    N = np.sum(C) # 동시발생행렬의 모든 셀 값을 합산 ex) 14
    S = np.sum(C, axis=0) # 동시발생행렬의 행별로 값을 모두 합산 ex) [1 4 2 2 2 2 1]
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            # C[i,j]: 동시발생 횟수
            # S[j], S[i]: 개별발생 횟수
            pmi = np.log2(C[i, j]*N / (S[j]*S[i])+eps)
            M[i, j] = max(0, pmi) # PPMI
    return M
