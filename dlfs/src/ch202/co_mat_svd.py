import sys
sys.path.append("d:/ai/dlfs/src")
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi
np.set_printoptions(suppress=True, precision=10)
np.set_printoptions(linewidth=160)

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)

C = create_co_matrix(corpus, vocab_size)
print("C:\n", C)

W = ppmi(C)
print("W:\n", W)

U, S, V = np.linalg.svd(W)
print("U:\n", U)
print("S:\n", S)
print("V:\n", V)

print("U[0]:\n", U[0])
print("U[0, :2]:\n", U[0, :2])

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id,0], U[word_id,1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()
