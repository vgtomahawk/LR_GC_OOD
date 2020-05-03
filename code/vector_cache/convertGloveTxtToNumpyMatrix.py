from io import open
import sys

import numpy as np
import pickle


embedding_file_name = sys.argv[1]
pickle_file_name = sys.argv[1]+".p"
d = int(sys.argv[2])

embedding_file = open(embedding_file_name, "r" , encoding="utf-8")
vocab={}

allLines = embedding_file.readlines()

embed_matrix = np.zeros((len(allLines),d))

for index,line in enumerate(allLines):
    words = line.split()
    vocab[words[0]] = index
    words_float = [float(x) for x in words[1:]]
    embed_matrix[index,:] = np.asarray(words_float)

print("Vocabulary Size:",len(vocab))

embed_struct = (vocab,embed_matrix)

pickle.dump(embed_struct, open(pickle_file_name,"wb"))
