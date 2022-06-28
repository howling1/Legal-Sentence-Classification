import sys
import pickle
import spacy
from luima_sbd.sbd_utils import text2sentences
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
import re
import numpy as np

# read the string content of the specified file
def read_file(filepath):
	with open(filepath,'rb') as f:
		file_str = str(f.read(),'latin1')

	return file_str

# load the best classifier: TFIDF + SVM
def load_classifier():
	file = open("./TFIDF_classifier.pickle", "rb")
	classifier = pickle.load(file)
	file.close()

	return classifier

# Load the TFIDF vectorizer model
def load_TFIDF_model():
	file = open("./TFIDF.pickle", "rb")
	classifier = pickle.load(file)
	file.close()

	return classifier

if __name__ == '__main__':
	filepath = sys.argv[1]
	file_str = read_file(filepath)
	classifier = load_classifier()
	TFIDF_model = load_TFIDF_model()
	segmented_sentences = text2sentences(file_str)
	result = []

	TFIDF_feature_vectors = TFIDF_model.transform(segmented_sentences).toarray()
	pred_types = list(classifier.predict(TFIDF_feature_vectors))

	for i in range(len(TFIDF_feature_vectors)):
		result.append({
			'sentence': segmented_sentences[i],
			'type': pred_types[i]
			})

	print(result)





	

	







	




