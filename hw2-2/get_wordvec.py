import logging

from gensim.models import word2vec

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence("./sel_conversation/question.txt")
    model = word2vec.Word2Vec(sentences, size=256)

    #保存模型，供日後使用
    model.save("word2vec.model")

    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model_name")

def load_model():
	model = word2vec.Word2Vec.load("word2vec.model")
	#print (model.wv['汽車'])
	#print (model.wv['火車'])
	#print (model.wv['我'])
	word1 = "艾姬"
	word2 = "躲貓貓"
	print (word1, word2)
	print ("similarity: ", model.similarity(word1, word2))
	#model.n_similarity('我','汽車')

if __name__ == "__main__":
    load_model()
