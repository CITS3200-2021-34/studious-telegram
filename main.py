from formatter.dataparser import DataParser
from embedder.sentence_embeddings import SentEmbeddings
from preprocessor.preprocess import PreProcessor
from gensim.models.doc2vec import Doc2Vec


def main():

    # Open text file and store in dictionary
    v = DataParser('help2002-2017.txt')
    threads = v.parse_text()
    p = PreProcessor()
    posts = p.get_posts(threads)
    questions, answers = p.create_data_structures(posts)

    # use dicitonary to get sentence embeddings per subject
    e = SentEmbeddings(questions, answers)
    #model, dict = e.doc2vec()

    # Load model and find similarities
    model = Doc2Vec.load("d2v.model")
    vect_q = e.vectorised_data(model)
    i = "Need help with lab 5"
    print()
    print(f'QUESTIONS: {i}')
    print()
    print("TOP SUGGESTED:")
    dict = e.get_similarity(model, i, vect_q)
    top_suggestions = e.get_top(dict, 3)
    print()
    temp = input(
        "Please Enter the suggested number (0 if no suggestion helps): ")
    suggestion = top_suggestions[int(temp)-1]
    print()
    e.get_question_answers(suggestion)
    print()


if __name__ == "__main__":
    main()
