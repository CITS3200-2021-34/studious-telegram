FROM tensorflow/tensorflow

WORKDIR /app

RUN pip install tensorflow_hub scipy sentence_transformers gensim questionary

RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')"

COPY ./qna/src ./

EXPOSE 8080

CMD ["python", "-u", "main.py"]