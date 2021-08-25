FROM tensorflow/tensorflow

WORKDIR /app

RUN pip install tensorflow_hub scipy sentence_transformers gensim

COPY ./qna/src ./

CMD ["python", "main.py"]