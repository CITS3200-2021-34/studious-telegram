FROM tensorflow/tensorflow:2.6.0

WORKDIR /app

COPY ./qna/requirements.txt ./

# Install python packages
RUN pip install -r requirements.txt

# Add unviversal encoder model
RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')"
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"


# Pull in source code
COPY ./qna/src ./

EXPOSE 8080

CMD ["python", "-u", "main.py"]