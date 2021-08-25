# QNA Module

This is the 'guts' of the project. This does the natural language processing (NLP) to match the current question to previously asked questions.

# Starting the app

Requires python 3.8+

```
python3 ./src/main.py
```

# Structure

The project has been built with interfaces for the major components to allow for a greater level of modularity.

`main.py` builds the question matcher and the user interface, links them, and then triggers the UI's main input loop.

### `AbstractQuestionMatcher`

This defines the question matcher, the core of the application.

Concrete question matcher implementations must be a subclass of this abstract matcher and implement all abstract methods. Question matchers are consumed by User Interfaces.

### `AbstractUserInterface`

This defines the user interface, the component that allows the user to interact with the question matcher.

Concrete user interface implementations must be a subclass and implement all abstract methods.

# Requirements

## Python modules

* `gensim`
* `nltk`
* `sklearn`
* `sentence_transformers`
* `tensorflow`
* `tensorflow_hub`
* `scipy`
* `numpy`
* `pandas`

# Data

This app relies on previous QnA data. This data, although not highly sensitive, should not be publicly shared so that risk is minimised. To achieve this all data should be stored in `./data`. This directory is included in `.gitignore` to help prevent accidental upload.