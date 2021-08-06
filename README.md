# studious-telegram
Question and answer identifier for CITS2002

# Starting the app

Requires python 3.8+

```
python3 ./main.py
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

# Data

This app relies on previous QnA data. This data, although not highly sensitive, should not be publicly shared so that risk is minimised. To achieve this all data should be stored in `./data`. This directory is included in `.gitignore` to help prevent accidental upload.

# GitHub usage

Quick rundown of the suggested git workflow.

1. Clone the repo
```
git clone https://github.com/CITS3200-2021-34/studious-telegram
```
2. Create a short-lived branch locally to add a feature/fix a bug
```
git checkout -b epic-feature

git commit -m "Changed the code to do stuff"
git push
```
3. Create a pull request using the GitHub website. Add detail about what has been changed etc.
4. Tests will run, your code gets reviewed and hopefully then merged
5. Delete the branch and start again for the next feature/bug
```
git branch -d epic-feature
```
