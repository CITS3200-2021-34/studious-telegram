# studious-telegram
Question and answer identifier for CITS2002

# Structure

Currently the project is made up of a single module, `qna`. The `qna` module is the guts of the project performing natural language processing to a given question to previously asked questions.

# Running the project

The project can be run in two ways. 

### Locally like any other python project. 

See instructions in the [`qna/README.md`](qna/README.md)

### In Docker

With docker installed ([Get Docker](https://docs.docker.com/get-docker/)) run the follow commands in order. These commands might be different under windows.

```
docker build --file ./docker/qna.dockerfile -t qna .
docker run -it qna
```

Alternatively for UNIX based systems you can use the start script.

```
./start.sh
```

You might have to change the script permissions `chmod +x ./start.sh`



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
