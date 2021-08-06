from domain.keyword import KeywordMatcher
from interface.basiccli import BasicCLI

def main():
    questionMatcher = KeywordMatcher()
    cli = BasicCLI()

    cli.setQuestionMatcher(questionMatcher)

    cli.start()


if __name__ == "__main__":
    main()