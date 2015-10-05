from polyglot_lib import *

def main():
    snippet = input("Welcome! Please input your code snippet!\n")
    return get_lang(snippet)

def get_lang(text):
    ans = random_tree.predict([text])
    print("Your language is probably {}.".format(ans))


main()
