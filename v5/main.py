import pandas as pd
import string

def main():
    data = pd.read_csv("train.tsv", sep='\t')
    sentiment = data.sentiment
    text = data.text
    strings = []
    bag = {}
    for s in text:
        tmp = s.replace('\n0', '')
        tmp = tmp.replace('\n', '')
        tmp = tmp.replace('\t', '')
        strings.append(tmp.translate(str.maketrans('', '', string.punctuation)))


    for s in strings:
        for word in s.split(' '):
            if word in bag:
                bag[word] += 1
            else:
                bag[word] = 0
                
    # aaaaa

if __name__ == "__main__":
    main()
