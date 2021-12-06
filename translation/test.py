from sacrebleu.metrics import BLEU

if __name__ == '__main__':
    bleu = BLEU()
    refs = [
        ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.']
]
    sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

    result = bleu.corpus_score(sys, refs)

    print(result)

    print(result.score)

