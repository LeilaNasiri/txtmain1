import stanza
from tqdm import tqdm
from stanza.models.common.doc import Word, Document, Sentence

empty_nlp = None
# ============== initial stanza NLP model ===================================
def initilize_nlp_model():
    global empty_nlp
    nlp = stanza.Pipeline('fa', processors='tokenize,pos,mwt,lemma,depparse', use_gpu=True,
                          pos_batch_size=3000)  # Build the pipeline, specify part-of-speech processor's batch size
    empty_nlp = nlp
    return nlp

# ============== Create Dataset ===================================
def get_dataset(nlp, path='.'):
    import os
    if nlp==None:
        nlp = initilize_nlp_model()
    try:
        print('reading dataset files (Start)')
        dataset = []
        for root, directories, files in os.walk(path, topdown=False):
            for name in tqdm(files):
                fname = os.path.join(root, name)

                file = open(fname, mode='r', encoding='utf-8')
                text = file.read().replace('\n','')
                file.close()

                dataset.append(nlp(text))
        print('reading dataset files (Done)')
    except:
        raise Exception("get_dataset: reading error.")

    return nlp, dataset


# ============== Create Dataset from Excel ===================================
def get_dataset_from_excel(nlp, path='.', fname='', colname=''):
    import os
    import pandas as pd
    if nlp == None:
        nlp = initilize_nlp_model()
    try:
        
        print('reading dataset files (Start)')

        df = pd.read_excel(path + fname)

        dataset = []
        i=1
        for t in tqdm(df[colname]):
            # if i>10:
            #     break
            dataset.append(nlp(t))
            i=i+1

        print('reading dataset files (Done)')
    except:
        raise Exception("get_dataset: reading error.")

    return nlp, dataset

# ============= Preprocess: Remove Stop Words ====================================
def prep_remove_stopwords(text, stopword_filename='stop.txt'):
    import io
    stoplist = []
    try:
        with io.open(stopword_filename, encoding='utf-8') as stop_file:
            for line in stop_file:
                stoplist.append(line.replace('\n', ''))
    except:
        raise Exception("Stop List reading error.")

    new_text = ""
    words = text.split(' ')
    for word in words:
        # print('word: %s' % word)

        if word not in stoplist:
            # print('printed word: %s' % word)
            new_text = new_text + " " + word
    print('Preprocess: Remove Stop Words (done)')
    return new_text

# ============= Preprocess: Remove Punctuations ====================================
def prep_remove_punctuation(text):
    symbols = "،؛,][«ـ»!ًٌٍَُِّ\"#$%&()*+-./:;<=>?@[\]^_`{|}~"
    symbols = symbols + 'abcdefghijklmnopqrstwxyz'
    for i in symbols:
        text = text.lower().replace('\n','').replace(i, ' ')
    text = text.replace("'", " ")
    text = text.replace('"', " ")
    print('Preprocess: Remove Punctuations (done)')
    return text

# ============= Preprocess: Remove Single Characters ====================================
def prep_remove_single_characters(text):
    words = text.split(" ")
    new_text = ""
    for w in words:
        if len(w.strip()) > 1:
            new_text = new_text + " " + w
    print('Preprocess: Remove Single Characters (done)')
    return new_text

# ============= Preprocess: Lemmatisation ====================================
def prep_lemmatisation(doc):
    text = [word.lemma for sent in doc.sentences for word in sent.words]
    print('Preprocess: Lemmatisation (done)')
    return ' '.join(text)

# ============= Preprocess: Converting Numbers ====================================
def prep_converting_numbers(text):
    # text = text.split(' ')
    # i = 0
    # while True:
        # if i>=len(text): break
        # if text[i].isdigit()==True:
        #     text.remove(text[i])
        # i=i+1
    for i in range(0,10):
        text = text.replace(str(i),'')
    return text

# ============= Preprocess ====================================
def preprocess(text):
    text = prep_remove_punctuation(text)
    text = prep_remove_single_characters(text)
    text = prep_converting_numbers(text)
    text = prep_remove_stopwords(text, stopword_filename='stop.txt')
    if empty_nlp != None:
        nlp = empty_nlp
    else:
        nlp = initilize_nlp_model()
    text = prep_lemmatisation(nlp(text))
    text = prep_remove_punctuation(text)
    text = prep_remove_stopwords(text, stopword_filename='stop.txt')
    text = prep_remove_punctuation(text)
    text = prep_remove_single_characters(text)
    return text

# ============= TFIDF calculator ====================================
def TFIDF(dataset, nKeywords=5):
    import math
    nKeywords = max(nKeywords, 1)
    all_doc_words = []
    each_doc_words = []
    tf_freq=[]
    idf_freq= {}
    tfidf_rate=[]
    for ds in dataset:
        each_doc_words.append(list(filter(None, ds.text.split(' '))))

        all_doc_words.extend(each_doc_words[-1])

        # calculate TF of each unique word
        tf_freq.append({})
        tf_unique_words = list(set(each_doc_words[-1]))

        for u in tf_unique_words: # u: string
            tf_freq[-1][u] = each_doc_words[-1].count(u)/len(each_doc_words[-1])  # tf(t,d) = count of t in d / number of words in d

    # calculate IDF of each unique word
    idf_unique_words = list(set(all_doc_words))
    for u in idf_unique_words:  # u: string
        idf_freq[u] = math.log(len(all_doc_words) / (all_doc_words.count(u)+1))   # idf(t) = log(N / (df + 1))

    for i in range(len(each_doc_words)):
        tfidf_rate.append({})
        for w in each_doc_words[i]:
            tfidf_rate[-1][w]=(tf_freq[i][w]*idf_freq[w])

    keywords=[]
    for doc in tfidf_rate:
        keywords.append({})
        sorteddict = dict(sorted(doc.items(), key=lambda item: item[1]))
        for x in list(sorteddict)[0:nKeywords]:
            keywords[-1][x]=sorteddict[x]

    return keywords


try:
  Document.add_property('word_count', default=0, getter=lambda self: len(self.text.split(" ")), setter=None)
  Sentence.add_property('word_count', default=0, getter=lambda self: len(self.text.split(" ")), setter=None)
  Word.add_property('char_count', default=0, getter=lambda self: len(self.text), setter=None)
  Document.add_property('preprocess', default=0, getter=lambda self: preprocess(self.text), setter=None)
except:
  print('Warning: Attributes was added before.')
