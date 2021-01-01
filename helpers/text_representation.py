from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import word2vec
import nltk


class TextRepresentation:
    def vectorize(corpus, type, params={}):
        vectorizer = get_vectorizer(type)
        return vectorizer(corpus, params)

def get_vectorizer(format):
    if format == 'BOW':
        return _vectorize_BOW
    elif format == 'B-NGRAM':
        return _vectorize_Bag_NGram
    elif format == 'TFIDF':
        return _vectorize_tfidf
    elif format == 'W2V':
        return _vectorize_w2v
    else:
        raise ValueError(format)

def _vectorize_BOW(corpus, params={}):
    vectorizer = CountVectorizer()
    vectorizer.fit(corpus)
    return vectorizer.transform(corpus)

def _vectorize_Bag_NGram(corpus, params={}):
    bv = CountVectorizer(ngram_range=(2, 2))
    return bv.fit_transform(corpus)
    #GET MORE INFO ABOUT THE MODEL
    # bv_matrix = bv_matrix.toarray()
    # # vocab = bv.get_feature_names()
    # # pd.DataFrame(bv_matrix, columns=vocab)

def _vectorize_tfidf(corpus, params={}):
    tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2',
                         use_idf=True, smooth_idf=True)
    return tv.fit_transform(corpus)
    # tt_matrix = tt_matrix.toarray()
    # vocab = cv.get_feature_names()
    # pd.DataFrame(np.round(tt_matrix, 2), columns=vocab)

def _vectorize_w2v(corpus, params={}):
    # tokenize sentences in corpus
    wpt = nltk.WordPunctTokenizer()
    tokenized_corpus = wpt.tokenize(corpus)
    # Set values for various parameters
    feature_size = 100  # Word vector dimensionality
    window_context = 30  # Context window size
    min_word_count = 1  # Minimum word count
    sample = 1e-3  # Downsample setting for frequent words
    w2v_model = word2vec.Word2Vec(tokenized_corpus, size=feature_size,
                                  window=window_context, min_count=min_word_count,
                                  sample=sample, iter=50)

def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,),dtype="float64")
    nwords = 0.
    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector
def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary,
num_features) for tokenized_sentence in corpus]
    return np.array(features)