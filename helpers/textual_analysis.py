from collections import Counter


def get_most_common_words(li):
    l = " ".join(li) #put into one big string
    l = l.split() #split into words
    #count words
    l = Counter(l)
    return l


def create_word_cloud(l):
    # Generate a word cloud image
    l = " ".join(l)
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    wordcloud1 = WordCloud(width=1600, height=800).generate(l)
    import matplotlib.pyplot as plt

    # Display the generated image:
    plt.imshow(wordcloud1, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()