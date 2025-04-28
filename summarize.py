import logging
import nltk
import heapq
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import word2vec
from rouge_score import rouge_scorer


class Preprocess:
    def __init__(self):
        pass

    def to_lower(self, text):
        return text.lower()

    def tokenize_sentence(self, text):
        sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        sentences = sent_tokenizer.tokenize(text)
        return sentences

    def preprocess_sentence(self, sentences):
        word_tokenizer = nltk.RegexpTokenizer(r"\w+")
        processed_sentences = []
        special_characters = re.compile("[^A-Za-z0-9 ]")
        for sentence in sentences:
            sentence = re.sub(special_characters, " ", sentence)
            words = word_tokenizer.tokenize(sentence)
            words = self.remove_stopwords(words)
            words = self.wordnet_lemmatize(words)
            processed_sentences.append(words)
        return processed_sentences

    def remove_stopwords(self, sentence):
        stop_words = stopwords.words('english')
        tokens = [token for token in sentence if token not in stop_words]
        return tokens

    def wordnet_lemmatize(self, sentence):
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token, pos='v') for token in sentence]
        return tokens

    def complete_preprocess(self, text):
        text_lower = self.to_lower(text)
        sentences = self.tokenize_sentence(text_lower)
        preprocessed_sentences = self.preprocess_sentence(sentences)
        return preprocessed_sentences

    def calculate_length(self, df):
        df["article_len"] = df["article"].apply(lambda x: len(x.split()))
        df["highlights_len"] = df["highlights"].apply(lambda x: len(x.split()))
        return df

    def most_similar_words(self, model, words):
        for word in words:
            print("Most similar to ", word, ": ", model.wv.most_similar(word))

    def word2vec_model(self, sentences, num_feature, min_word_count, window_size, down_sampling, sg):
        num_thread = 5
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = word2vec.Word2Vec(sentences,
                                  vector_size=num_feature,
                                  min_count=min_word_count,
                                  window=window_size,
                                  sample=down_sampling,
                                  workers=num_thread,
                                  sg=sg,
                                  epochs=20)
        return model

    def top_10_frequent_words(self, model):
        model.sorted_vocab
        top_words = model.wv.index_to_key[:10]
        return top_words


class NewsSummarization:
    def __init__(self):
        pass

    def extractive_summary(self, text, sentence_len=8, num_sentences=3):
        word_frequencies = {}
        preprocessor = Preprocess()
        tokenized_article = preprocessor.complete_preprocess(text)

        for sentence in tokenized_article:
            for word in sentence:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        maximum_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / maximum_frequency

        sentence_scores = {}
        sentence_list = nltk.sent_tokenize(text)

        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) > sentence_len:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]

        summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)
        return summary

    def get_rouge_score(self, actual_summary, generated_summary):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(actual_summary, generated_summary)
        return scores

    def evaluate_extractive(self, dataset, metric):
        summaries = [self.extractive_summary(text) for text in dataset["article"]]
        score = metric.compute(predictions=summaries, references=dataset["highlights"])
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = dict((rn, round(score[rn].mid.fmeasure * 100, 2)) for rn in rouge_names)
        return rouge_dict

    def evaluate_abstractive(self, dataset, metric, summarizer):
        summaries = [summarizer(text, max_length=120, min_length=80, do_sample=False)[0]['summary_text'] for text in
                     dataset["article"]]
        score = metric.compute(predictions=summaries, references=dataset["highlights"])
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = dict((rn, round(score[rn].mid.fmeasure * 100, 2)) for rn in rouge_names)
        return rouge_dict
