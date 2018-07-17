import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from utils import *

def cos_sim(vec1, vec2):
    return np.array(vec1)@np.array(vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
#        print(tag_name)
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name[0] + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim) #### YOUR CODE HERE ####
#        tag_w2v = unpickle_file(os.path.join(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], os.path.normpath('%s.pkl' % tag_name))
#        flag = 0
        for i in range(len(thread_ids)):
            if i == 0:
                mx_sim = cos_sim(question_vec, thread_embeddings[0])
                continue
            if cos_sim(question_vec, thread_embeddings[i]) > mx_sim:
                best_thread = i  #### YOUR CODE HERE ####
        
        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals 
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param
        
        ########################
        #### YOUR CODE HERE ####
        ########################
        
        chitchat_bot = ChatBot("GalaxyDream_bot", trainer='chatterbot.trainers.ChatterBotCorpusTrainer')
        chitchat_bot.train("chatterbot.corpus.english")
        return chitchat_bot

    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        self.chitchat_bot = self.create_chitchat_bot()
        
        prepared_question = text_prepare(question) #### YOUR CODE HERE ####
#        print('hello')
#        print(prepared_question)
        features = unpickle_file(RESOURCE_PATH['TFIDF_VECTORIZER']).transform([prepared_question]) #### YOUR CODE HERE ####
        intent = unpickle_file(RESOURCE_PATH['INTENT_RECOGNIZER']).predict(features) #### YOUR CODE HERE ####

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chitchat_bot.get_response(prepared_question) #### YOUR CODE HERE ####
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = unpickle_file(RESOURCE_PATH['TAG_CLASSIFIER']).predict(features) #### YOUR CODE HERE ####
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = ThreadRanker(paths = RESOURCE_PATH).get_best_thread(prepared_question, tag) #### YOUR CODE HERE ####
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)

