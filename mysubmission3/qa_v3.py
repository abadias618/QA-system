import sys
from parse_v1 import *
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')

from fuzzywuzzy import fuzz

def get_only_nouns_in_string(tuples_tags):
    only_nouns = ""
    for t in tuples_tags:
        if t[1] == "NOUN":
            only_nouns += t[0] + " "
    return only_nouns.strip()

def get_only_verbs_in_string(tuples_tags):
    only_verbs = ""
    for t in tuples_tags:
        if t[1] == "VERB":
            only_verbs += t[0] + " "
    return only_verbs.strip()

def get_only_verbs_and_nouns_in_string(tuples_tags):
    only_only_verbs_and_nouns = ""
    for t in tuples_tags:
        if t[1] == "VERB" or t[1] == "NOUN":
            only_only_verbs_and_nouns += t[0] + " "
    return only_only_verbs_and_nouns.strip()

def add_synonyms(q):
    words = q
    final =""
    for w in words:
        synset = wn.synsets(w)
        if len(synset) < 1:
            final += w + " "
            continue
        synsets = wn.synset(synset[0].name()).hyponyms()
        for s in synsets:
            synonym = s.name()[:len(s.name())-5].replace("_"," ")
            final += w + " " + synonym + " "
    return final.strip()

def make_window(story, k):
    divided = []
    splitted = story.split()
    for i, w in enumerate(splitted):
        j = i + k
        if i+k > len(splitted):
            continue
        divided.append(" ".join(splitted[i:j]))
    return divided

if __name__ == "__main__":

    path, question_IDs = read_input_file(sys.argv[1])
    stop_wds = set(stopwords.words("english"))
    punctuation = [".","?",",",":",";","(",")","-","$","!"]

    for file in question_IDs:
        story = parse_story_file(path, file)
        questions = parse_questions_file(path, file)
        tokenized_story = sent_tokenize(story["text"])
        #print("STORY:\n",story)
        #lowercase_story = [s.lower() for s in tokenized_story]

        # POS-TAG and remove stop words and punctuation.
        tagged_story = []
        for s in tokenized_story:
            #s = [w for w in s.split() if w not in stop_wds and w not in punctuation]
            tagged_story.append(pos_tag(s.split(), tagset="universal"))

        # For each question tokenize question and remove stop words and punctuation
        for key in questions.keys():
            tokenized_q = word_tokenize(questions[key][1])
            #tokenized_q = [w for w in tokenized_q if w not in stop_wds and w not in punctuation]
            #lowercase_q = [w.lower() for w in tokenized_q]
            universal_tagged_question = pos_tag(tokenized_q, tagset="universal")
            pos_tagged_question = pos_tag(tokenized_q)
            # Compare similarity of each sentence in the story with the question with Levenshtein Distance
            fuzzy_noun_scores = []
            for sentence in tagged_story:
                score = fuzz.WRatio(get_only_nouns_in_string(universal_tagged_question), get_only_nouns_in_string(sentence))
                fuzzy_noun_scores.append(score)

            nouns_max_score = max(fuzzy_noun_scores)
            noun_possibles = []
            for i in range(len(fuzzy_noun_scores)):
                if fuzzy_noun_scores[i] == nouns_max_score:
                    noun_possibles.append(tokenized_story[i])
            #print("scores",*fuzzy_noun_scores)
            #print("Possibles N:\n",noun_possibles)
            #print("Answer: " + tokenized_story[fuzzy_noun_scores.index(nouns_max_score)] + "\n")
            #print("q NOUNS",get_only_nouns_in_string(universal_tagged_question))

            #print("\n\n\n")

            fuzzy_verb_scores = []
            q = get_only_verbs_in_string(universal_tagged_question)
            q = [w for w in q.split() if w not in stop_wds]
            q = add_synonyms(q)
            for sentence in tagged_story:
                if len(q) > 0:
                    score = fuzz.ratio(q, get_only_verbs_in_string(sentence))
                else:
                    score = 0
                fuzzy_verb_scores.append(score)

            verbs_max_score = max(fuzzy_verb_scores)
            verb_possibles = []
            for i in range(len(fuzzy_verb_scores)):
                if fuzzy_verb_scores[i] == verbs_max_score:
                    verb_possibles.append(tokenized_story[i])
            #print("scores",*fuzzy_verb_scores)
            #print("Possibles V:\n",verb_possibles)
            #print("Answer: " + tokenized_story[fuzzy_verb_scores.index(verbs_max_score)] + "\n")
            #print("q VERBS",get_only_verbs_in_string(universal_tagged_question))
            
            #print("\n\n\n")

            fuzzy_noun_window_scores = []
            n = get_only_nouns_in_string(universal_tagged_question)
            window_story = make_window(story["text"], len(n.split()))
            #print("window_story",window_story)
            for sentence in window_story:
                if len(q) > 0:
                    score = fuzz.WRatio(n, sentence)
                else:
                    score = 0
                fuzzy_noun_window_scores.append(score)
            noun_window_max_score = max(fuzzy_noun_window_scores)
            located_s = 0
            #print("xXX",window_story[fuzzy_noun_window_scores.index(noun_window_max_score)])
            for s in tokenized_story:
                if window_story[fuzzy_noun_window_scores.index(noun_window_max_score)] in s:
                    located_s = tokenized_story.index(s)
            fuzzy_noun_sentence_window_scores = [0 for s in tokenized_story]
            fuzzy_noun_sentence_window_scores[located_s] = noun_window_max_score
            #print("scores",*fuzzy_noun_window_scores)
            #print("Answer: " + tokenized_story[located_s] + "\n")
            #print("q VERBS",get_only_verbs_in_string(universal_tagged_question))


            # CALCULATIONS
            final = []
            for i in range(len(tokenized_story)):
                final.append((fuzzy_noun_scores[i] * 0.50) + (fuzzy_verb_scores[i] * 0.25) + (fuzzy_noun_sentence_window_scores[i] * 0.25))

            id = questions[key][0]
            answer = tokenized_story[final.index(max(final))]
            answer = pos_tag(answer.split(), tagset="universal")
            answer = get_only_nouns_in_string(answer)
            #print("Question",questions[key][1])#delete
            print("QuestionID: " + id)
            print("Answer: " + answer + "\n")
            #print("Answer: " + tokenized_story[fuzzy_scores.index(max_score)] + "\n")
            #print("pos q", pos_tagged_question)
            #print("universal q", universal_tagged_question)
            
            
            
            
