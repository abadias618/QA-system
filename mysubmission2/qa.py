import sys
from parse_v1 import *
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

from fuzzywuzzy import fuzz

def get_only_nouns_in_string(tuples_tags):
    only_nouns = ""
    for t in tuples_tags:
        if t[1] == "NOUN":
            only_nouns += t[0] + " "
    return only_nouns.strip()

if __name__ == "__main__":
    path, question_IDs = read_input_file(sys.argv[1])
    stop_wds = set(stopwords.words("english"))
    punctuation = [".","?",",",":",";","(",")","-","$","!"]
    for file in question_IDs:
        story = parse_story_file(path, file)
        questions = parse_questions_file(path, file)
        tokenized_story = sent_tokenize(story["text"])

        #lowercase_story = [s.lower() for s in tokenized_story]

        tagged_story = []
        for s in tokenized_story:
            s = [w for w in s.split() if w not in stop_wds and w not in punctuation]
            tagged_story.append(pos_tag(s, tagset="universal"))

        #print(tagged_story)
        for key in questions.keys():
            tokenized_q = word_tokenize(questions[key][1])
            tokenized_q = [w for w in tokenized_q if w not in stop_wds and w not in punctuation]
            #lowercase_q = [w.lower() for w in tokenized_q]
            tagged_question = pos_tag(tokenized_q, tagset="universal")
            
            fuzzy_scores = []
            for sentence in tagged_story:
                score = fuzz.token_sort_ratio(get_only_nouns_in_string(tagged_question), get_only_nouns_in_string(sentence))
                fuzzy_scores.append(score)
            max_score = max(fuzzy_scores)
            id = questions[key][0]
            print("QuestionID: " + id)
            print("Answer: " + tokenized_story[fuzzy_scores.index(max_score)] + "\n")
            
