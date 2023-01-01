#!/usr/bin/env python3
import sys
import os
import json
from nltk import word_tokenize

def read_input_file(input_file_name):
    path = ''
    questions = []
    with open(input_file_name) as file:
        for i, line in enumerate(file):
            if i == 0:
                path = str(line.strip())
                continue
            
            questions.append(line.strip())

    return path, questions

def parse_story_file(path, file_name):
    f = os.path.join(path, str(file_name + '.story'))
    f = os.path.relpath(f)
    headline = ""
    date = ""
    story_id = ""
    text = []
    text_switch = False
    
    with open(f) as file:
        for line in file:
            if not text_switch:
                if "HEADLINE: " in line:
                    headline = line.strip().replace("HEADLINE: ","")
                elif "DATE: " in line:
                    date = line.strip().replace("DATE: ","")
                elif "STORYID: " in line:
                    story_id = line.strip().replace("STORYID: ","")
                elif "TEXT:" in line:
                    text_switch = True
            else:
                if line != '\n':
                    text.append(line.strip())
                

    d = {}
    d["headline"] = headline
    d["date"] = date
    d["story_id"] = story_id    
    d["text"] = " ".join(text)

    return d

def parse_questions_file(path, file_name):
    f = os.path.join(path, str(file_name + '.questions'))
    question_ids = []
    questions = []
    difficulties = []
    with open(f) as file:
        for line in file:
            if "QuestionID: " in line:
                line = line.strip().replace("QuestionID: ","")
                question_ids.append(line)
            elif "Question: " in line:
                line = line.strip().replace("Question: ","")
                questions.append(line)
            elif "Difficulty: " in line:
                line = line.strip().replace("Difficulty: ","")
                difficulties.append(line)

    d = {}
    key = 1
    for id, q, diff in zip(question_ids, questions, difficulties):
        d[key] = [id, q, diff]
        key += 1

    return d

def parse_answers_file(path, file_name, split_multiple_answers=False):
    f = os.path.join(path, str(file_name + '.answers'))
    question_ids = []
    questions = []
    difficulties = []
    answers = []
    with open(f) as file:
        for line in file:
            if "QuestionID: " in line:
                line = line.strip().replace("QuestionID: ","")
                question_ids.append(line)
            elif "Question: " in line:
                line = line.strip().replace("Question: ","")
                questions.append(line)
            elif "Difficulty: " in line:
                line = line.strip().replace("Difficulty: ","")
                difficulties.append(line)
            elif "Answer: " in line:
                line = line.strip().replace("Answer: ","")
                answers.append(line)

    d = {}
    key = 1
    for id, q, diff, a in zip(question_ids, questions, difficulties, answers):
        if split_multiple_answers and "|" in a:
            possible_answers = [x.strip() for x in a.split("|")]
            for i in range(len(possible_answers)):
                d[key] = [id + "-a" +str(i+1), q, diff, possible_answers[i]]
                key += 1
            continue

        d[key] = [id, q, diff, a]
        key += 1

    return d

def get_answer_start_idx(answer, story):
    idx = story.find(answer)
    if idx == -1:
        w_in_answer = answer.split()
        idx = max([story.find(w) for w in w_in_answer])
    return idx + 1 if idx != -1 else 0


def save_training_dataset():
    input_file = sys.argv[1]
    path, question_IDs = read_input_file(input_file)
    json_list = []
    for q in question_IDs:
        story = parse_story_file(path, q)
        answers = parse_answers_file(path, q, True)
        for key in answers.keys():
            row = {}
            row["answers"] = {"answer_start":[get_answer_start_idx(answers[key][3], story["text"])], "text":[answers[key][3]]}
            row["context"] = story["text"]
            row["id"] = answers[key][0]
            row["question"] = answers[key][1]
            row["title"] = story["headline"]
            json_list.append(row)

    with open("data.json2", "w") as result_file:
        json.dump(json_list, result_file, indent=4)

