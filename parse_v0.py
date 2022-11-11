#!/usr/bin/env python3
import sys
import os

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
                    text.append(line.strip().split())
                

    d = {}
    d["headline"] = headline
    d["date"] = date
    d["story_id"] = story_id
    d["text"] = text

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


if __name__ == "__main__":
    input_file = sys.argv[1]
    path, question_IDs = read_input_file(input_file)
    print(path, question_IDs)
    print()
    print(parse_questions_file("/home/abd/Documents/nlpProject/devset-official/", "1999-W02-5"))
    print()
    print(parse_story_file("/home/abd/Documents/nlpProject/devset-official/", "1999-W02-5"))
    print()