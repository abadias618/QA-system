
import sys
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from parse_v1 import parse_story_file, parse_questions_file, read_input_file

# https://gist.githubusercontent.com/deepesh321/22b2624c65b466ff6bee403a6fd6d689/raw/8e48b646e51f4fbf41ee01f6dced7549143db465/bert_question_answering

model = AutoModelForQuestionAnswering.from_pretrained('./devset-mitre-trained')
tx = AutoTokenizer.from_pretrained('./devset-mitre-trained')

path, question_IDs = read_input_file(sys.argv[1])
for q in question_IDs:
    story = parse_story_file(path, q)
    qs = parse_questions_file(path, q)
    for key in qs.keys():
        question = qs[key][1]
        paragraph = story["text"]
        
        encoding = tx.encode_plus(text=question,text_pair=paragraph,truncation=True,max_length=512)
        inputs = encoding['input_ids']
        sentence_embedding = encoding['token_type_ids']
        tokens = tx.convert_ids_to_tokens(inputs)

        m  = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
        start_scores, end_scores = m[0],m[1]
        start_scores = torch.reshape(start_scores, (-1,))
        end_scores = torch.reshape(end_scores, (-1,))
        start_index = torch.argmax(m[0])
        end_index = torch.argmax(m[1])
        answer = ' '.join(tokens[start_index:end_index+1])
        subword_edit = ""
        for w in answer.split():
            if w == "[CLS]":
                continue
            if w[0:2] == "##":
                subword_edit += w[2:]
            else:
                subword_edit += " " + w

        id = qs[key][0]
        print("QuestionID: " + id)
        print("Answer:" + subword_edit + "\n")