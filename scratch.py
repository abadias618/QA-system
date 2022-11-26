import json
from nltk import pos_tag, word_tokenize, CFG
import nltk
nltk.download('universal_tagset')

import grammars

with open("data.json") as file:
    data = json.load(file)
print(type(data))
questions = []
answers = []
stories = []
for _ in data:
    questions.append(_["question"])
    answers.append(_["answers"]["text"])
    stories.append(_["context"])

#for a,q in zip(answers, questions):
#    print(q, a)
# https://pdf.sciencedirectassets.com/271647/1-s2.0-S0306457318X00053/1-s2.0-S0306457317308981/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEIn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDJyTxuAdKjYnb4QyMHn3KN0cjSXl6B%2F5XBjXvgy6HzygIhAKbzVteNWOYIAV%2BDmzrBFNbWGTaaWBLpD0y%2B1cUMfY6bKtUECIL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgwfjZWWgyLosTwARzwqqQQYjO2Dd%2FnFbtytPa%2BNkRdlBO47dokWFD%2FCKDORtgkWMy6XNGkVwDDEzAn6KJufWsFmSApV87gif1cFvTRplcgq%2BB18snrIj%2FL95nqc0%2BX3VUJDOS9nbMhJShI%2FwLZCVi2nA2goGqJnY9zflItiaxlTQ3%2B7kHRTlZPZ0ybgbfM6A37s71KI95Z94OonSHw6gVvV9jbzx9xcvIK4ABBgltG8JMCc7jzflo6bS6PBCpDOCJ1gJ%2FSjXYlAVOr9OyuCAXJUBL%2FHe5%2B0wW%2BFDOZXPKIfSsU2xSVzTIIdcVzgh%2FlBApnbTdfpw2X3i3rIwOFhbCuoac4DR8Qx5J82ZM4f9B2iAhzUV9JsLllJr2zupWiZPfQNfHLYgdH6cwto%2BCqkDjLjLOVsk9qN2h6EXBeUDBRNfljJ6EqWidaxqoryJmpydKNeIvpPQn4ybJJLxG4Sb4wqVcY01Za2Nv8mtOW5pJ36WH7ODTIvuUorux6E4919HBFttH1D%2B766WISVjKPKVhPtD2X6ZXgq8LFDRBch54feq1ftJ6fr3phZRXn8aVcgN4lm3MXSHvizZ2CS%2BNrOI4A%2BguaQkLTjAjL9lgscfYYaqXy%2FQgBMdACSpRSdbL4G5zmLgxkUMlScJWKIM%2FE6C4xMcA8kI9PFXH75wOqB1UqbQwBcqqRN4m68lN%2BTJmvqrWFWDcCG3m7A5lfJo87B7%2BBznMzClj0YgnZMFqksBPBL1hFFSrWSAU2FMIu%2FtpsGOqgB%2BuaSw9tQmMf34iIZ2Xt7iIdFKWMQ%2F8it%2Bu4MJyvUOtBu51qrWuOyunT8AqxGVrmvEC8qia6HZy4J7gEVgKPgr6wOgdmGasMiDUFRHaBCO%2FeZWHlyp17kQJH%2BbuBYGRu6pbWaf9cgw2169ucaHr53ARYQth5PO3wk5lxeRkhF19LDVDM3jprcOdN8vQSF%2B%2FJS%2FrsO1YzQs%2BkgroXdKi%2FveAf%2BWlKlMkFi&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20221111T014535Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY2C7QNDY6%2F20221111%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=3768498d1bdea54bd12d4a5f6069ba6c0192359230b2b8a45f4fb5870532dfce&hash=9ad6a34309498afe92aea7d3e65d9c841f9d593ce36dfebc0710ff74cab183a8&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0306457317308981&tid=spdf-53fc7418-c64d-41fc-af65-263c4e9937a1&sid=7c0a36d35a30574e4f288ed2fd090877961cgxrqa&type=client&ua=58505e54515d5707&rr=7683596e88c827e8
# https://pure.port.ac.uk/ws/portalfiles/portal/12843865/MHDW_18_ALAA.pdf
def manually_label_question(q):
    factoid_words = ["what", "where", "why", "who", "whose", "when", "which", "how", "how many", "how often", "how far","how much", "how long", "how old"]
    choice_words = ["or"]
    causal_words = ["how","why", "because"]
    confirmation_words = ["can", "can't", "is", "isn't","was","wasn't","were","weren't"]
    hypothetical_words = ["what if", "what would you do if", "what would happen if", "if"]
    list_words = ["are in", "list"]
    quantity_words = ["many", "much", "amount", "quantity", "old","age", "time", "$"]

    q_type = ""
    for w in q.lower().split():
        # if w in choice_words:
        #     q_type = "choice"
        #     break
        # elif w in causal_words:
        #     q_type = "causal"
        #     break
        # elif w in confirmation_words:
        #     q_type = "confirmation"
        #     break
        # elif w in hypothetical_words:
        #     q_type = "hypothetical"
        # elif w in list_words:
        #     q_type = "list"
        #     break
        if w in quantity_words:
            q_type = "quantity"
            break
        elif w in factoid_words:
            q_type = "factoid"
            break
    return q_type

def convert_universal_tag_to_CFG_format(tag):
    CFG_tag = "U"
    if tag == "VERB":
        CFG_tag = "V"
    elif tag == "NOUN":
        CFG_tag = "N"
    elif tag == "DET":
        CFG_tag = "det" # determiner / article
    elif tag == "PRON":
        CFG_tag = "p" # pronun
    elif tag == "ADV":
        CFG_tag = "adv"
    elif tag == "ADJ":
        CFG_tag = "adj"
    elif tag == "ADP":
        CFG_tag = "adp" # adpositional phrase
    elif tag == "NUM":
        CFG_tag = "num"
    elif tag == "PRON":
        CFG_tag = "pron" # pronoun
    elif tag == "CONJ":
        CFG_tag = "conj" # conjunciton
    elif tag == "PRT":
        CFG_tag = "prt" # particle
    return CFG_tag

from nltk.parse import RecursiveDescentParser

for q in questions[0:1]:
    q_type = ""
    if manually_label_question(q) == "factoid":
        grammar_str = grammars.FACTOID
        q_type = "factoid"
    # elif manually_label_question(q) == "choice":
    #     grammar_str = grammars.CHOICE
    #     q_type = "choice"
    # elif manually_label_question(q) == "causal":
    #     grammar_str = grammars.CAUSAL
    #     q_type = "causal"
    # elif manually_label_question(q) == "confirmation":
    #     grammar_str = grammars.CONFIRMATION
    #     q_type = "confirmation"
    # elif manually_label_question(q) == "hypothetical":
    #     grammar_str = grammars.HYPOTHETICAL
    #     q_type = "hypothetical"
    # elif manually_label_question(q) == "list":
    #     grammar_str = grammars.LIST
    #     q_type = "list"
    elif manually_label_question(q) == "quantity":
        grammar_str = grammars.QUANTITY
        q_type = "quantity"

    tokenized_q = word_tokenize(q)
    tokenized_q = [w.lower() for w in tokenized_q]
    tags = pos_tag(tokenized_q, tagset='universal')
    for t in tags:
        grammar_str += convert_universal_tag_to_CFG_format(t[1]) + " -> " + str("\'"+ t[0] +"\'") + "\n\t"

    grammar = CFG.fromstring(grammar_str)
    #print("GRAMMAR",grammar)
    rd = RecursiveDescentParser(grammar)
    print("q_type",q_type)
    print("t_q",tokenized_q)

    for i in range(len(tokenized_q)):
        j = i+2
        if j < len(tokenized_q):
            sub_tokenized = [tokenized_q[i],tokenized_q[i+1],tokenized_q[j]]
            for t in rd.parse(sub_tokenized):
                print("T",t, sub_tokenized)
    

    # grammar = CFG.fromstring(
    # """
    # S -> NP VP
    # PP -> P NP
    # NP -> 'the' N | N PP | 'the' N PP
    # VP -> V NP | V PP | V NP PP
    # N -> 'cat'
    # N -> 'dog'
    # N -> 'rug'
    # V -> 'chased'
    # V -> 'sat'
    # P -> 'in'
    # P -> 'on'
    # """
    # )
    
# for q,a,s in zip(questions[0:10],answers[0:10],stories[0:10]):
#     print(q)
#     print(manually_label_question(q.lower()))
#     print(pos_tag(word_tokenize(q), tagset='universal'))
#     print(a)
#     print(pos_tag(word_tokenize(a[0]), tagset='universal'))
#     #print(pos_tag(word_tokenize(s), tagset='universal'))
#     #print(s)
