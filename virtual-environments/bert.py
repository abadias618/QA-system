from transformers import AutoTokenizer, AutoModel

# used to tokenize the
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# load the pre-trained model
model = AutoModel.from_pretrained("bert-base-uncased")

# example input sentence
text = "alphabet"

# encode and pass through the BERT
#print("t vocab",tokenizer.vocab)
#print("t vocab",tokenizer)
print("t vocab size",tokenizer.vocab_size)
encoded_input = tokenizer(text, return_tensors="pt")
print("encoded_input=",encoded_input)
output = model(**encoded_input)
print("output=",output.last_hidden_state.shape)
print("output=",output.pooler_output.shape)
print("output=",output.pooler_output)