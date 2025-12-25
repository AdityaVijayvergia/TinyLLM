import pickle

# Load the tokenizer
with open(".cache/tokenizer/64K/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

raw_str = """
Hello World. I am Aditya. Today is Thursday, Dec 24, 2025. The time is 18:31:07.
"""

tokens = tokenizer.encode(raw_str) 
print("Token Count: ", len(tokens))
print("Tokens: ", tokens)


print("Decoded Tokens: ", end="")
for i, token in enumerate(tokens):
    # ANSCII Color codes
    print(f"\033[{i%4+41}m{tokenizer.decode([token])}\033[0m", end="")
