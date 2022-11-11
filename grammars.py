FACTOID = """
S -> Q VP
S -> Q NP
Q -> adv|"what"|"where"|"why"|"who"|"whose"|"when"|"which"|"how"|"how many"|"how often"|"how far"|"how much"|"how long"|"how old"
VP -> V | V NP | V NP V | NP V | NP V NP | V adj | adj V
NP -> N | N N | N N N | N V | adj N | adp N
PP -> p N | p N N
D -> det N | det V | det N N | det N V
"""
CHOICE ="""
S -> NP VP
PP -> P NP
NP -> N | N PP | N PP
VP -> V NP | V PP | V NP PP
"""
CAUSAL ="""
S -> NP VP
PP -> P NP
NP -> N | N PP | N PP
VP -> V NP | V PP | V NP PP
"""
CONFIRMATION="""
S -> NP VP
S -> adv V VP N N N V U
PP -> P NP
NP -> N | N PP | N PP
VP -> V NP | V PP | V NP PP
VP -> N N
"""
HYPOTHETICAL="""
S -> NP VP
PP -> P NP
NP -> N | N PP | N PP
VP -> V NP | V PP | V NP PP
"""
LIST="""
S -> NP VP
PP -> P NP
NP -> N | N PP | N PP
VP -> V NP | V PP | V NP PP
"""
QUANTITY="""
S -> NP VP
PP -> P NP
NP -> N | N PP | N PP
VP -> V NP | V PP | V NP PP
"""