import re  # regular expression (regex)

text_to_search = '''
abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890
Ha HaHa
google.com
321-555-4321
123.555.1234
123*555*1234
800-555-1234
900-555-1234
Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T
'''
sentence = 'Start a sentence and then bring it to an end'

# r"" è una raw string (ad esempio vede \t come \t e non come un tab)
pattern = re.compile(r"Mr\.?", re.IGNORECASE)  #re.IGNORECASE è una flag (ne esistono molte altre che fanno altre cose)

print("\nMethod 1\n")
# finditer restituisce una lista con i match e le relative posizioni
matches1 = pattern.finditer(text_to_search)
for match1 in matches1:
    print(match1)

print("\nMethod 2\n")
# findall restituisce una lista con i soli match
matches2 = pattern.findall(text_to_search)
for match2 in matches2:
    print(match2)


print("\nMethod 3\n")
# search restituisce solamente il primo match e la relativa posizione
match3 = pattern.search(text_to_search)
print(match3)


'''
MetaCharacters (Need to be escaped if you want to seatch for those):
. ^ $ * + ? { } [ ] \ | ( )


Char    - What is going to be matched

.       - Any Character Except New Line
\d      - Digit (0-9)
\D      - Not a Digit (0-9)
\w      - Word Character (a-z, A-Z, 0-9, _)
\W      - Not a Word Character
\s      - Whitespace (space, tab, newline)
\S      - Not Whitespace (space, tab, newline)

\b      - Word Boundary
\B      - Not a Word Boundary
^       - Beginning of a String
$       - End of a String

[]      - Matches any one of the characters specified inside the brackets
[^]     - Matches any one of the characters not specified inside the brackets

|       - Either Or
( )     - Group

Quantifiers modify a search so that we do not need to find exactly that charcater one and only one time
We can use them to match even if that character is not present or if it's present multiple times
 
Quant   - How many of such characters need to be present in order to get a match

*       - 0 or More
+       - 1 or More
?       - 0 or One
{3}     - Exact Number
{3,7}   - Range of Numbers (Minimum, Maximum)


Example: regex expression for email searching
[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+

Understand this:
    [a-zA-Z0-9_.+-]     matcha tutte le lowercase, uppercase, numeri, e poi i segni _.+-
    +                   matcha se [a-zA-Z0-9_.+-] è presente una o più volte
    @                   matcha @
    [a-zA-Z0-9-]        matcha tutte le lowercase, uppercase, numeri, e il segno -
    +                   matcha se [a-zA-Z0-9-] è presente una o più volte
    \.                  matcha un . ()
    [a-zA-Z0-9-.]       matcha tutte le lowercase, uppercase, numeri, il segno - o un .
    +                   matcha se [a-zA-Z0-9-.] è presente una o più volte
'''

