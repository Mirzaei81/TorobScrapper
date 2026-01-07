import re 
from unidecode import unidecode
pat = "\(([\u0660-\u0669\u06F0-\u06F9]+) (سال|ماه).*\)"
s = "★۵ (۸۲ ماه در ترب)"
g = re.search(pat,s)
print(g.group(2)=="ماه",unidecode(g.group(1)))