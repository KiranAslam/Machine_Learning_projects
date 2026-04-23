import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
base_stop_words = set(stopwords.words('english'))
negation_words = set(['not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'hardly', 'scarcely', 'barely'])
final_stop_words = base_stop_words - negation_words

def preprocessing(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.replace(":)", " happy_emoji ").replace(":-)", " happy_emoji ")
    text = text.replace(":(", " sad_emoji ").replace(":-(", " sad_emoji ")

    text = re.sub(r'\b(not|no|never)\s+(\w+)', r'\1_\2', text)

    text = re.sub(r'[^a-zA-Z_\s]', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in final_stop_words]
    return ' '.join(cleaned_words)

#print(preprocessing("I am not happy with this product! :( http://example.com #disappointed"))
#print(preprocessing("This is the best day ever! :) #excited"))
