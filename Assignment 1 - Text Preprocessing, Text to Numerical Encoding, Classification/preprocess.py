import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
class Preprocess(object):
    def __init__(self):
        self.regex = '^\s+|\W+|[0-9]|\s+$' # for autograder consistency, please do not change

    def clean_text(self, text):
        '''
        Clean the input text string:
            1. Remove HTML formatting (use Beautiful Soup)
            2. Remove non-alphabet characters such as punctuation or numbers and replace with ' '
               You may refer back to the slides for this part (For autograder consistency, 
               we implement this part for you, please do not change it.)
            3. Remove leading or trailing white spaces including any newline characters
            4. Convert to lower case
            5. Tokenize and remove stopwords using nltk's 'english' vocabulary
            6. Rejoin remaining text into one string using " " as the word separator
            
        Args:
            text: string 
        
        Return:
            cleaned_text: string
        '''
        # Step 1
        soup = BeautifulSoup(text)
        cleaned_text = soup.get_text()

        # Step 2 is implemented for you, please do not change
        cleaned_text = re.sub(self.regex,' ',cleaned_text).strip()
        
        # Step 3
        cleaned_text = cleaned_text.strip()
        
        # Step 4
        cleaned_text = cleaned_text.lower()
        print(cleaned_text)

        # Step 5
        tokens = word_tokenize(cleaned_text)
        cleaned_tokens = []
        stop_words = set(stopwords.words('english'))
        for word in tokens:
            if word not in stop_words:
                cleaned_tokens.append(word)
        
        cleaned_text = " ".join(cleaned_tokens)

        return cleaned_text

    def clean_dataset(self, data):
        '''
        Given an array of strings, clean each string in the array by calling clean_text()
            
        Args:
            data: list of N strings
        
        Return:
            cleaned_data: list of cleaned N strings
        '''
        ret_list = []
        for item in data:
            cleaned_item = self.clean_text(item)
            ret_list.append(cleaned_item)

        return ret_list

# Note that clean_wos is outside of the Preprocess class
def clean_wos(x_train, x_test):
    '''
    ToDo: Clean both the x_train and x_test dataset using clean_dataset from Preprocess
    
    Input:
        x_train: list of N strings
        x_test: list of M strings
        
    Output:
        cleaned_text_wos: list of cleaned N strings
        cleaned_text_wos_test: list of cleaned M strings
    '''

    preprocessor = Preprocess()
    cleaned_x_train = preprocessor.clean_dataset(x_train)
    cleaned_x_text = preprocessor.clean_dataset(x_test)

    return cleaned_x_train, cleaned_x_text