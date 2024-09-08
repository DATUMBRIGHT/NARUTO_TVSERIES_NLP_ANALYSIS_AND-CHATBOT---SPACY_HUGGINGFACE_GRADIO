from bs4 import BeautifulSoup


class Cleaner:

    def __init__(self):

        pass
    
    def remove_html_tags(self,text):
        soup = BeautifulSoup(text,'xlmx')
        text = soup.text
        return text
    
    
    def clean(self,text):
        text = self.text.lower()  # Convert to lower case
        text = text.strip()  # Remove leading and trailing spaces
        
        # Remove special characters, URLs, and extra spaces
        text = re.sub(r'https?://\S+', ' ', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters (keep letters, numbers, and spaces)
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        
        return text