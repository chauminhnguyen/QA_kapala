import re
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from glob import glob
import os


class Base_Preprocess:
    def __init__(self, corpus_path):
        self.documents = []
        self.corpus_path = corpus_path
        self.run()

    def remove_redundant_section(self, text):
        """Removes the "Mục lục" and tail section from the input text.

        Args:
            text: The input text.

        Returns:
            The text with the "Mục lục" and tail section removed.
        """

        # Find the start and end of the "Mục lục" section.
        start_match = re.search(r"<h3>Mục lục</h3>", text)
        end_match = re.search(r"<h2>", text)

        # If the section is found, remove it.
        if start_match and end_match:
            text = text[:start_match.start()] + text[end_match.end():]

        start_match, end_match = None, None

        start_match = re.search(r"vui lòng liên hệ", text)
        end_match = re.search(r"tamanhhospital.vn", text)

        if start_match and end_match:
            text = text[:start_match.start()] + text[end_match.end():]

        return text

    def clean_text(self, text):
        """Cleans the input text by removing HTML tags, punctuation, capitalization, and emojis.

        Args:
            text: The input text.

        Returns:
            The cleaned text.
        """

        # Remove HTML entities.
        text = re.sub(r"&[^;]+;", "", text)

        # Remove HTML tags.
        text = re.sub(r"<[^>]+>", "", text)

        # Remove punctuation.
        # text = text.translate(str.maketrans('', '', string.punctuation))

        # Lowercase the text.
        text = text.lower()

        # # Remove emojis.
        # text = re.sub(r"[^\w\s]", "", text)

        # Replace \n
        text = text.replace("\n", ".")

        return text

    def run(self):
        folder_path = glob(os.path.join(self.corpus_path, '*'))
        data = []
        fname = []
        for idx, fpath in enumerate(folder_path):
            with open(fpath) as f:
                try:
                    data.append(self.remove_redundant_section(f.read().split("<h1>")[1]))
                    fname.append({"document" : fpath.split('/')[-1]})
                except:
                    pass
        
        text_splitter = RecursiveCharacterTextSplitter(
            separators = ["<h2>", "\n\n", "<h3>",],
            chunk_size = 500,
            chunk_overlap  = 100,
            length_function = len,
        )

        docs = text_splitter.create_documents(data, metadatas=fname)

        for idx, text in enumerate(docs):
            doc = Document(page_content=self.clean_text(text.page_content), id=idx, metadata=text.metadata)
            self.documents.append(doc.page_content)
    
    def get_documents(self):
        return self.documents