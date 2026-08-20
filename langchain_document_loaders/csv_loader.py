from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='customers.csv') # row-based
docs = loader.load()
print(len(docs))
print(docs[0])