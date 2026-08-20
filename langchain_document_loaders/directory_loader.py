from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='Assignments',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()
docs1 = loader.lazy_load()
# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)
# print(docs[47].page_content)

# Load Vs Lazy Load
for document in docs:
    print(document.metadata)

for document in docs1:
    print(document.metadata)