from langchain_community.document_loaders import PyPDFLoader

pdf_path = r"C:\Users\Shashi Rekha KU\Desktop\pdf\Week2_Day3_LangChain_LangSmith.pdf"

loader = PyPDFLoader(pdf_path)

pages = loader.load()

print("Total pages:", len(pages))

print("\nFirst page content:\n")
print(pages[0].page_content)