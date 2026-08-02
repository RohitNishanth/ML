import dotenv
import voyageai

dotenv.load_dotenv()
vo = voyageai.Client()
# This will automatically use the environment variable VOYAGE_API_KEY.
# Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")

documents = [
    "How are You ?"
]
document1 = [
    "How is it going?"
]
# Embed the documents
documents_embeddings = vo.embed(
    documents, model="voyage-4-large", input_type="document"
).embeddings

documents_embeddings1 = vo.embed(
    document1, model="voyage-4-large", input_type="document"
).embeddings

print(documents_embeddings[0][:5])
print(documents_embeddings1[0][:5])

