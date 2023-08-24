import ingest as ing
import pandas as pd


borough_plan_text = ing.parse_file([open("Lambeth_2030-Our_Future_Our_Lambeth.pdf")])
print("Borough plan text created")

#print(borough_plan_text)

borough_plan_docs, borough_plan_page_docs = ing.text_to_docs(borough_plan_text)
print("Borough plan docs created")

embedding_model = "thenlper/gte-base"

embeddings = ing.load_embeddings(model_name = embedding_model)
ing.embed_faiss_save_to_zip(borough_plan_docs, save_to="faiss_embedding", model_name = embedding_model)