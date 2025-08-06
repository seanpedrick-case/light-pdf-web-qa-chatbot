from tools.ingest import parse_file, text_to_docs, load_embeddings_model, embed_faiss_save_to_zip

borough_plan_text, file_names = parse_file([open("Lambeth_2030-Our_Future_Our_Lambeth.pdf")])
print("Borough plan text created")

#print(borough_plan_text)

borough_plan_docs = text_to_docs(borough_plan_text)
print("Borough plan docs created")

embedding_model =  "mixedbread-ai/mxbai-embed-xsmall-v1" # "mixedbread-ai/mxbai-embed-xsmall-v1" #

embeddings = load_embeddings_model(embeddings_model = embedding_model)
embed_faiss_save_to_zip(borough_plan_docs, save_folder="borough_plan", embeddings_model_object= embeddings, save_to="faiss_embedding", model_name = embedding_model)