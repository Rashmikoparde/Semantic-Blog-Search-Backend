from flask import Flask, request, Response, jsonify, send_file
import pandas as pd
import pickle as pkl
import flask
data=pd.read_csv('blogdata_entire_data_6000.csv')
import json
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle as pkl
import json
embedder= SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

app = flask.Flask(__name__)
app.config["DEBUG"] = True


with open('CorpusEmbeddings.pkl','rb') as f:
     corpus_embeddings= pkl.load(f)

@app.route('/search/', methods=['GET'])
def get_fiiles():
    search_query = request.args['search_query']
    # Query sentences:
    queries = [search_query]
    query_embeddings = embedder.encode(queries,show_progress_bar=True)

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    closest_n = 10
    print("\nTop 5 most similar sentences in corpus:")
    final_result = []
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n=========================================================")
        print("==========================Query==============================")
        print("===",query,"=====")
        print("=========================================================")
        print(results[0:closest_n])
        for idx, distance in results[0:closest_n]:
            print(idx, data.iloc[idx]['label'])
            x = {
              'score': (1-distance),
              'pos_text':data.iloc[idx]['text'],
              'label': data.iloc[idx]['label'],
              'gender': data.iloc[idx]['gender'],
              'zodiac_sign': data.iloc[idx]['zodiac']
    
            }
            final_result.append(x)
    return jsonify(final_result)
                


if __name__ == '__main__':
    app.run(host='localhost', port=9000, debug=True)
