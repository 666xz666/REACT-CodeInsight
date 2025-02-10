import time
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import os
import shutil

import lucene
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField, IntPoint
from org.apache.lucene.index import DirectoryReader, IndexWriter, IndexWriterConfig
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.store import NIOFSDirectory
from java.nio.file import Paths

tqdm.pandas()


class EnhancedRetriever:
    def __init__(self, source_file, index_dir='index'):
        self.source_data = pd.read_csv(source_file)

        def string_to_numpy(s):
            return np.fromstring(s, dtype=np.float32, sep=',')

        # 假设 feature_vector 是一个字符串形式的张量，需要转换为 numpy 数组
        self.source_data['feature_vector'] = self.source_data['feature_vector'].apply(string_to_numpy)

        checkpoint = "Salesforce/codet5p-110m-embedding"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(self.device)

        # Initialize PyLucene
        lucene.initVM()
        self.analyzer = StandardAnalyzer()
        self.index_dir = index_dir

        self.index = self.load_or_create_index()

    def load_or_create_index(self):
        if os.path.exists(self.index_dir):
            print("Loading existing index...")
            return NIOFSDirectory(Paths.get(self.index_dir))
        else:
            print("Creating new index...")
            os.makedirs(self.index_dir)
            index = NIOFSDirectory(Paths.get(self.index_dir))
            config = IndexWriterConfig(self.analyzer)
            writer = IndexWriter(index, config)

            for i, row in self.source_data.iterrows():
                doc = Document()
                doc.add(TextField("diff", row['diff'], Field.Store.YES))
                doc.add(TextField("ast_seq", row['ast_seq'], Field.Store.YES))  # 新增字段
                doc.add(StringField("index", str(i), Field.Store.YES))
                doc.add(IntPoint("index", i))
                writer.addDocument(doc)

            writer.close()
            return index

    @staticmethod
    def truncate_sequence(full_string):
        index_ = full_string.find('+++ b')
        if index_ == -1:
            return ""
        return full_string[index_:].replace('\n', '')

    def normalize_bm25_scores(self, bm25_scores):
        min_score = np.min(bm25_scores)
        max_score = np.max(bm25_scores)

        if max_score == min_score:
            return np.zeros_like(bm25_scores)

        normalized_scores = (bm25_scores - min_score) / (max_score - min_score)
        return normalized_scores

    def calculate_bm25(self, diffs):
        start_time = time.time()
        searcher = IndexSearcher(DirectoryReader.open(self.index))
        searcher.setSimilarity(BM25Similarity())
        scores_matrix = []

        print("Calculating BM25 with PyLucene")
        for diff in tqdm(diffs, desc="BM25 Queries"):
            query = QueryParser("diff", self.analyzer).parse(QueryParser.escape(diff))
            hits = searcher.search(query, len(self.source_data)).scoreDocs

            scores = np.zeros(len(self.source_data))
            for hit in hits:
                doc = searcher.doc(hit.doc)
                doc_id = int(doc.get("index"))
                if doc_id < len(scores):
                    scores[doc_id] = hit.score

            scores_matrix.append(scores)

        end_time = time.time()
        print(f"Time to calculate BM25 with PyLucene: {end_time - start_time}")

        scores_matrix = np.array(scores_matrix)
        normalized_scores_matrix = np.apply_along_axis(self.normalize_bm25_scores, 1, scores_matrix)

        return normalized_scores_matrix

    def calculate_cosine_sim(self, diffs):
        def text_to_embedding(text, tokenizer, model, device):
            inputs = tokenizer.encode(text, max_length=512, truncation=True, padding='max_length',
                                      return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs)[0]
            embedding = outputs.cpu().numpy()
            return embedding

        start_time = time.time()
        query_embeddings = [text_to_embedding(diff, self.tokenizer, self.model, self.device) for diff in diffs]
        source_embeddings = np.stack(self.source_data['feature_vector'].values)  # 使用融合后的特征向量
        cosine_sim_matrix = cosine_similarity(np.stack(query_embeddings), source_embeddings)
        print(cosine_sim_matrix.shape)
        end_time = time.time()
        print(f"Time to calculate cosine similarity: {end_time - start_time}")

        return cosine_sim_matrix

    def retrieve(self, diffs, bm25=True, cosine=True):
        cosine_sim_matrix = None
        bm25_scores_matrix = None
        if cosine:
            cosine_sim_matrix = self.calculate_cosine_sim(diffs)

        if bm25:
            bm25_scores_matrix = self.calculate_bm25(diffs)

        if cosine and bm25:
            if cosine_sim_matrix.shape == bm25_scores_matrix.shape:
                weighted_sim_matrix = cosine_sim_matrix + bm25_scores_matrix
            else:
                raise ValueError("Cosine similarity matrix and BM25 scores matrix have different shapes")
        elif cosine:
            weighted_sim_matrix = cosine_sim_matrix
        elif bm25:
            weighted_sim_matrix = bm25_scores_matrix
        else:
            raise ValueError("At least one of cosine or bm25 must be True")

        most_similar_diffs = []
        most_similar_messages = []
        same_diff_count = 0

        for i in range(len(diffs)):
            max_index = 0
            second_max_index = 1
            max_value = weighted_sim_matrix[i][0]
            second_max_value = weighted_sim_matrix[i][1]

            for idx, value in enumerate(weighted_sim_matrix[i]):
                if value > max_value:
                    second_max_value = max_value
                    second_max_index = max_index
                    max_value = value
                    max_index = idx
                elif value > second_max_value:
                    second_max_value = value
                    second_max_index = idx

            if (self.truncate_sequence(self.source_data['diff'][max_index]) == self.truncate_sequence(diffs[i]) and
                    self.truncate_sequence(diffs[i]) != ""):
                most_similar_index = second_max_index
                same_diff_count += 1
            else:
                most_similar_index = max_index
            most_similar_diffs.append(self.source_data['diff'][most_similar_index])
            most_similar_messages.append(self.source_data['message'][most_similar_index])

        df = pd.DataFrame(
            {
                'query_diff': diffs,
                'retrieved_diff': most_similar_diffs,
                'retrieved_message': most_similar_messages
            }
        )

        print(f"Number of same diffs: {same_diff_count} out of {len(diffs)}")

        return df

    def retrieve_bm25(self, diffs):
        searcher = IndexSearcher(DirectoryReader.open(self.index))
        searcher.setSimilarity(BM25Similarity())

        most_similar_diffs = []
        most_similar_messages = []
        same_diff_count = 0

        print("Retrieving BM25 with PyLucene")
        for diff in tqdm(diffs, desc="BM25 Queries"):
            query = QueryParser("diff", self.analyzer).parse(QueryParser.escape(diff))
            hits = searcher.search(query, 2).scoreDocs  # Only get the top 2 hits

            if len(hits) > 0:
                max_doc = searcher.doc(hits[0].doc)
                max_diff = max_doc.get("diff")
                max_index = int(max_doc.get("index"))
            else:
                max_diff = self.source_data['diff'][0]
                max_index = 0
                print("Empty Hits.")

            if len(hits) > 1:
                second_max_doc = searcher.doc(hits[1].doc)
                second_max_diff = second_max_doc.get("diff")
                second_max_index = int(second_max_doc.get("index"))
            else:
                second_max_diff = self.source_data['diff'][0]
                second_max_index = 0

            if (self.truncate_sequence(max_diff) == self.truncate_sequence(diff) and
                    self.truncate_sequence(diff) != ""):
                most_similar_diff = second_max_diff
                most_similar_index = second_max_index
                same_diff_count += 1
            else:
                most_similar_diff = max_diff
                most_similar_index = max_index

            most_similar_diffs.append(most_similar_diff)
            most_similar_messages.append(self.source_data['message'][most_similar_index] if most_similar_index != -1 else "")

        df = pd.DataFrame(
            {
                'query_diff': diffs,
                'retrieved_diff': most_similar_diffs,
                'retrieved_message': most_similar_messages
            }
        )

        print(f"Number of same diffs: {same_diff_count} out of {len(diffs)}")

        return df