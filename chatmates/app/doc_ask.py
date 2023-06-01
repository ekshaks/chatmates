
from typing import List, Optional


from docarray import BaseDoc, DocList
from docarray.index import HnswDocumentIndex as HI

from chatmates.embed import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings

from chatmates.common import log


class DocAsk:
    def __init__(self, IDoc, index_path='./idoc_index/default') -> None:
        '''
        @IDoc : BaseDoc instance to be stored
        '''
        super().__init__()
        self.encoder = None
        self.embed_dim = 768
        self.idocs: List[IDoc] = []

        self.index = HI[IDoc](work_dir=index_path)
        self.IDoc = IDoc

    
    def _embed_text(self, text):
        if self.encoder is None:
            self.encoder = HuggingFaceInstructEmbeddings()

        vector = self.encoder.embed_query(text) 
        return vector
    
    def ingest_embed_index(self, json_doc):
        self.doc = json_doc #simple list of docs (f1 | f2 | ..)*

        log('build idoc')
        self.idocs = [self.IDoc.from_part(part) for part in self.doc]
        log('embedding')
        for doc in self.idocs:
            doc.make_rep(self._embed_text)

        log('indexing')
        dl = DocList[self.IDoc](self.idocs)
        self.dl = dl
        self.index.index(dl)
        
    def query(self, query_text, search_field, limit=10, out_fields=None):
        qrep = self._embed_text(query_text)
        store = self.index
        #vector_q = store.build_query().find(qrep).limit(10).build()
        #vector_results = store.execute_query(vector_q)
        results, scores = store.find(qrep, limit=limit, search_field=search_field)

        results = [ {key: getattr(res, key) for key in out_fields}
                for res in results]
        #results_text = [res.text for res in results]
        res_scores = list(zip(results, scores))
        for res, score in res_scores:
            print(score, res)

        return res_scores

    



