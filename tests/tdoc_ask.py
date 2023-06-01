import sys
sys.path.insert(0,'..')

from typing import List, Optional


from dataclasses import dataclass
import click

from chatmates import app
from chatmates.app.doc_ask import DocAsk

from chatmates.mind_tasks import answer_query_from_context
import random

from docarray.typing import NdArray
from docarray import BaseDoc
from pydantic import Field

class IDoc(BaseDoc):
    text: str  # concatenation of title, overview and actors names
    dense_rep: Optional[NdArray[768]] = Field(
        is_embedding=True
    )  # vector representation of the document
    section: Optional[str] = ''

    def make_rep(self, embed_fn):
        self.dense_rep = embed_fn(self.text)

    @staticmethod
    def from_part(part):
        return IDoc(text=part['text'], dense_rep=None, section=part['section'])
    
@click.command()
@click.option("--doc", '-d', help="Text File")
@click.option("--build_index", '-b', is_flag=True, help="build index")
@click.option("--query", '-q', help="query")
@click.option("--model", '-m', help="model id")
def ask_doc(doc, build_index=False, query=None, model='7B'):

    DA = DocAsk(IDoc)

    if build_index:
        assert doc is not None
        with open(doc, 'r') as fp:
            lines = fp.readlines()
            #doc = [dict(text=line) for line in lines if line.strip()] 
            doc = [dict(text=line, section=str(random.randint(0,100)) ) for line in lines if line.strip()] 


    if build_index:
        DA.ingest_embed_index(doc)

    if query: 
        res_scores: '(text, score)*' = DA.query(query, search_field='dense_rep', 
                                                out_fields=['text', 'section'])
        #response = answer_query_from_context(query, res_scores, model_id=model)
        #print(response)

    
@click.group()
def cli():
    pass

cli.add_command(ask_doc)


if __name__ == '__main__':
    cli()