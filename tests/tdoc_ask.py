import sys
sys.path.insert(0,'..')


from dataclasses import dataclass
import click

from chatmates import app
from chatmates.app.doc_ask import DocAsk

from chatmates.mind_tasks import answer_query_from_context

@click.command()
@click.option("--doc", '-d', help="Text File")
@click.option("--build_index", '-b', is_flag=True, help="build index")
@click.option("--query", '-q', help="query")
@click.option("--model", '-m', help="model id")
def ask_doc(doc, build_index=False, query=None, model='7B'):

    DA = DocAsk()

    if build_index:
        if doc is None:
            assert False
            doc = [
                dict(text='a. All employees are eligible to apply for remote work arrangements, subject to their job requirements and departmental approval.'),
                dict(text="b. Remote work may be granted on a full-time or part-time basis, depending on the nature of the job and employee's performance."),
                dict(text="Managers should communicate the decision to the employee within [X] days of receiving the request.")
            ]
        else:
            with open(doc, 'r') as fp:
                lines = fp.readlines()
                doc = [dict(text=line) for line in lines if line.strip()] 


    if build_index:
        DA.ingest_embed_index(doc)

    if query: 
        res_scores: '(text, score)*' = DA.query(query)
        response = answer_query_from_context(query, res_scores, model_id=model)
        print(response)

    
@click.group()
def cli():
    pass

cli.add_command(ask_doc)


if __name__ == '__main__':
    cli()