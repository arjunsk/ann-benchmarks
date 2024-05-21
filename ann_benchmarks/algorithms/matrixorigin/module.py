import subprocess
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ..base.module import BaseANN


class PGVector(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None

        if metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY l2_distance(embedding, :v) LIMIT :n"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        engine = create_engine("mysql+pymysql://root:111@127.0.0.1:6001/ann")
        Session = sessionmaker(bind=engine)
        session = Session()
        session.execute(text("DROP TABLE IF EXISTS items"))
        session.execute(text("CREATE TABLE items (id int, embedding vecf32(:dim))"), {"dim": X.shape[1]})
        print("copying data...")
        with session.copy("COPY items (id, embedding) FROM STDIN") as copy:
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        print("creating index...")
        if self._metric == "euclidean":
            session.execute(
                text("CREATE INDEX USING ivfflat on tbl(embedding) lists=:k op_type 'vector_l2_ops'"),{"k": self._m})
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        session.commit()
        self._cur = session

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute(text("SET @probe_limit=:limit"), {"limit": ef_search})

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        return 0
        # if self._cur is None:
        #     return 0
        # self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        # return self._cur.fetchone()[0] / 1024

    def __str__(self):
        return f"MatrixOrigin(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
