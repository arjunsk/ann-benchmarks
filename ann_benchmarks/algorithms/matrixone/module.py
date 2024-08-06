import subprocess
import sys

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ..base.module import BaseANN


def to_db_binary(value):
    if value is None:
        return value

    value = np.asarray(value, dtype='<f4')
    if value.ndim != 1:
        raise ValueError('expected ndim to be 1')


class MatrixOne(BaseANN):
    def __init__(self, metric, lists):
        self._metric = metric
        self._lists = lists
        self._cur = None

        if metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY l2_distance(embedding, %s) LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        subprocess.run("service mo start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

        engine = create_engine("mysql+mysqldb://root:111@127.0.0.1:6001/")
        Session = sessionmaker(bind=engine)
        cur = Session()

        cur.execute("SET GLOBAL experimental_ivf_index = 1;")
        cur.execute("DROP TABLE IF EXISTS items;")
        cur.execute("CREATE TABLE items (id int, embedding vecf32(%d));" % X.shape[1])
        # cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        print("copying data...")
        sql_insert = text("INSERT INTO items (id, embedding) VALUES (:id, cast(unhex(:data) as blob));")
        for i, embedding in enumerate(X):
            cur.execute(sql_insert, {"id": i, "data": to_db_binary(embedding)})
        print("creating index...")
        if self._metric == "euclidean":
            cur.execute(
                "CREATE INDEX USING ivfflat on items (embedding) WITH op_type = 'vector_l2_ops'  lists=%d;" % self._lists)
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        self._cur = cur

    def set_query_arguments(self, probes):
        self._probes = probes
        self._cur.execute("SET ivfflat.probes = %d" % probes)

    def query(self, v, n):
        self._cur.execute(self._query, (v, n))
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        return 0

    def __str__(self):
        return f"MatrixOne(lists={self._lists}, probes={self._probes})"
