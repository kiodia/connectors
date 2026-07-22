import bz2
import gzip
import lzma
import pickle

import brotli


class SomeObject():

    a = 'some data'
    b = 123
    c = 'more data'

    def __init__(self, i):
        self.i = i


data = [SomeObject(i) for i in range(1, 1000000)]

with open('no_compression.pickle', 'wb') as f:
    pickle.dump(data, f)

with gzip.open("gzip_test.gz", "wb") as f:
    pickle.dump(data, f)

with bz2.BZ2File('bz2_test.pbz2', 'wb') as f:
    pickle.dump(data, f)

with lzma.open("lzma_test.xz", "wb") as f:
    pickle.dump(data, f)

with open('no_compression.pickle', 'rb') as f:
    pdata = f.read()
    with open('brotli_test.bt', 'wb') as b:
        b.write(brotli.compress(pdata))