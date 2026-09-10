import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from shelf_audit.recognizer import Recognizer,MODEL_ID,PREPROCESS_ID,file_hash


class IndexIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'reference.jpg').write_bytes(b'reference fixture')
        np.save(self.root/'embeddings.npy',np.array([[1.,0.],[0.,1.]],dtype=np.float32))
        self.meta=dict(model=MODEL_ID,preprocess=PREPROCESS_ID,embedding_dimension=2,catalog_root=str(self.root),embeddings_sha256=file_hash(self.root/'embeddings.npy'),records=[dict(class_id=c,product=c,crop_file='reference.jpg',sha256=file_hash(self.root/'reference.jpg')) for c in ['a','b']])
        self.save()

    def save(self):
        (self.root/'index.json').write_text(json.dumps(self.meta))

    def test_load_valid_index_without_model(self):
        r=Recognizer(self.root,load_encoder=False)
        self.assertEqual(len(r.records),2)

    def test_changed_reference_requires_rebuild(self):
        (self.root/'reference.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Catalog changed'):
            Recognizer(self.root,load_encoder=False)

    def test_changed_vectors_rejected(self):
        np.save(self.root/'embeddings.npy',np.zeros((2,2)))
        with self.assertRaisesRegex(ValueError,'integrity'):
            Recognizer(self.root,load_encoder=False)

    def test_wrong_preprocessing_rejected(self):
        self.meta['preprocess']='other';self.save()
        with self.assertRaisesRegex(ValueError,'preprocessing mismatch'):
            Recognizer(self.root,load_encoder=False)

if __name__=='__main__':unittest.main()
