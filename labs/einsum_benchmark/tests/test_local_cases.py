import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import unittest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
import benchmark as b
import local_cases as local


class LocalCases(unittest.TestCase):
    def parent(self):
        return dict(id='chain', equation='ab,bc,cd,de->ae', dtype='float64',
                    shapes=[[2,3],[3,4],[4,5],[5,6]],
                    paths=dict(opt_flops=[[0,1],[0,2],[0,1]],
                               opt_size=[[2,3],[0,1],[0,1]]))

    def test_linear_to_ssa_matches_naive_for_random_paths(self):
        rng = random.Random(5)
        for n in range(2, 40):
            live, path, left, right = list(range(n)), [], [], []
            for step in range(n-1):
                li, ri = rng.sample(range(len(live)), 2)
                path.append([li,ri]); left.append(live[li]); right.append(live[ri])
                live = [v for i,v in enumerate(live) if i not in (li,ri)] + [n+step]
            actual_l, actual_r = local.ssa_path(path,n)
            self.assertEqual(list(actual_l),left)
            self.assertEqual(list(actual_r),right)

    def test_cut_preserves_external_indices_not_only_original_output(self):
        parent = self.parent();terms,output=b.validate(parent)
        selected=[1,2]
        paths={name:local.restrict_path(local.ssa_path(path,4),4,selected)
               for name,path in parent['paths'].items()}
        record=local.local_record(parent,terms,output,selected,2,paths,42)
        self.assertEqual(record['equation'],'bc,cd->bd')
        self.assertEqual(record['shapes'],[[3,4],[4,5]])
        for name in paths:
            self.assertTrue(b.verify(record,b.lower(record,name)))

    def test_hyperedge_and_shared_output_are_boundary_axes(self):
        parent=dict(id='hyper',equation='ij,jk,jl->ik',shapes=[[2,3],[3,4],[3,5]],
                    dtype='float32',paths=dict(opt_flops=[[0,1],[0,1]],opt_size=[[1,2],[0,1]]))
        terms,output=b.validate(parent)
        paths={name:local.restrict_path(local.ssa_path(path,3),3,[0,1])
               for name,path in parent['paths'].items()}
        record=local.local_record(parent,terms,output,[0,1],2,paths,42)
        self.assertEqual(record['equation'],'ij,jk->ikj')
        self.assertTrue(b.verify(record,b.lower(record,'opt_flops')))

    def test_repeatable_sampling_and_no_dimension_or_dtype_changes(self):
        parent=self.parent()
        first,missing=local.extract(parent,[2,4],42)
        self.assertEqual((first,missing),local.extract(parent,[2,4],42))
        for record in first:
            self.assertEqual(record['dtype'],parent['dtype'])
            for i,shape in zip(record['provenance']['input_tensor_ids'],record['shapes']):
                self.assertEqual(shape,parent['shapes'][i])
            for name in record['paths']:
                self.assertTrue(b.verify(record,b.lower(record,name)))

    def test_whole_network_restriction_preserves_both_paths(self):
        parent=self.parent()
        for name,path in parent['paths'].items():
            self.assertEqual(local.restrict_path(local.ssa_path(path,4),4,range(4)),path)

    def test_unavailable_connected_region_recorded(self):
        parent=self.parent();parent['equation']='ab,cd,ef,gh->abcdefgh'
        parent['shapes']=[[2,3],[4,5],[6,7],[8,9]]
        cases,missing=local.extract(parent,[2,4],42)
        self.assertEqual(cases,[])
        self.assertEqual([x['size'] for x in missing],[2,4])

    def test_local_and_outside_contraction_equals_original(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy optional for numeric reference')
        rng=np.random.default_rng(4)
        a,bv,c,d=[rng.integers(-2,3,size=shape) for shape in self.parent()['shapes']]
        local_result=np.einsum('bc,cd->bd',bv,c)
        np.testing.assert_array_equal(np.einsum('ab,bd,de->ae',a,local_result,d),
                                      np.einsum('ab,bc,cd,de->ae',a,bv,c,d))


if __name__=='__main__':
    unittest.main()
