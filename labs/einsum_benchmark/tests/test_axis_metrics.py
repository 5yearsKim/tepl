import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark as b


class AxisMetrics(unittest.TestCase):
    def tree(self):
        return dict(root=3, nodes=[
            dict(op='input', name='A0', inputs=[]),
            dict(op='input', name='A1', inputs=[]),
            dict(op='dot_general', inputs=[0,1], attrs=dict(zip(b.FIELDS,[[1],[0],[],[]]))),
            dict(op='transpose', inputs=[2], attrs={'permutation':[1,0]}),
            dict(op='reduce_sum', inputs=[0], attrs={'dimensions':[0]})])

    def test_count_fields_and_positions_and_ignore_unreachable_nodes(self):
        counts=b.axis_inference_metrics(self.tree())
        self.assertEqual(counts['axis_fields'],5)
        self.assertEqual(counts['nonempty_axis_fields'],3)
        self.assertEqual(counts['empty_axis_fields'],2)
        self.assertEqual(counts['axis_positions'],4)
        self.assertEqual(counts['contracting_positions'],2)
        self.assertEqual(counts['permutation_positions'],2)
        self.assertEqual(counts['reduction_positions'],0)
        self.assertEqual(counts['rhs_operations'],2)
        self.assertEqual(counts['rhs_contractions'],1)

    def test_batch_and_reduction_positions_are_occurrence_counts(self):
        tree=self.tree()
        tree['nodes'][2]['attrs']=dict(zip(b.FIELDS,[[],[],[0,1],[0,1]]))
        tree['nodes'][3]=dict(op='reduce_sum',inputs=[2],attrs={'dimensions':[1]})
        counts=b.axis_inference_metrics(tree)
        self.assertEqual(counts['batching_positions'],4)
        self.assertEqual(counts['reduction_positions'],1)
        self.assertEqual(counts['axis_positions'],5)
        self.assertEqual(counts['axis_fields'],5)
        self.assertEqual(counts['empty_axis_fields'],2)

    def test_scalar_empty_permutation_is_inferred_field_with_no_positions(self):
        tree=dict(root=1,nodes=[dict(op='input',inputs=[]),
                                dict(op='transpose',inputs=[0],attrs={'permutation':[]})])
        counts=b.axis_inference_metrics(tree)
        self.assertEqual(counts['axis_fields'],1)
        self.assertEqual(counts['empty_axis_fields'],1)
        self.assertEqual(counts['axis_positions'],0)

    def test_symbolic_values_not_counted_as_concrete_inference(self):
        tree=self.tree();tree['nodes'][3]['attrs']['permutation']=['auto']
        with self.assertRaises(ValueError):b.axis_inference_metrics(tree)


if __name__=='__main__':unittest.main()
