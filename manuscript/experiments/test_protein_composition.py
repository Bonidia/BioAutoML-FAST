"""Fixed-width protein composition, including sequences shorter than a window."""
import csv
import importlib.util
from itertools import product
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io


SCRIPT = Path(__file__).resolve().parents[2] / 'other-methods/ExtractionTechniques-Protein.py'
spec = importlib.util.spec_from_file_location('protein_composition', SCRIPT)
composition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(composition)


class ProteinCompositionTests(unittest.TestCase):
    def check_composition(self, k, combined=False):
        sequences = ['', 'F', 'V', 'FF', 'ACD', 'ACDEFGHIKLMNPQRSTVWYACD']
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.fasta'
            output = Path(directory) / 'output.csv'
            source.write_text(''.join(f'>record_{i}\n{seq}\n' for i, seq in enumerate(sequences)))
            composition.finput = str(source)
            composition.foutput = str(output)
            composition.labelDataset = 'positive'
            composition.caracteres = list(alphabet)
            with redirect_stdout(io.StringIO()):
                (composition.findKmers if combined else composition.nacSeq)(k)
            with output.open() as stream:
                rows = list(csv.reader(stream))
            sizes = range(1, k + 1) if combined else [k]
            self.assertEqual(len(rows), len(sequences) + 1)
            self.assertEqual(len(rows[0]), 2 + sum(20 ** size for size in sizes))
            for row, sequence in zip(rows[1:], sequences):
                self.assertEqual(len(row), len(rows[0]))
                expected = []
                for size in sizes:
                    windows = [sequence[i:i + size] for i in range(len(sequence) - size + 1)]
                    for letters in product(alphabet, repeat=size):
                        expected.append(windows.count(''.join(letters)) / len(windows) if windows else 0.0)
                self.assertEqual(list(map(float, row[1:-1])), expected)

    def test_aac_dpc_tpc(self):
        for k in (1, 2, 3):
            with self.subTest(k=k):
                self.check_composition(k)

    def test_combined_kmers(self):
        self.check_composition(3, combined=True)

    def test_undefined_pair_descriptors_use_missing_values(self):
        for name, width in [('DDE', 400), ('CTDT', 39)]:
            path = SCRIPT.parent / 'iFeature-modified/codes' / f'{name}.py'
            descriptor_spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(descriptor_spec)
            descriptor_spec.loader.exec_module(module)
            rows = getattr(module, name)([['empty', ''], ['one', 'F'], ['pair', 'FF']], order=None)
            self.assertEqual(len(rows[0]), width + 1)
            self.assertEqual(rows[1][1:], [None] * width)
            self.assertEqual(rows[2][1:], [None] * width)
            self.assertTrue(all(value is not None for value in rows[3][1:]))

    def test_single_residue_fourier_sample_statistics_are_missing(self):
        import numpy as np
        path = SCRIPT.parent / 'mathfeature-modified/methods/Mappings-Protein.py'
        descriptor_spec = importlib.util.spec_from_file_location('mapping', path)
        module = importlib.util.module_from_spec(descriptor_spec)
        descriptor_spec.loader.exec_module(module)
        features = []
        with np.errstate(invalid='ignore'):
            module.feature_extraction(features, [np.float64(196)], [np.float64(14)])
        self.assertEqual(len(features), 19)
        self.assertEqual(features[0], 196)
        self.assertTrue(np.isnan(features[7]))
        self.assertTrue(np.isnan(features[13]))


if __name__ == '__main__':
    unittest.main()
