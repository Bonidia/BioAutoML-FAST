"""Fast checks for paired searches, overlap accounting, safety and orchestration."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import optuna
import pandas as pd

import _experiment_support as support
from audit_sequence_overlap import audit_dataset
import run_remaining_experiments as controller


class RemainingExperimentsTests(unittest.TestCase):
    def test_pruning_is_scoped_and_only_changes_pruner(self):
        original = optuna.create_study
        for enabled in (True, False):
            with support.pruning_context(enabled):
                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=63))
                for _ in range(5):
                    trial = study.ask()
                    trial.report(1.0, 0)
                    study.tell(trial, 1.0)
                trial = study.ask()
                trial.report(0.0, 0)
                self.assertEqual(trial.should_prune(), enabled)
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                self.assertEqual(study.trials[-1].state, optuna.trial.TrialState.PRUNED)
            self.assertIs(optuna.create_study, original)

    def test_generation_without_test_set(self):
        args = support.generation_arguments(dict(dataset={'task': 1}, tuning=150, n_cpu=32,
            search_seed=6305, output='/tmp/model'), {'train': 'features.csv', 'train_label': 'labels.csv'})
        self.assertNotIn('--test', args)
        self.assertEqual(args[args.index('--task') + 1], '1')
        self.assertEqual(args[args.index('--search_seed') + 1], '6305')

    def test_overlap_counts_records_not_cartesian_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            positive, negative, test = (root / n for n in ('positive.fasta', 'negative.fasta', 'test.fasta'))
            positive.write_text('>p1\nACGU\n>p2\nACGT\n')
            negative.write_text('>n1\nACGT\n')
            test.write_text('>t1\nacgt\n>t2\nACGT\n>t3\nAAAA\n')
            dataset = dict(dataset='fixture', task=0, dtype='dnarna',
                train_files=[str(positive), str(negative)], train_labels=['positive', 'negative'],
                test_files=[str(test)], test_labels=['positive'], train_count=3, test_count=3)
            row = audit_dataset(dataset, root)
            self.assertEqual(row['overlapping_test_records'], 2)
            self.assertEqual(row['unique_overlapping_sequences'], 1)
            self.assertEqual(row['matched_train_records'], 3)
            self.assertEqual(row['conflicting_pairs'], 2)
            self.assertEqual(row['duplicate_train_records'], 2)
            matches = pd.read_csv(root / 'overlapping_test_records.csv')
            self.assertEqual(len(matches), 2)
            self.assertEqual(matches.matching_train_records.tolist(), [3, 3])
            dataset.update(test_files=[], test_labels=[], test_count=0)
            self.assertIsNone(audit_dataset(dataset, root)['overlapping_test_records'])

    def test_regression_overlap_numeric_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, test = root / 'train.fa', root / 'test.fa'
            train.write_text('>a|1.00\nACD\n>b|2\nACD\n')
            test.write_text('>t|1\nACD\n')
            dataset = dict(dataset='regression', task=1, dtype='protein', train_files=[str(train)],
                test_files=[str(test)], train_labels=['x'], test_labels=['x'], train_count=2, test_count=1)
            row = audit_dataset(dataset, root)
            self.assertEqual(row['conflicting_pairs'], 1)

    def test_paired_differences_use_same_dataset_seed_and_task(self):
        rows = [dict(dataset='regression', task=1, dtype='protein', primary_metric='R2',
            run=1, search_seed=6301, pruning=p, test_R2=score, measured_minutes=minutes)
            for p, score, minutes in ((True, .5, 10), (False, .6, 12))]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            support.save_results(rows[:1], root, pruning=True)
            self.assertEqual(len(pd.read_csv(root / 'paired_results.csv')), 0)
            support.save_results(rows, root, pruning=True)
            pair = pd.read_csv(root / 'paired_results.csv').iloc[0]
            self.assertAlmostEqual(pair.score_off_minus_on, .1)
            self.assertEqual(pair.minutes_off_minus_on, 2)

    def test_completed_worker_reused_but_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'run_1'
            output.mkdir()
            config = {'output': str(output), 'search_seed': 6301}
            support.atomic_json(output / 'run.json', dict(settings=config, status='completed'))
            with patch.object(support, 'validate_record') as validate, patch.object(support.subprocess, 'Popen') as popen:
                support.run_worker(root, config, True)
                validate.assert_called_once_with(output)
                popen.assert_not_called()
                with self.assertRaises(ValueError):
                    support.run_worker(root, dict(config, search_seed=6302), True)

    def test_experiment_preserves_output_and_checks_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'experiment'
            with support.experiment(root, {'seed': 63}, False):
                pass
            with self.assertRaises(FileExistsError):
                with support.experiment(root, {'seed': 63}, False):
                    pass
            with self.assertRaises(ValueError):
                with support.experiment(root, {'seed': 64}, True):
                    pass
            self.assertEqual(json.loads((root / 'protocol.json').read_text())['status'], 'completed')

    def test_input_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'input'
            path.write_text('original')
            config = dict(code_sha256=support.code_hashes(), packages=support.homology.package_versions(),
                          dataset={'input_hashes': {str(path): support.file_hash(path)}}, assets={})
            support.verify_config(config)
            path.write_text('changed')
            with self.assertRaises(ValueError):
                support.verify_config(config)

    def test_controller_stops_on_task_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'workflow'
            calls = []
            def stage(name, script, root, state, dry_run=False):
                calls.append((name, dry_run))
                if name == 'pruning_comparison' and not dry_run:
                    raise subprocess.CalledProcessError(1, [script])
            with patch.object(controller, 'ROOT', root), patch.object(controller, 'run_stage', side_effect=stage), \
                 patch.object(controller.os, 'sched_getaffinity', return_value=set(range(32))), \
                 patch.object(controller.shutil, 'disk_usage', return_value=SimpleNamespace(free=1024**4)):
                with self.assertRaises(subprocess.CalledProcessError):
                    controller.foreground()
            self.assertNotIn(('benchmark_repeats', False), calls)
            self.assertEqual(json.loads((root / 'protocol.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
