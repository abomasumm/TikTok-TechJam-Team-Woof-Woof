"""Small contract tests for split isolation and submission validation."""
import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy  # noqa: F401 - importing verifies the runtime dependency
except ModuleNotFoundError:
    NUMPY_AVAILABLE = False
else:
    NUMPY_AVAILABLE = True
    from baseline import main as baseline_main, run_fm
    from data import iter_interactions, load
    from submit import HEADER, main as submit_main, read_submission, write_submission


LOG_FIELDS = [
    'user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'is_click',
    'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate',
    'long_view', 'play_time_ms', 'duration_ms', 'profile_stay_time',
    'comment_stay_time', 'is_profile_enter', 'is_rand', 'tab',
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(NUMPY_AVAILABLE, 'numpy is not installed')
class DataAndSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = self.temp_dir.name
        self._write_user_features()
        self._write_video_features()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_video_features(self):
        path = os.path.join(self.data_dir, 'video_features_basic_pure.csv')
        with open(path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                'video_id', 'author_id', 'music_id', 'video_type', 'upload_type',
            ])
            writer.writeheader()
            for video_id in ('v1', 'v2', 'v3', 'v4'):
                writer.writerow({
                    'video_id': video_id,
                    'author_id': f'a-{video_id}',
                    'music_id': 'm1',
                    'video_type': 'NORMAL',
                    'upload_type': 'LongImport',
                })

    def _write_user_features(self):
        path = os.path.join(self.data_dir, 'user_features_pure.csv')
        fields = [
            'user_id', 'follow_user_num_range', 'register_days_range',
            'fans_user_num_range', 'friend_user_num_range',
            'user_active_degree',
        ]
        with open(path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerow({field: ('u1' if field == 'user_id' else '0') for field in fields})

    def _row(self, user_id, video_id, date, label, duration='1000'):
        row = {field: '0' for field in LOG_FIELDS}
        row.update({
            'user_id': user_id,
            'video_id': video_id,
            'date': str(date),
            'long_view': str(label),
            'duration_ms': duration,
            'tab': '1',
        })
        return row

    def _write_log(self, filename, rows):
        path = os.path.join(self.data_dir, filename)
        with open(path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def _write_complete_logs(self, heldout_duration='1000'):
        self._write_log('log_standard_4_08_to_4_21_pure.csv', [
            self._row('u1', 'v1', 20220408, 1),
            self._row('u1', 'v2', 20220421, 0),
        ])
        self._write_log('log_standard_4_22_to_5_08_pure.csv', [
            self._row('u1', 'v3', 20220422, 1),
            self._row('u1', 'v4', 20220428, 0),
            self._row('u1', 'v1', 20220429, 1, duration=heldout_duration),
        ])

    def test_train_only_does_not_require_late_log(self):
        self._write_log('log_standard_4_08_to_4_21_pure.csv', [
            self._row('u1', 'v1', 20220408, 1),
        ])

        splits = load(self.data_dir, split_names=('train',))

        self.assertEqual(list(splits), ['train'])
        self.assertEqual(len(splits['train']), 1)

    def test_development_load_does_not_materialize_heldout_payload(self):
        # If a held-out row were converted, its deliberately invalid duration
        # would raise. Train+valid loading must ignore that payload.
        self._write_complete_logs(heldout_duration='DO_NOT_PARSE')

        splits = load(self.data_dir, split_names=('train', 'valid'))

        self.assertEqual(list(splits), ['train', 'valid'])
        self.assertNotIn('test', splits)
        self.assertEqual(len(splits['train']), 2)
        self.assertEqual(len(splits['valid']), 2)

    def test_static_feature_ablation_skips_heldout_payload_by_default(self):
        self._write_complete_logs(heldout_duration='DO_NOT_PARSE')

        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / 'ablation_features.py'),
                '--data_dir', self.data_dir,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("'valid': 2", completed.stdout)
        self.assertNotIn("'test':", completed.stdout)

    def test_default_load_remains_backward_compatible(self):
        self._write_complete_logs()

        splits = load(self.data_dir)

        self.assertEqual(list(splits), ['train', 'valid', 'test'])
        self.assertEqual([len(splits[name]) for name in splits], [2, 2, 1])

    def test_streaming_research_signals_respect_requested_splits(self):
        self._write_complete_logs()
        interactions = list(iter_interactions(
            self.data_dir,
            split_names=('train', 'valid'),
            columns=('user_id', 'video_id', 'time_ms', 'long_view', 'play_time_ms'),
        ))
        self.assertEqual([split for split, _ in interactions], ['train', 'train', 'valid', 'valid'])
        self.assertTrue(all(len(values) == 5 for _, values in interactions))

    def test_run_fm_scores_only_requested_split(self):
        self._write_complete_logs(heldout_duration='DO_NOT_PARSE')
        splits = load(self.data_dir, split_names=('train', 'valid'))

        metrics = run_fm(
            splits, k=2, epochs=1, bs=2, patience=1, seed=0,
            verbose=False, eval_splits=('valid',),
        )

        self.assertEqual(list(metrics), ['valid'])

    @mock.patch('baseline.run_random')
    @mock.patch('baseline.load')
    def test_baseline_cli_is_validation_only_by_default(self, mock_load, mock_run):
        mock_load.return_value = {'train': [], 'valid': []}
        mock_run.return_value = {
            'valid': {'GAUC': 0.5, 'nDCG@5': 0.25, 'primary': 0.375},
        }

        baseline_main(['--model', 'random', '--data_dir', 'unused'])

        mock_load.assert_called_once_with(
            'unused', split_names=('train', 'valid'),
        )
        self.assertEqual(mock_run.call_args.kwargs['eval_splits'], ('valid',))

    def test_write_submission_rejects_short_and_long_scores(self):
        rows = [
            (20220422, 'u1', 'v1', 'a1', '1', 1000.0, 1),
            (20220422, 'u1', 'v2', 'a2', '1', 1000.0, 0),
        ]
        path = os.path.join(self.data_dir, 'submission.csv')

        with self.assertRaisesRegex(ValueError, 'score'):
            write_submission(path, rows, [0.1])
        with self.assertRaisesRegex(ValueError, 'score'):
            write_submission(path, rows, [0.1, 0.2, 0.3])
        self.assertFalse(os.path.exists(path))

    def test_submission_round_trip_and_alignment_check(self):
        rows = [
            (20220422, 'u1', 'v1', 'a1', '1', 1000.0, 1),
            (20220422, 'u1', 'v2', 'a2', '1', 1000.0, 0),
        ]
        path = os.path.join(self.data_dir, 'submission.csv')
        write_submission(path, rows, [0.125, -0.5])
        self.assertEqual(read_submission(path, rows), [0.125, -0.5])

        with open(path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(HEADER)
            writer.writerow([0, 'wrong-user', 'v1', '0.1'])
            writer.writerow([1, 'u1', 'v2', '0.2'])
        with self.assertRaisesRegex(ValueError, '对齐错误'):
            read_submission(path, rows)

    def test_cli_refuses_heldout_scoring_before_loading_data(self):
        missing_data_dir = os.path.join(self.data_dir, 'does-not-exist')
        with self.assertRaises(SystemExit) as raised:
            submit_main([
                '--score', '--split', 'test', '--data_dir', missing_data_dir,
                os.path.join(self.data_dir, 'submission.csv'),
            ])
        self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
