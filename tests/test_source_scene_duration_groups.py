from copy import deepcopy
import unittest
from unittest.mock import patch

from lite import duration_budget


class SourceSceneSummaryTests(unittest.TestCase):
    def test_known_scene_shots_share_a_total_without_modifying_clip_ids(self):
        clips = [{'scene': value, 'seconds': 6} for value in ('S1-01', 'S1-02a', 'S2', 'S3-01', 'S1-main')]
        original = deepcopy(clips)
        with patch.object(duration_budget.task_parameters, 'longform', return_value=True), patch.object(duration_budget, 'issues', return_value=[]):
            result = duration_budget.director_summary({'max_final_seconds':630}, clips, source_scene_ids=['S1', 'S2'])
            previous = duration_budget.director_summary({'max_final_seconds':630}, clips)
        self.assertEqual([row['scene'] for row in result['by_scene']], ['S1', 'S2', 'S3-01', 'S1-main'])
        self.assertEqual(result['by_scene'][0], {'scene':'S1','frames':288,'shot_count':2,'seconds':12})
        self.assertEqual(result['total_frames'], previous['total_frames'])
        self.assertEqual(len(previous['by_scene']), 5)
        self.assertEqual(clips, original)
