"""Action ID compatibility checks without launching the engine."""

from types import SimpleNamespace
import unittest

from s2clientprotocol import data_pb2
from bch.arena import ABILITIES, Arena, _macro_ability_ids
from bch.lower_action.macros import Command


class ArenaAbilityTests(unittest.TestCase):
    def move(self, **values):
        return data_pb2.AbilityData(ability_id=16, available=True, link_name="move", button_name="Move", **values)

    def test_new_engine_retains_general_move_and_global_ids(self):
        ids = _macro_ability_ids({3794: data_pb2.AbilityData(ability_id=3794), 16: self.move()})
        self.assertEqual(ids, ABILITIES)
        self.assertIsNot(ids, ABILITIES)

    def test_old_engine_uses_verified_move_without_changing_other_ids(self):
        ids = _macro_ability_ids({16: self.move()})
        self.assertEqual(ids, {"smart": 1, "attack": 3674, "move": 16, "heal": 386})
        self.assertEqual(ABILITIES["move"], 3794)

    def test_missing_or_mislabeled_legacy_move_is_rejected(self):
        for available in ({}, {16: data_pb2.AbilityData(ability_id=16, available=True, link_name="attack", button_name="Attack")},
                          {16: data_pb2.AbilityData(ability_id=16, available=False, link_name="move", button_name="Move")}):
            with self.subTest(available=available), self.assertRaisesRegex(RuntimeError, "verified Move"):
                _macro_ability_ids(available)

    def test_raw_action_uses_the_instance_engine_mapping(self):
        arena = SimpleNamespace(abilities=_macro_ability_ids({16: self.move()}), bounds=((0, 32), (0, 32)))
        action = Arena._raw_action(arena, Command("move", (101, 202), (12.5, 16.0)))
        self.assertEqual(action.action_raw.unit_command.ability_id, 16)
        self.assertEqual(list(action.action_raw.unit_command.unit_tags), [101, 202])
        self.assertEqual(action.action_raw.unit_command.target_world_space_pos.x, 12.5)


if __name__ == "__main__":
    unittest.main()
