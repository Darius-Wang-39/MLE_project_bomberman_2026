"""Check predictions against the actual course engine, not a second simulator."""
import logging
import unittest
from types import SimpleNamespace

import numpy as np

from agent_code.dueling_engine_aligned.features import FeatureExtractor
from environment import BombeRLeWorld
from items import Bomb, Explosion


class EngineTimingTests(unittest.TestCase):
    def setUp(self):
        self.fx = FeatureExtractor()
        self.world = w = BombeRLeWorld.__new__(BombeRLeWorld)
        w.arena = np.zeros((17, 17), dtype=int)
        w.arena[[0, -1], :] = -1
        w.arena[:, [0, -1]] = -1
        w.bombs, w.coins, w.explosions = [], [], []
        self.owner = SimpleNamespace(name="test", bombs_left=True, bomb_sprite=None,
                                     x=5, y=5, dead=False, add_event=lambda event: None)
        self.owner.get_state = lambda: ("test", 0, self.owner.bombs_left, (5, 5))
        w.active_agents = [self.owner]
        w.round, w.step, w.user_input = 1, 1, "WAIT"
        w.logger = logging.getLogger("engine-timing-test")

    def state(self):
        return self.world.get_state_for_agent(self.owner)

    def assert_forecast(self, steps=7):
        state = self.state()
        schedule = self.fx._bomb_schedule(state, False)
        for step in range(1, steps + 1):
            # Engine order: actions/movement, existing flames, bombs, damage.
            for _, _, pos in schedule:
                blocked = any((b.x, b.y) == pos for b in self.world.bombs)
                self.assertEqual(self.fx._bomb_blocks(pos, step, schedule), blocked)
            self.world.update_explosions()
            self.world.update_bombs()
            actual = set().union(*(set(e.blast_coords) for e in self.world.explosions
                                   if e.is_dangerous()))
            for x in range(1, 16):
                for y in range(1, 16):
                    self.assertEqual(self.fx._is_dangerous((x, y), step, schedule,
                                                         state["explosion_map"]),
                                     (x, y) in actual, (step, x, y))

    def test_overlapping_bombs_do_not_chain(self):
        self.world.bombs = [Bomb((5, 5), self.owner, 0, 3, None),
                            Bomb((7, 5), self.owner, 3, 3, None)]
        self.assertEqual([t for _, t, _ in self.fx._bomb_schedule(self.state(), False)], [1, 4])
        self.assert_forecast()

    def test_all_existing_bomb_timers(self):
        for timer in range(5):
            with self.subTest(timer=timer):
                self.setUp()
                self.world.bombs = [Bomb((5, 5), self.owner, timer, 3, None)]
                self.assert_forecast()

    def test_existing_flames_and_smoke(self):
        for stage, timer in ((0, 2), (0, 1), (1, 2), (1, 1)):
            with self.subTest(stage=stage, timer=timer):
                self.setUp()
                exp = Explosion([(5, 5)], [], self.owner, timer)
                exp.stage = stage
                self.world.explosions = [exp]
                self.assert_forecast()

    def test_new_own_bomb_first_explodes_on_step_five(self):
        schedule = self.fx._bomb_schedule(self.state(), True)
        self.assertEqual(schedule[0][1], 5)
        self.world.perform_agent_action(self.owner, "BOMB")
        self.assert_forecast()

    def test_existing_flame_overlaps_later_bomb(self):
        self.world.explosions = [Explosion([(5, 5), (6, 5)], [], self.owner, 2)]
        self.world.bombs = [Bomb((6, 5), self.owner, 3, 3, None)]
        self.assert_forecast()

    def test_movement_cannot_enter_bomb_on_detonation_turn(self):
        self.world.bombs = [Bomb((6, 5), self.owner, 0, 3, None)]
        schedule = self.fx._bomb_schedule(self.state(), False)
        self.world.perform_agent_action(self.owner, "RIGHT")
        self.assertEqual((self.owner.x, self.owner.y), (5, 5))
        self.assertTrue(self.fx._bomb_blocks((6, 5), 1, schedule))
        self.world.update_bombs()
        self.assertTrue(self.world.tile_is_free(6, 5))
        self.assertFalse(self.fx._bomb_blocks((6, 5), 2, schedule))

    def test_blast_passes_crates_but_stops_at_stone(self):
        self.world.arena[6, 5] = 1
        self.world.arena[8, 5] = -1
        self.world.bombs = [Bomb((5, 5), self.owner, 1, 3, None)]
        blast = self.fx.blast_tiles(self.world.arena, (5, 5))
        self.assertIn((7, 5), blast)
        self.assertNotIn((8, 5), blast)
        self.assert_forecast()


if __name__ == "__main__":
    unittest.main()
