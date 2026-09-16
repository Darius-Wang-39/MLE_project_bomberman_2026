"""Regression tests for the engine's survivor/death callback ordering."""
from collections import deque
from contextlib import contextmanager
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from environment import BombeRLeWorld, GenericWorld


MODULES = ("dueling_ddqn_agent", "dueling_priority23")


def state(step, round_id=1):
    return {"round": round_id, "step": step}


@contextmanager
def trainer(name):
    module = importlib.import_module(f"agent_code.{name}.train")
    replay = importlib.import_module(f"agent_code.{name}.replay")
    runtime = SimpleNamespace(
        pending_step=None, n_step_buffer=deque(), environment_steps=0,
        learner_updates=0, replay=replay.PrioritizedReplayBuffer(100),
        feature_extractor=SimpleNamespace(safe_action_mask=lambda s: np.ones(6, dtype=bool)),
        policy_net=Mock(), optimizer=Mock(),
    )
    rewards = {"WAITED": -0.15, "COIN_COLLECTED": 4.0,
               "SURVIVED_ROUND": 2.0, "GOT_KILLED": -15.0}
    with patch.object(module, "vectorize_state", side_effect=lambda fx, s: np.array(
            [s["round"], s["step"]], dtype=np.float32)), \
         patch.object(module, "_reward", side_effect=lambda a, s, act, ns, ev: sum(rewards[e] for e in ev)) as reward, \
         patch.object(module, "optimize_model") as optimize, \
         patch.object(module.torch, "save") as save, \
         patch.object(module.Path, "replace"):
        yield module, runtime, reward, optimize, save


class TerminalRecordingTests(unittest.TestCase):
    def test_survivor_last_action_counted_once_and_terminal(self):
        for name in MODULES:
            with self.subTest(agent=name), trainer(name) as (m, a, reward, opt, save):
                events = ["COIN_COLLECTED"]
                m.game_events_occurred(a, state(400), "RIGHT", state(401), events)
                self.assertEqual(len(a.replay), 0)
                events.append("SURVIVED_ROUND")
                m.end_of_round(a, state(400), "RIGHT", events)
                self.assertEqual(a.environment_steps, 1)
                self.assertEqual(len(a.replay), 1)
                row = a.replay.storage[0]
                self.assertTrue(row.done)
                self.assertEqual(row.reward, 6.0)
                self.assertEqual(row.discount, m.GAMMA)
                self.assertFalse(row.next_mask.any())
                self.assertFalse(row.next_state.any())
                self.assertEqual(reward.call_count, 1)
                self.assertEqual(opt.call_count, 1)
                self.assertIsNone(a.pending_step)
                self.assertFalse(a.n_step_buffer)
                save.assert_called_once()

    def test_n_step_returns_flush_at_survivor_without_extra_step(self):
        for name in MODULES:
            with self.subTest(agent=name), trainer(name) as (m, a, reward, opt, save):
                for step in range(1, 8):
                    m.game_events_occurred(a, state(step), "RIGHT", state(step + 1), ["COIN_COLLECTED"])
                m.end_of_round(a, state(7), "RIGHT", ["COIN_COLLECTED", "SURVIVED_ROUND"])
                self.assertEqual(a.environment_steps, 7)
                self.assertEqual(len(a.replay), 7)
                self.assertEqual(reward.call_count, 7)
                self.assertEqual(opt.call_count, 7)
                for i, row in enumerate(a.replay.storage[:7]):
                    count = min(m.N_STEP, 7 - i)
                    expected = sum(m.GAMMA ** j * (6.0 if i + j == 6 else 4.0) for j in range(count))
                    self.assertAlmostEqual(row.reward, expected)
                    self.assertAlmostEqual(row.discount, m.GAMMA ** count)
                    self.assertEqual(row.done, i + count == 7)
                    self.assertEqual(row.state.tolist(), [1, i + 1])
                    if not row.done:
                        self.assertEqual(row.next_state.tolist(), [1, i + count + 1])
                self.assertFalse(a.n_step_buffer)

    def test_death_on_first_action_has_no_ordinary_callback(self):
        for name in MODULES:
            with self.subTest(agent=name), trainer(name) as (m, a, reward, opt, save):
                m.end_of_round(a, state(1), "WAIT", ["WAITED", "GOT_KILLED"])
                self.assertEqual(a.environment_steps, 1)
                self.assertEqual(len(a.replay), 1)
                self.assertTrue(a.replay.storage[0].done)
                self.assertAlmostEqual(a.replay.storage[0].reward, -15.15)

    def test_death_flushes_previous_pending_then_fatal_action(self):
        for name in MODULES:
            with self.subTest(agent=name), trainer(name) as (m, a, reward, opt, save):
                for step in range(1, 6):
                    m.game_events_occurred(a, state(step), "RIGHT", state(step + 1), ["COIN_COLLECTED"])
                m.end_of_round(a, state(6), "LEFT", ["GOT_KILLED"])
                self.assertEqual(a.environment_steps, 6)
                self.assertEqual(len(a.replay), 6)
                self.assertFalse(a.replay.storage[0].done)
                self.assertTrue(a.replay.storage[1].done)
                self.assertAlmostEqual(a.replay.storage[1].reward,
                    sum(4 * m.GAMMA ** j for j in range(4)) - 15 * m.GAMMA ** 4)
                self.assertEqual(a.replay.storage[5].action, m.ACTIONS.index("LEFT"))
                self.assertEqual(a.replay.storage[5].reward, -15)
                self.assertFalse(a.n_step_buffer)

    def test_events_are_copied_not_mutated_by_later_engine_events(self):
        for name in MODULES:
            with self.subTest(agent=name), trainer(name) as (m, a, reward, opt, save):
                events = ["COIN_COLLECTED"]
                m.game_events_occurred(a, state(1), "RIGHT", state(2), events)
                events.append("SURVIVED_ROUND")
                m.game_events_occurred(a, state(2), "WAIT", state(3), ["WAITED"])
                self.assertEqual(reward.call_args.args[-1], ["COIN_COLLECTED"])

    def test_two_rounds_do_not_share_pending_or_n_step_transitions(self):
        for name in MODULES:
            with self.subTest(agent=name), trainer(name) as (m, a, reward, opt, save):
                m.game_events_occurred(a, state(400), "WAIT", state(401), ["WAITED"])
                m.end_of_round(a, state(400), "WAIT", ["WAITED", "SURVIVED_ROUND"])
                m.end_of_round(a, state(1, 2), "WAIT", ["GOT_KILLED"])
                self.assertEqual(a.environment_steps, 2)
                self.assertEqual(len(a.replay), 2)
                self.assertEqual(a.replay.storage[0].state.tolist(), [1, 400])
                self.assertEqual(a.replay.storage[1].state.tolist(), [2, 1])
                self.assertAlmostEqual(a.replay.storage[0].reward, 1.85)
                self.assertEqual(a.replay.storage[1].reward, -15)

    def test_actual_engine_survivor_callback_sequence(self):
        for name in MODULES:
            with self.subTest(agent=name), trainer(name) as (m, a, reward, opt, save):
                events = ["COIN_COLLECTED"]
                agent = SimpleNamespace(train=True, dead=False,
                    process_game_events=lambda new: m.game_events_occurred(a, state(400), "RIGHT", new, events),
                    wait_for_game_event_processing=lambda: None,
                    add_event=events.append,
                    round_ended=lambda: m.end_of_round(a, state(400), "RIGHT", events))
                world = BombeRLeWorld.__new__(BombeRLeWorld)
                world.agents = world.active_agents = [agent]
                world.logger = Mock()
                world.round = 1
                world.args = SimpleNamespace(save_replay=False)
                world.get_state_for_agent = lambda agent: state(401)
                with patch.object(GenericWorld, "end_round"):
                    world.send_game_events()
                    world.end_round()
                self.assertEqual(a.environment_steps, 1)
                self.assertEqual(len(a.replay), 1)
                self.assertEqual(a.replay.storage[0].reward, 6)

    def test_actual_engine_death_callback_sequence(self):
        for name in MODULES:
            with self.subTest(agent=name), trainer(name) as (m, a, reward, opt, save):
                agent = SimpleNamespace(train=True, dead=True,
                    process_game_events=Mock(), wait_for_game_event_processing=Mock(),
                    round_ended=lambda: m.end_of_round(a, state(1), "WAIT", ["GOT_KILLED"]))
                world = BombeRLeWorld.__new__(BombeRLeWorld)
                world.agents, world.active_agents = [agent], []
                world.logger, world.round = Mock(), 1
                world.args = SimpleNamespace(save_replay=False)
                with patch.object(GenericWorld, "end_round"):
                    world.send_game_events()
                    world.end_round()
                agent.process_game_events.assert_not_called()
                self.assertEqual(a.environment_steps, 1)
                self.assertEqual(len(a.replay), 1)
                self.assertEqual(a.replay.storage[0].reward, -15)
