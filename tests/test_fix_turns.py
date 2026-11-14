"""The gripper turn check in launch/fix_turns.py, with a pretend servo bus (no hardware)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "launch"))
import fix_turns  # noqa: E402


class FakeBus:
    def __init__(self, gripper):
        self.gripper, self.rebooted = gripper, False

    def read32(self, servo, address):
        return self.gripper

    def send(self, servo, instruction, params):
        if servo == fix_turns.GRIPPER and instruction == 0x08:     # Reboot: back to the real single turn
            self.rebooted, self.gripper = True, self.gripper % 4096


class GripperTurnTest(unittest.TestCase):
    def test_a_turn_off_is_reset_by_a_restart(self):
        bus = FakeBus(5168)                      # what the real gripper read, open
        fix_turns.time.sleep = lambda s: None
        fix_turns.check_gripper(bus, check_only=False)
        self.assertTrue(bus.rebooted)
        self.assertEqual(bus.gripper, 1072)

    def test_a_good_reading_or_check_only_changes_nothing(self):
        for reading, check in ((1072, False), (2682, False), (5168, True)):
            bus = FakeBus(reading)
            fix_turns.check_gripper(bus, check_only=check)
            self.assertFalse(bus.rebooted)


if __name__ == "__main__":
    unittest.main()
