"""
Gamepad emulator for rl_basic_example via keyboard.

Publishes to /hardware/gamepad_keys (interface_protocol/msg/GamepadKeys).

Controls:
  WASD - move (W/S = forward/back -> LEFT_STICK_X, A/D = left/right -> LEFT_STICK_Y)
  QE   - rotate (Q = left, E = right -> RIGHT_STICK_Y)
"""

import rclpy
from rclpy.node import Node
from interface_protocol.msg import GamepadKeys
from std_msgs.msg import Header
from pynput import keyboard
import sys
import termios


class GamepadEmulator(Node):
    def __init__(self):
        super().__init__('gamepad_emulator')

        self.publisher = self.create_publisher(GamepadKeys, '/hardware/gamepad_keys', 10)

        # Analog stick states
        self.keys_pressed = {
            'w': False, 's': False,  # LEFT_STICK_Y (forward/back)
            'a': False, 'd': False,  # LEFT_STICK_X (left/right)
            'q': False, 'e': False,  # RIGHT_STICK_Y (yaw)
        }

        self.max_velocity = 1.0
        self.publish_rate = 20.0

        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_gamepad)

        self.keyboard_listener = None
        self.old_terminal_settings = None
        self.disable_terminal_echo()

        self.get_logger().info("Gamepad emulator started. Press Ctrl+C to exit.")
        self.get_logger().info("Controls: WASD (move), QE (rotate)")

    def on_key_press(self, key):
        try:
            key_char = key.char.lower()
            if key_char in self.keys_pressed:
                self.keys_pressed[key_char] = True
        except AttributeError:
            pass

    def on_key_release(self, key):
        try:
            key_char = key.char.lower()
            if key_char in self.keys_pressed:
                self.keys_pressed[key_char] = False
        except AttributeError:
            pass

    def publish_gamepad(self):
        msg = GamepadKeys()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'gamepad_emulator'
        msg.hardware_connected = True

        msg.digital_states = [0] * 12
        # analog_states: fixed size 6
        # indices: LT=0, RT=1, LEFT_STICK_X=2, LEFT_STICK_Y=3, RIGHT_STICK_X=4, RIGHT_STICK_Y=5
        msg.analog_states = [0.0] * 6

        # LEFT_STICK_X (index 2): W = forward (+1), S = back (-1)
        if self.keys_pressed['w']:
            msg.analog_states[GamepadKeys.LEFT_STICK_X] = self.max_velocity
        elif self.keys_pressed['s']:
            msg.analog_states[GamepadKeys.LEFT_STICK_X] = -self.max_velocity

        # LEFT_STICK_Y (index 3): A = left (+1), D = right (-1)
        if self.keys_pressed['a']:
            msg.analog_states[GamepadKeys.LEFT_STICK_Y] = self.max_velocity
        elif self.keys_pressed['d']:
            msg.analog_states[GamepadKeys.LEFT_STICK_Y] = -self.max_velocity

        # RIGHT_STICK_Y (index 5): Q = left (+1), E = right (-1)
        if self.keys_pressed['q']:
            msg.analog_states[GamepadKeys.RIGHT_STICK_Y] = self.max_velocity
        elif self.keys_pressed['e']:
            msg.analog_states[GamepadKeys.RIGHT_STICK_Y] = -self.max_velocity

        self.publisher.publish(msg)

    def start_keyboard_listener(self):
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.keyboard_listener.start()

    def stop_keyboard_listener(self):
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        self.enable_terminal_echo()

    def disable_terminal_echo(self):
        try:
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            new_settings = termios.tcgetattr(sys.stdin)
            new_settings[3] = new_settings[3] & ~termios.ECHO
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)
        except (termios.error, AttributeError, OSError):
            self.old_terminal_settings = None

    def enable_terminal_echo(self):
        if self.old_terminal_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
            except (termios.error, AttributeError):
                pass


def main(args=None):
    rclpy.init(args=args)

    node = GamepadEmulator()
    node.start_keyboard_listener()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.stop_keyboard_listener()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
