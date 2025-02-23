import sys
import select
import platform

class KeyboardHandler:
    def __init__(self):
        self.platform = platform.system().lower()
        if self.platform == 'windows':
            import msvcrt
            self.msvcrt = msvcrt
        else:
            import termios
            import tty
            self.termios = termios
            self.tty = tty
            self.orig_settings = termios.tcgetattr(sys.stdin)

    def init(self):
        if self.platform != 'windows':
            self.tty.setcbreak(sys.stdin.fileno())

    def restore(self):
        if self.platform != 'windows':
            self.termios.tcsetattr(sys.stdin, self.termios.TCSADRAIN, self.orig_settings)

    def check_key(self):
        if self.platform == 'windows':
            if self.msvcrt.kbhit():
                key = self.msvcrt.getch()
                return key == b'\x1b'  # ESC key
        else:
            dr, _, _ = select.select([sys.stdin], [], [], 0)
            if dr:
                key = sys.stdin.read(1)
                return key == '\x1b'  # ESC key
        return False
