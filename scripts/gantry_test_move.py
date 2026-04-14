#!/usr/bin/env python3
"""Send a single relative move to the gantry Arduino and print its reply.

Usage:
  python3 scripts/gantry_test_move.py 50 50
  python3 scripts/gantry_test_move.py 50 50 --port /dev/ttyCH341USB0
"""

import argparse
import sys
import time
import serial


def main():
    parser = argparse.ArgumentParser(description="Send dx, dy (mm) to gantry")
    parser.add_argument("dx", type=float, help="X displacement in mm")
    parser.add_argument("dy", type=float, help="Y displacement in mm")
    parser.add_argument("--port", default="/dev/ttyCH341USB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--wait", type=float, default=5.0,
                        help="Seconds to listen for replies after sending")
    args = parser.parse_args()

    try:
        s = serial.Serial(args.port, args.baud, timeout=2)
    except serial.SerialException as e:
        print(f"ERROR: cannot open {args.port}: {e}")
        sys.exit(1)

    time.sleep(2)
    while s.in_waiting:
        s.readline()

    cmd = f"{args.dx:.2f}, {args.dy:.2f}\n"
    s.write(cmd.encode("utf-8"))
    print(f"sent {cmd.strip()}")

    deadline = time.time() + args.wait
    while time.time() < deadline:
        if s.in_waiting:
            line = s.readline().decode(errors="replace").strip()
            if line:
                print(line)
        else:
            time.sleep(0.05)

    s.close()


if __name__ == "__main__":
    main()
