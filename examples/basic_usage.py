from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from xsprops import i_beam, props, pretty


def main() -> None:
    section = i_beam(b=200, h=300, tw=8, tf=12, r1=10)
    print(pretty(props(section), n=2))


if __name__ == "__main__":
    main()
