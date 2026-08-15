from pathlib import Path
import sys

source, target = map(Path, sys.argv[1:3])
data = source.read_bytes()
data = data.replace(b"\xef\xbb\xbf", b"")
data = data.replace("\ufeff".encode("utf-8"), b"")
target.write_bytes(data)
