import sys
import subprocess

args = sys.argv[1:]

new_args = []
has_libx264 = False
for arg in args:
    if "libx264" in arg:
        has_libx264 = True
    new_args.append(arg)

if has_libx264:
    # Insert crop filter to make width/height divisible by 2
    new_args.insert(-1, "-vf")
    new_args.insert(-1, "crop=trunc(iw/2)*2:trunc(ih/2)*2")

# Run the real ffmpeg binary, forwarding stdin, stdout, and stderr
process = subprocess.Popen(
    ["ffmpeg_bin.exe"] + new_args,
    stdin=subprocess.PIPE,
    stdout=sys.stdout,
    stderr=sys.stderr
)

try:
    while True:
        chunk = sys.stdin.buffer.read(65536)  # Use 64KB buffer for speed
        if not chunk:
            break
        process.stdin.write(chunk)
except Exception:
    pass

try:
    process.stdin.close()
except Exception:
    pass

process.wait()
sys.exit(process.returncode)
