# memory/__init__.py
import sys, pathlib
pkg_dir = pathlib.Path(__file__).parent
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))
