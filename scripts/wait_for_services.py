"""Wait until model services report healthy (used by the launch scripts).

Polls GET /health on each port with retries/backoff. Exits 0 when all are
healthy, exits 1 after --timeout seconds (warning, never hangs forever).

Usage:
    venv\\Scripts\\python.exe scripts\\wait_for_services.py [port ...]
    (default: the 7 voting models 5001 5004 5005 5009 5010 5011 5013)
"""
import sys
import time
import urllib.request

DEFAULT_PORTS = [5001, 5004, 5005, 5009, 5010, 5011, 5013]
TIMEOUT_S = 600
RETRY_S = 10


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                    timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> int:
    ports = [int(p) for p in sys.argv[1:]] or DEFAULT_PORTS
    deadline = time.time() + TIMEOUT_S
    pending = list(ports)
    while pending:
        for port in list(pending):
            if healthy(port):
                print(f"  [OK] :{port} healthy", flush=True)
                pending.remove(port)
        if pending:
            if time.time() >= deadline:
                print(f"  [WARN] timed out waiting for: {pending} "
                      f"— continuing anyway", flush=True)
                return 1
            time.sleep(RETRY_S)
    print("  [OK] all services healthy", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
