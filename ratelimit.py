"""Protects the paid endpoints (/research, /teardown) from being run in a loop.

Two limits, both counted in memory (no database needed):
  - per visitor:  RATE_LIMIT_PER_HOUR runs per IP address in any rolling hour (default 10)
  - whole app:    DAILY_RUN_LIMIT runs per rolling 24 hours across ALL visitors (default 200),
                  so many different IP addresses cannot drain the credits either.
Set either to 0 to switch it off. Counts reset when the server restarts.
"""
import math
import os
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Optional, Tuple

from fastapi import HTTPException, Request


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, default)))
    except ValueError:
        return default


HOUR = 3600
DAY = 24 * HOUR


class RateLimiter:
    def __init__(self, per_ip_per_hour: int, daily_total: int, clock: Callable[[], float] = time.monotonic):
        self.per_ip_per_hour = per_ip_per_hour
        self.daily_total = daily_total
        self.clock = clock
        self._lock = threading.Lock()
        self._by_ip: Dict[str, Deque[float]] = {}
        self._all: Deque[float] = deque()

    @staticmethod
    def _trim(times: Deque[float], window: float, now: float) -> None:
        while times and now - times[0] >= window:
            times.popleft()

    def check_and_record(self, ip: str) -> Tuple[bool, Optional[str], int]:
        """Returns (allowed, message, retry_after_seconds). Records the run only if allowed."""
        now = self.clock()
        with self._lock:
            self._trim(self._all, DAY, now)
            if self.daily_total and len(self._all) >= self.daily_total:
                wait = math.ceil(DAY - (now - self._all[0]))
                return False, (
                    "The research agent has reached its daily limit. "
                    f"Please try again in about {_human(wait)}."
                ), wait

            times = self._by_ip.setdefault(ip, deque())
            self._trim(times, HOUR, now)
            if self.per_ip_per_hour and len(times) >= self.per_ip_per_hour:
                wait = math.ceil(HOUR - (now - times[0]))
                return False, (
                    f"You've reached the limit of {self.per_ip_per_hour} reports per hour. "
                    f"Please try again in about {_human(wait)}."
                ), wait

            times.append(now)
            self._all.append(now)
            # Forget visitors with no recent runs so memory does not grow forever.
            if len(self._by_ip) > 10_000:
                for key in [k for k, v in self._by_ip.items() if not v or now - v[-1] >= HOUR]:
                    del self._by_ip[key]
            return True, None, 0


def _human(seconds: int) -> str:
    minutes = max(1, math.ceil(seconds / 60))
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = math.ceil(minutes / 60)
    return f"{hours} hour{'s' if hours != 1 else ''}"


# Render (and most hosts) put the visitor's real IP as the LAST entry of
# X-Forwarded-For; anything before it can be typed in by the visitor, so it is
# not trusted. PROXY_HOPS = how many proxies sit in front of the app (Render: 1).
PROXY_HOPS = _env_int("PROXY_HOPS", 1)


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    hops = [h.strip() for h in forwarded.split(",") if h.strip()]
    if PROXY_HOPS and len(hops) >= PROXY_HOPS:
        return hops[-PROXY_HOPS]
    return request.client.host if request.client else "unknown"


limiter = RateLimiter(
    per_ip_per_hour=_env_int("RATE_LIMIT_PER_HOUR", 10),
    daily_total=_env_int("DAILY_RUN_LIMIT", 200),
)


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency for the paid endpoints. Runs after the request body is
    validated, so rejected (invalid) requests never use up anyone's quota."""
    allowed, message, retry_after = limiter.check_and_record(client_ip(request))
    if not allowed:
        raise HTTPException(status_code=429, detail=message, headers={"Retry-After": str(retry_after)})
