"""Schedule query-dependent SparF stages on isolated FDM slots.

Unlike dense KV prefetch, every request arrives with its query. SELECT and
TOKEN_READ use the load worker; EXACT_V and PV use the compute worker. The CSD
itself owns selected indices and overlaps the V read with QK in EXACT_V.
Autonomous slots submit independent parents; CEMU owns engine admission and CU
arbitration. Guest executor width is bounded by the number of FDM slots.
"""
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from queue import Queue
from threading import Lock

from .cemu_attention_slot_scheduler import CemuAttentionSharedWorkers


class CemuSparfSlotScheduler:
    def __init__(self, slots, workers=None):
        self.slots = tuple(slots)
        if len(self.slots) < 2 or len({id(slot) for slot in self.slots}) != len(self.slots):
            raise ValueError("SparF requires at least two distinct slots")
        self.autonomous = all(getattr(slot, "autonomous", False) for slot in self.slots)
        if self.autonomous:
            if any(slot.staged for slot in self.slots):
                raise ValueError("autonomous slots cannot enable Guest stages")
        elif any(not slot.staged or getattr(slot, "autonomous", False) for slot in self.slots):
            raise ValueError("SparF slots must use the same staged or autonomous mode")
        self.workers = workers or CemuAttentionSharedWorkers()
        self._available = Queue()
        self._lock = Lock()
        self._users = 0
        self._accepting = False
        self._closing = False
        self._requests = set()
        self._autonomous_executor = None

    @property
    def is_open(self):
        with self._lock:
            return self._accepting

    def open(self):
        with self._lock:
            if self._closing:
                raise RuntimeError("SparF scheduler is closing")
            if self._users:
                self._users += 1
                return self
            opened = []
            try:
                for slot in self.slots:
                    slot.open()
                    opened.append(slot)
                self.workers.acquire()
            except Exception:
                for slot in reversed(opened):
                    slot.close()
                raise
            if self.autonomous:
                self._autonomous_executor = ThreadPoolExecutor(
                    max_workers=len(self.slots), thread_name_prefix="cemu-sparf-submit"
                )
            self._available = Queue()
            for slot in self.slots:
                self._available.put(slot)
            self._users = 1
            self._accepting = True
        return self

    def submit_decode(self, *args, **kwargs):
        # Reserve in the producer, never block the load worker on a slot whose
        # next load stage is waiting behind it in that same worker's queue.
        with self._lock:
            if not self._accepting:
                raise RuntimeError("SparF scheduler is not open")
            result = Future()
            result.set_running_or_notify_cancel()
            self._requests.add(result)
        slot = self._available.get()
        try:
            self.workers.submit_load(self._prepare, slot, result, args, kwargs)
        except Exception as error:
            self._finish(slot, result, error=error)
        return result

    def _prepare(self, slot, result, args, kwargs):
        request = None
        try:
            request = slot.prepare_decode(*args, **kwargs)
            if self.autonomous:
                self._autonomous_executor.submit(self._autonomous, slot, request, result)
                return
        except Exception as error:
            if request is not None:
                self._abort(slot, request)
            self._finish(slot, result, error=error)
            return
        self._advance(slot, request, result)

    def _autonomous(self, slot, request, result):
        try:
            slot.execute_autonomous(request)
            self.workers.submit_store(self._collect, slot, request, result)
        except Exception as error:
            self._abort(slot, request)
            self._finish(slot, result, error=error)

    def _advance(self, slot, request, result):
        try:
            slot.execute_stage(request)
            if request.completed_stages == 4:
                self.workers.submit_store(self._collect, slot, request, result)
            else:
                submit = (self.workers.submit_load if request.completed_stages < 2
                          else self.workers.submit_compute)
                submit(self._advance, slot, request, result)
        except Exception as error:
            self._abort(slot, request)
            self._finish(slot, result, error=error)

    def _collect(self, slot, request, result):
        try:
            output = slot.collect_decode(request)
        except Exception as error:
            self._abort(slot, request)
            self._finish(slot, result, error=error)
            return
        self._finish(slot, result, output=output)

    @staticmethod
    def _abort(slot, request):
        try:
            slot.abort_decode(request)
        except RuntimeError:
            # A failed execute/collect already releases the lease.
            pass

    def _finish(self, slot, result, output=None, error=None):
        self._available.put(slot)
        if error is None:
            result.set_result(output)
        else:
            result.set_exception(error)
        with self._lock:
            self._requests.discard(result)

    def run_decode(self, *args, **kwargs):
        return self.submit_decode(*args, **kwargs).result()

    def modeled_runtime_summary(self):
        summaries = [slot.modeled_runtime_summary() for slot in self.slots]
        return sum(s[0] for s in summaries), sum(s[1] for s in summaries)

    def close(self):
        with self._lock:
            if not self._users:
                return
            self._users -= 1
            if self._users:
                return
            self._accepting = False
            self._closing = True
            requests = tuple(self._requests)
        try:
            wait(requests)
            if self._autonomous_executor is not None:
                self._autonomous_executor.shutdown(wait=True)
                self._autonomous_executor = None
            # Drain workers before removing their FDM ranges. ExitStack also
            # closes the remaining slots if one device's teardown fails.
            with ExitStack() as cleanup:
                for slot in self.slots:
                    cleanup.callback(slot.close)
                cleanup.callback(self.workers.release)
        finally:
            with self._lock:
                self._closing = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
