"""Execute claimed proactive tasks through the real follow-up callback."""

import threading
import time
from datetime import timedelta

from database import tasks
from database.connection import current_database_path, using_database


class ProactiveEngine:
    def __init__(self, execute_task, *, lease_seconds=300, max_tasks_per_tick=20, patient_ids=None):
        if not callable(execute_task):
            raise ValueError("execute_task must be callable")
        tasks._positive(lease_seconds, "lease_seconds")
        tasks._positive(max_tasks_per_tick, "max_tasks_per_tick", 1000)
        self.execute_task = execute_task
        self.lease_seconds = lease_seconds
        self.max_tasks_per_tick = max_tasks_per_tick
        self.patient_ids = tasks._scope_ids(patient_ids)

    def tick(self, now=None):
        """One pass; supplying now advances a simulation without changing its clock."""
        if self.patient_ids == ():
            return {"generated": 0, "recovered": 0, "reconciled": 0,
                    "claimed": 0, "executed": 0, "results": []}
        clock = tasks.utc_datetime(now)
        generated = tasks.generate_due_tasks(clock, patient_ids=self.patient_ids)
        recovered = tasks.recover_expired_leases(clock, patient_ids=self.patient_ids)
        reconciled = tasks.reconcile_waiting_sessions(patient_ids=self.patient_ids)
        results = []
        for _ in range(self.max_tasks_per_tick):
            # Production callbacks may take time; simulated clocks stay deterministic.
            claim_time = tasks.utc_datetime() if now is None else clock
            task = tasks.claim_task(claim_time, lease_seconds=self.lease_seconds, patient_ids=self.patient_ids)
            if task is None:
                break
            # A heartbeat prevents another worker reclaiming a slow model invocation.
            stop = threading.Event()
            path = current_database_path()
            started = time.monotonic()

            def keep_lease(task=task, path=path, claim_time=claim_time, started=started, stop=stop):
                while not stop.wait(max(0.1, self.lease_seconds / 3)):
                    with using_database(path):
                        heartbeat_time = None if now is None else claim_time + timedelta(seconds=time.monotonic() - started)
                        if not tasks.renew_task_lease(task["task_id"], task["claim_token"], heartbeat_time,
                                                      lease_seconds=self.lease_seconds):
                            return

            worker = threading.Thread(target=keep_lease, daemon=True, name="proactive-lease")
            worker.start()
            result, error = None, None
            try:
                result = self.execute_task(task)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                stop.set()
                worker.join(timeout=1)
            finished = tasks.finish_task(task["task_id"], task["claim_token"], result, error,
                                         None if now is None else clock)
            results.append(finished or {"task_id": task["task_id"], "status": "lease_lost"})
        return {"generated": len(generated), "recovered": recovered, "reconciled": reconciled,
                "claimed": len(results), "executed": len(results), "results": results}
