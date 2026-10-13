"""Two-agent runs through real proactive tasks, reports and station delivery."""
import copy
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone

from agent.state_manager import add_trace
from agent.trace_view import present_trace
from backend.proactive import ProactiveEngine
from backend.runtime_v5 import AgentRuntime, TERMINAL
from database import repository, collaboration, tasks
from database.connection import init_db, using_database
from database.simulation import SimulationStore
from simulation.patient import get_scenario, SimulatedPatient
from simulation.scripted import ScriptedDemoModel


class SimulationManager:
    def __init__(self, root):
        self.store = SimulationStore(root)
        self._jobs = set()
        self._guard = threading.Lock()
        self._slots = threading.BoundedSemaphore(4)

    def create(self, scenario_id, *, model_mode="live", trigger_type=None, max_turns=20, delay_seconds=1, background=True):
        scenario = get_scenario(scenario_id)
        if model_mode not in {"live", "scripted"}:
            raise ValueError("model_mode must be live or scripted")
        if not 1 <= max_turns <= 30 or not 0 <= delay_seconds <= 60:
            raise ValueError("Invalid simulation budget")
        trigger = trigger_type or scenario["trigger_type"]
        if trigger not in {"scheduled", "high_bp", "missed_measurement"}:
            raise ValueError("Unknown trigger")
        if trigger == "high_bp" and scenario["hidden_facts"]["bp_systolic"] < 160 and scenario["hidden_facts"]["bp_diastolic"] < 100:
            raise ValueError("This fictional patient has no abnormal BP event at the existing trigger threshold")
        options = {"model_mode": model_mode, "trigger_type": trigger, "max_turns": max_turns, "delay_seconds": delay_seconds}
        payload, run_token = self.store.create(scenario_id, options)
        with using_database(self.store.database(payload["run_id"])):
            init_db()
            repository.seed_demo_patient(scenario["patient"], scenario["hidden_facts"]["bp_history"])
            tokens = {}
            for role in ("doctor", "family"):
                user_id = "simulation-" + role
                collaboration.provision_user(user_id, role)
                collaboration.link_patient(user_id, scenario["patient"]["patient_id"])
                tokens[role + "_token"] = collaboration.issue_token(user_id)
            due = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
            if trigger == "high_bp":
                values = scenario["hidden_facts"]
                event = tasks.record_bp_event(scenario["patient"]["patient_id"], "simulation-event", values["bp_systolic"], values["bp_diastolic"], source="simulation_device")
                payload["task_id"] = event["task"]["task_id"]
            else:
                plan = tasks.create_plan(scenario["patient"]["patient_id"], trigger, due.isoformat(),
                                         interval_seconds=86400, window_seconds=60)
                payload["plan_id"] = plan["plan_id"]
            payload["due_at"] = due.isoformat()
        self.store.save(payload)
        if background:
            self.launch(payload["run_id"])
        return {**payload, "run_token": run_token, **tokens}

    def launch(self, run_id):
        with self._guard:
            if run_id in self._jobs:
                return
            self._jobs.add(run_id)
        def work():
            try:
                with self._slots:
                    self.execute(run_id)
            finally:
                with self._guard:
                    self._jobs.discard(run_id)
        threading.Thread(target=work, daemon=True, name="simulated-followup").start()

    def execute(self, run_id, *, model_override=None):
        owner = str(uuid.uuid4())
        if not self.store.claim(run_id, owner):
            return self.store.get(run_id, trusted=True)
        payload = self.store.get(run_id, trusted=True)
        stop = threading.Event()
        # Maintain run ownership even during slow model calls; restart can reclaim abandoned runs.
        def heartbeat():
            while not stop.wait(30):
                self.store.renew(run_id, owner)
        threading.Thread(target=heartbeat, daemon=True, name="simulation-lease").start()
        try:
            with using_database(self.store.database(run_id)):
                scenario = get_scenario(payload["scenario_id"])
                if payload["options"]["trigger_type"] == "missed_measurement":
                    scenario["trigger_type"] = "missed_measurement"
                    scenario["required_topics"] = ["medication", "current_symptoms", "red_flags", "monitoring_plan"]
                model = model_override or (ScriptedDemoModel() if payload["options"]["model_mode"] == "scripted" else None)
                runtime = AgentRuntime(model)
                payload["status"] = "running"
                self.store.save(payload, owner=owner)
                engine = ProactiveEngine(runtime.start_task, max_tasks_per_tick=1)
                if not payload.get("session_id"):
                    # Explicit simulation clock; real plans use wall time in the service worker.
                    engine.tick(max(tasks.utc_datetime(payload["due_at"]), datetime.now(timezone.utc)))
                    created = tasks.list_tasks(scenario["patient"]["patient_id"])
                    if not created:
                        raise RuntimeError("Proactive trigger did not create a task")
                    task = created[0]
                    payload.update(task_id=task["task_id"], session_id=task["session_id"])
                    self.store.save(payload, owner=owner)
                    # Simulation advances only its task clock for finite persisted retries.
                    while task["status"] == "retry":
                        engine.tick(max(tasks.utc_datetime(task["due_at"]), datetime.now(timezone.utc)))
                        task = tasks.get_task(task["task_id"])
                    if task["status"] not in {"waiting_for_user", "completed"}:
                        raise RuntimeError("Proactive Agent execution failed: " + str(task.get("last_error")))
                    self.store.save(payload, owner=owner)
                sid = payload["session_id"]
                task = tasks.get_task(payload["task_id"])
                if task["status"] in {"queued", "retry", "running"}:
                    engine.tick(max(tasks.utc_datetime(task["due_at"]), datetime.now(timezone.utc)))
                    task = tasks.get_task(task["task_id"])
                    if task["status"] not in {"waiting_for_user", "completed"}:
                        raise RuntimeError("Proactive execution remains pending")
                patient = SimulatedPatient(payload["scenario_id"], model_call=lambda purpose, prompt, data:
                                           runtime.model_call(call_state, purpose, prompt, data))
                patient_turns = sum(t.get("role") == "patient" and t.get("execution") == "completed" for t in payload["transcript"])
                for index in range(patient_turns, payload["options"]["max_turns"]):
                    state = repository.read_session(sid)
                    waiting = payload["transcript"][-1] if payload["transcript"] else {}
                    if waiting.get("role") == "patient" and waiting.get("execution") == "pending":
                        runtime.turn(sid, waiting["message"], turn_index=index)
                        waiting["execution"] = "completed"
                        self.store.save(payload, owner=owner)
                        continue
                    if state["followup_status"] in TERMINAL:
                        break
                    if state["followup_status"] == "processing":
                        state = runtime.turn(sid)
                    if state["followup_status"] != "waiting_for_user":
                        break
                    question = state.get("last_agent_response", "")
                    context = copy.deepcopy(state.get("question_context") or {})
                    # A persisted patient entry awaiting execution is replayed once after restart.
                    waiting = payload["transcript"][-1] if payload["transcript"] else {}
                    if waiting.get("role") == "patient" and waiting.get("execution") == "pending":
                        answer = waiting
                    else:
                        payload["transcript"].append({"role": "agent", "message": question, "question_context": context})
                        call_state = copy.deepcopy(state)
                        answer = patient.answer(question, context, turn_index=index)
                        for entry in call_state["trace"][len(state["trace"]):]:
                            state["trace"].append(entry)
                        add_trace(state, "SIMULATED_PATIENT", {"mode": answer["mode"], "model_status": answer["model_status"]})
                        repository.persist_session(sid, state)
                        payload["transcript"].append({"role": "patient", **answer, "execution": "pending"})
                        self.store.save(payload, owner=owner)
                    # Durable turn index prevents a crash replay duplicating BP observations.
                    state = repository.read_session(sid)
                    if state.get("simulation_turn_index", -1) < index:
                        state = runtime.turn(sid, answer["message"], turn_index=index)
                    payload["transcript"][-1]["execution"] = "completed"
                    self.store.save(payload, owner=owner)
                state = runtime.finalize(sid) or repository.read_session(sid)
                if not payload["transcript"] or payload["transcript"][-1].get("role") != "agent":
                    payload["transcript"].append({"role": "agent", "message": state.get("last_agent_response", ""),
                                                   "question_context": state.get("question_context", {})})
                from simulation.evaluation import evaluate_run
                payload.update(status="completed" if state["followup_status"] in TERMINAL else
                               "failed" if state["followup_status"] == "interrupted" else "budget_exhausted",
                               followup_status=state["followup_status"], evaluation=evaluate_run(scenario, payload["transcript"], state))
                result = runtime.task_result(sid)
                tasks.sync_session_status(sid, result["status"], report_generated=result["report_generated"], station_messages=result["station_messages"])
                payload["task"] = tasks.get_task(payload["task_id"])
        except Exception as error:
            payload.update(status="failed", error_type=type(error).__name__, error="本次模拟执行未完成；记录已保存，请核对运行状态后重新开始。")
        finally:
            stop.set()
            self.store.save(payload, owner=owner, release=True)
        return payload

    def view(self, run_id, token):
        payload = self.store.get(run_id, token)
        if payload.get("session_id"):
            with using_database(self.store.database(run_id)):
                state = repository.read_session(payload["session_id"])
                payload["trace"] = present_trace(state or {}, fictional=True)
                payload["followup_status"] = (state or {}).get("followup_status")
                payload["task"] = tasks.get_task(payload["task_id"])
        payload["real_model_requests"] = sum(e["detail"].get("provider_request_count", 0) for e in payload.get("trace", []))
        return payload

    def recover(self):
        for payload in self.store.pending():
            self.launch(payload["run_id"])
