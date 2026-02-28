#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests


# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class Config:
    model: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://172.27.176.1:11434").rstrip("/")

    seed: int = int(os.getenv("SEED", "42"))
    max_ticks: int = int(os.getenv("MAX_TICKS", "60"))

    # tick pacing
    tick_timeout_s: int = int(os.getenv("TICK_TIMEOUT_S", "70"))

    # LLM timeouts (read timeout must be close to tick)
    ollama_connect_timeout_s: float = float(os.getenv("OLLAMA_CONNECT_TIMEOUT_S", "3"))
    ollama_req_timeout_s: int = int(os.getenv("OLLAMA_REQ_TIMEOUT_S", "68"))

    # model options
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.25"))
    num_predict: int = int(os.getenv("TICK_NUM_PREDICT", "20"))
    num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "1024"))

    # prompt limits
    market_limit: int = int(os.getenv("PROMPT_MARKET_LIMIT", "8"))
    loans_limit: int = int(os.getenv("PROMPT_LOANS_LIMIT", "8"))
    book_limit: int = int(os.getenv("PROMPT_BOOK_LIMIT", "8"))
    board_tail: int = int(os.getenv("PROMPT_BOARD_TAIL", "10"))

    # economy knobs
    market_size_cap: int = int(os.getenv("MARKET_SIZE_CAP", "14"))
    new_tasks_per_tick: int = int(os.getenv("NEW_TASKS_PER_TICK", "3"))

    # LLM policy
    llm_every_n_ticks: int = int(os.getenv("LLM_EVERY_N_TICKS", "2"))  # не дёргать LLM каждый тик

    log_csv: str = os.getenv("LOG_CSV", "run_log.csv")


CFG = Config()
random.seed(CFG.seed)


# =========================
# UTIL
# =========================
def now_s() -> float:
    return time.time()


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def gini(values: List[int]) -> float:
    vals = [v for v in values if v >= 0]
    if not vals:
        return 0.0
    vals.sort()
    n = len(vals)
    s = sum(vals)
    if s == 0:
        return 0.0
    cum = 0
    for i, v in enumerate(vals, 1):
        cum += i * v
    return (2 * cum) / (n * s) - (n + 1) / n


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            obj = json.loads(text[a : b + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


# =========================
# HTTP / OLLAMA
# =========================
def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


SESSION = build_session()


class OllamaClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, *, timeout_s: int) -> Tuple[str, str]:
        """
        Returns: (text, status)
        status in: ok / http_5xx / http_4xx / timeout / conn / other
        ВАЖНО: никогда не кидает исключение наружу — чтобы симуляция не падала.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": CFG.temperature,
                "num_predict": CFG.num_predict,
                "num_ctx": CFG.num_ctx,
                "stop": ["\n\n", "```", "\r\n\r\n"],
            },
        }

        try:
            r = SESSION.post(
                url,
                json=payload,
                timeout=(CFG.ollama_connect_timeout_s, float(timeout_s)),
            )
            if r.status_code >= 500:
                return "", "http_5xx"
            if r.status_code >= 400:
                return "", "http_4xx"
            data = r.json()
            return (data.get("response") or "").strip(), "ok"
        except requests.exceptions.ReadTimeout:
            return "", "timeout"
        except requests.exceptions.ConnectTimeout:
            return "", "timeout"
        except requests.exceptions.ConnectionError:
            return "", "conn"
        except requests.exceptions.RequestException:
            return "", "other"
        except Exception:
            return "", "other"


OLLAMA = OllamaClient(CFG.base_url, CFG.model)


# =========================
# MODELS
# =========================
@dataclass
class Task:
    task_id: str
    tier: str  # small/mid/growth/jackpot
    description: str
    cost: int
    reward: int
    ttl: int

    @property
    def roi(self) -> float:
        return self.reward / max(1, self.cost)


@dataclass
class Loan:
    loan_id: str
    lender_id: int
    borrower_id: int
    principal: int
    rate_pct: int  # процент (например 12 означает 12%)
    interest: int  # уже посчитанная сумма процентов в токенах
    due_tick: int
    status: str = "open"

    def total_due(self) -> int:
        return self.principal + self.interest


@dataclass
class BorrowOrder:
    order_id: str
    borrower_id: int
    amount: int
    max_rate_pct: int  # готов платить до N%
    created_tick: int
    ttl: int = 4
    status: str = "open"


@dataclass
class LendOrder:
    order_id: str
    lender_id: int
    amount: int
    min_rate_pct: int  # минимум N%
    due_in: int
    created_tick: int
    ttl: int = 4
    status: str = "open"


@dataclass
class MemoryItem:
    tick: int
    action: str
    outcome: str  # ok / rejected / err / filled / expired
    note: str = ""


@dataclass
class AgentState:
    agent_id: int
    role: str  # agent / bank

    budget: int
    score: int = 0
    reputation: int = 0

    # personality knobs
    risk: float = 0.5  # 0..1 (высокий риск -> охотится за jackpot)
    debt_aversion: float = 0.5  # 0..1 (высокая -> раньше возвращает, меньше занимает)

    # LLM health
    llm_cooldown: int = 0

    memory: List[MemoryItem] = field(default_factory=list)

    def short(self) -> str:
        tag = "BANK" if self.role == "bank" else "AG"
        return f"A{self.agent_id}({tag}) b={self.budget} s={self.score} rep={self.reputation}"


# =========================
# TASK MARKET
# =========================
TASK_COUNTER = 0


def gen_task() -> Task:
    """
    Специально для "монополии":
    - small: дешёвые, всегда доступны (экономика не умирает)
    - mid: нормальные
    - growth: вкусные, но часто требуют кредита
    - jackpot: очень вкусные, почти всегда требуют кредита, ttl короткий, но не 2
    """
    global TASK_COUNTER
    TASK_COUNTER += 1
    tid = f"t{TASK_COUNTER}"

    roll = random.random()

    # JACKPOT ~14% (кредит почти обязателен)
    if roll < 0.14:
        cost = random.choice([130, 150, 170, 190])
        reward = random.choice([420, 480, 540, 600])
        return Task(
            task_id=tid,
            tier="jackpot",
            description="JACKPOT: Объясни RAG очень просто (1–2 предложения).",
            cost=cost,
            reward=reward,
            ttl=5,
        )

    # GROWTH ~26% (часто нужен кредит)
    if roll < 0.40:
        cost = random.choice([60, 70, 80, 90])
        reward = random.choice([130, 150, 170, 190])
        return Task(
            task_id=tid,
            tier="growth",
            description="GROWTH: В 2 предложения: чем pytest лучше unittest?",
            cost=cost,
            reward=reward,
            ttl=6,
        )

    # MID ~35%
    if roll < 0.75:
        return Task(
            task_id=tid,
            tier="mid",
            description="Кратко: разница GET и POST (1–2 предложения).",
            cost=random.choice([14, 16, 18]),
            reward=random.choice([28, 32, 36]),
            ttl=7,
        )

    # SMALL остальное
    return Task(
        task_id=tid,
        tier="small",
        description="В 1 строку: что такое HTTP?",
        cost=random.choice([4, 5, 6]),
        reward=random.choice([7, 8, 9]),
        ttl=8,
    )


def market_view(tasks: List[Task], limit: int) -> str:
    if not tasks:
        return "(пусто)"
    lines = []
    for t in tasks[:limit]:
        lines.append(f"{t.task_id} [{t.tier}] c={t.cost} r={t.reward} ttl={t.ttl} roi={t.roi:.2f} | {t.description}")
    return "\n".join(lines)


# =========================
# CREDIT MARKET
# =========================
LOAN_COUNTER = 0
BORROW_COUNTER = 0
LEND_COUNTER = 0


def find_agent(agents: List[AgentState], agent_id: int) -> Optional[AgentState]:
    return next((a for a in agents if a.agent_id == agent_id), None)


def new_borrow_order(borrower_id: int, amount: int, max_rate_pct: int, tick: int, ttl: int) -> BorrowOrder:
    global BORROW_COUNTER
    BORROW_COUNTER += 1
    return BorrowOrder(
        order_id=f"bor{BORROW_COUNTER}",
        borrower_id=borrower_id,
        amount=amount,
        max_rate_pct=clamp(max_rate_pct, 0, 40),
        created_tick=tick,
        ttl=clamp(ttl, 2, 8),
    )


def new_lend_order(lender_id: int, amount: int, min_rate_pct: int, due_in: int, tick: int, ttl: int) -> LendOrder:
    global LEND_COUNTER
    LEND_COUNTER += 1
    return LendOrder(
        order_id=f"len{LEND_COUNTER}",
        lender_id=lender_id,
        amount=amount,
        min_rate_pct=clamp(min_rate_pct, 0, 40),
        due_in=clamp(due_in, 2, 9),
        created_tick=tick,
        ttl=clamp(ttl, 2, 8),
    )


def create_loan(lender: AgentState, borrower: AgentState, amount: int, rate_pct: int, due_tick: int) -> Loan:
    global LOAN_COUNTER
    LOAN_COUNTER += 1
    interest = int(math.ceil(amount * (rate_pct / 100.0)))
    return Loan(
        loan_id=f"loan{LOAN_COUNTER}",
        lender_id=lender.agent_id,
        borrower_id=borrower.agent_id,
        principal=amount,
        rate_pct=rate_pct,
        interest=interest,
        due_tick=due_tick,
        status="open",
    )


def credit_book_view(borrows: List[BorrowOrder], lends: List[LendOrder], limit: int) -> str:
    b = [x for x in borrows if x.status == "open" and x.amount > 0][:limit]
    l = [x for x in lends if x.status == "open" and x.amount > 0][:limit]
    lines = ["BORROW:"]
    lines += [f"- {x.order_id}: b{x.borrower_id} amt={x.amount} max%={x.max_rate_pct} ttl={x.ttl}" for x in b] or [
        "- (нет)"
    ]
    lines.append("LEND:")
    lines += [
        f"- {x.order_id}: l{x.lender_id} amt={x.amount} min%={x.min_rate_pct} due_in={x.due_in} ttl={x.ttl}" for x in l
    ] or ["- (нет)"]
    return "\n".join(lines)


def expire_orders(tick: int, borrows: List[BorrowOrder], lends: List[LendOrder], board: List[str]) -> None:
    for o in borrows:
        if o.status != "open":
            continue
        o.ttl -= 1
        if o.ttl <= 0:
            o.status = "expired"
            board.append(f"Тик {tick}: {o.order_id} expired (borrow).")
    for o in lends:
        if o.status != "open":
            continue
        o.ttl -= 1
        if o.ttl <= 0:
            o.status = "expired"
            board.append(f"Тик {tick}: {o.order_id} expired (lend).")


def clear_credit_market(
    *,
    tick: int,
    agents: List[AgentState],
    borrows: List[BorrowOrder],
    lends: List[LendOrder],
    loans: List[Loan],
    board: List[str],
) -> Tuple[int, int]:
    deals = 0
    vol = 0

    open_b = [b for b in borrows if b.status == "open" and b.amount > 0]
    open_l = [lo for lo in lends if lo.status == "open" and lo.amount > 0]

    # need first, cheapest rate first
    open_b.sort(key=lambda x: (-x.amount, x.created_tick))
    open_l.sort(key=lambda x: (x.min_rate_pct, x.created_tick))

    for b in open_b:
        borrower = find_agent(agents, b.borrower_id)
        if not borrower:
            b.status = "expired"
            continue

        for l in open_l:
            if l.status != "open":
                continue
            lender = find_agent(agents, l.lender_id)
            if not lender:
                l.status = "expired"
                continue
            if lender.budget <= 0:
                continue

            if b.max_rate_pct > 0 and l.min_rate_pct > b.max_rate_pct:
                continue

            amt = min(b.amount, l.amount, lender.budget)
            if amt <= 0:
                continue

            rate = l.min_rate_pct if b.max_rate_pct <= 0 else min(l.min_rate_pct, b.max_rate_pct)
            due_tick = tick + l.due_in

            lender.budget -= amt
            borrower.budget += amt

            loan = create_loan(lender, borrower, amt, rate, due_tick)
            loans.append(loan)

            b.amount -= amt
            l.amount -= amt
            deals += 1
            vol += amt

            board.append(
                f"Тик {tick}: CREDIT {loan.loan_id} l#{lender.agent_id}->b#{borrower.agent_id} amt={amt} rate={rate}% int={loan.interest} due={due_tick}"
            )

            if b.amount <= 0:
                b.status = "filled"
            if l.amount <= 0:
                l.status = "filled"

            if b.status == "filled":
                break

    return deals, vol


# =========================
# LOANS / MARKET EXPIRY
# =========================
def expire_market_and_loans(
    tick: int,
    market: List[Task],
    loans: List[Loan],
    agents: List[AgentState],
    board: List[str],
) -> int:
    defaults_added = 0

    for t in list(market):
        t.ttl -= 1
        if t.ttl <= 0:
            market.remove(t)
            board.append(f"Тик {tick}: {t.task_id} expired (task).")

    for l in loans:
        if l.status == "open" and tick > l.due_tick:
            l.status = "defaulted"
            defaults_added += 1
            borrower = find_agent(agents, l.borrower_id)
            lender = find_agent(agents, l.lender_id)
            if borrower:
                borrower.reputation -= 8
                borrower.budget = max(0, borrower.budget - int(l.principal * 0.2))  # штраф на бюджет
            if lender:
                lender.reputation -= 3
            board.append(f"Тик {tick}: {l.loan_id} DEFAULT (b{l.borrower_id}).")

    return defaults_added


def count_defaults(loans: List[Loan]) -> int:
    return sum(1 for l in loans if l.status == "defaulted")


# =========================
# ACTIONS / PROMPT
# =========================
PERSONAS: Dict[str, str] = {
    "agent": "Ты агент в экономической игре. Цель: максимизировать score. Кредиты можно брать и давать, но дефолт = плохо.",
    "bank": "Ты банк/кредитор. Цель: заработать на процентах и поддерживать рынок ликвидностью. Избегай дефолтов.",
}

ACTION_SCHEMA = """
Верни ТОЛЬКО JSON (без текста). Один action:
1) {"action":"solve","task_id":"t1","answer":"..."}
2) {"action":"borrow","amount":120,"max_rate_pct":15,"ttl":4}
3) {"action":"lend","amount":120,"min_rate_pct":10,"due_in":4,"ttl":4}
4) {"action":"repay","loan_id":"loan1"}
5) {"action":"post","message":"..."}
6) {"action":"skip","reason":"..."}
""".strip()


def normalize_action(obj: Dict[str, Any]) -> Dict[str, Any]:
    act = str(obj.get("action", "")).strip().lower()

    if act == "solve":
        return {
            "action": "solve",
            "task_id": str(obj.get("task_id", "")).strip(),
            "answer": str(obj.get("answer", "")).strip(),
        }
    if act == "borrow":
        return {
            "action": "borrow",
            "amount": max(0, to_int(obj.get("amount"))),
            "max_rate_pct": clamp(to_int(obj.get("max_rate_pct", 0), 0), 0, 40),
            "ttl": clamp(to_int(obj.get("ttl", 4), 4), 2, 8),
        }
    if act == "lend":
        return {
            "action": "lend",
            "amount": max(0, to_int(obj.get("amount"))),
            "min_rate_pct": clamp(to_int(obj.get("min_rate_pct", 0), 0), 0, 40),
            "due_in": clamp(to_int(obj.get("due_in", 4), 4), 2, 9),
            "ttl": clamp(to_int(obj.get("ttl", 4), 4), 2, 8),
        }
    if act == "repay":
        return {"action": "repay", "loan_id": str(obj.get("loan_id", "")).strip()}
    if act == "post":
        msg = str(obj.get("message", "")).strip()
        return {"action": "post", "message": msg[:220]}
    if act == "skip":
        return {
            "action": "skip",
            "reason": str(obj.get("reason", "unclear")).strip() or "unclear",
        }

    return {"action": "skip", "reason": "unclear"}


def board_tail(board: List[str], keep: int) -> List[str]:
    return board[-keep:]


def build_prompt(
    *,
    tick: int,
    agent: AgentState,
    agents: List[AgentState],
    market: List[Task],
    loans: List[Loan],
    borrows: List[BorrowOrder],
    lends: List[LendOrder],
    board: List[str],
) -> str:
    persona = PERSONAS.get(agent.role, PERSONAS["agent"])
    others = "\n".join(f"- {a.short()}" for a in agents if a.agent_id != agent.agent_id) or "- (нет)"
    bt = "\n".join(f"- {x}" for x in board_tail(board, CFG.board_tail)) or "- (пусто)"

    open_loans = [l for l in loans if l.status == "open"]
    open_loans_s = (
        "\n".join(
            f"{l.loan_id}: b{l.borrower_id} owes {l.total_due()} (p={l.principal}+i={l.interest}) due={l.due_tick} to l{l.lender_id}"
            for l in open_loans[-CFG.loans_limit :]
        )
        or "(нет)"
    )

    return f"""
{persona}

tick={tick}
YOU: id={agent.agent_id} role={agent.role} budget={agent.budget} score={agent.score} rep={agent.reputation}
PERSONALITY: risk={agent.risk:.2f} debt_aversion={agent.debt_aversion:.2f}

OTHERS:
{others}

TASKS:
{market_view(market, CFG.market_limit)}

OPEN LOANS:
{open_loans_s}

CREDIT BOOK:
{credit_book_view(borrows, lends, CFG.book_limit)}

LAST EVENTS:
{bt}

{ACTION_SCHEMA}
""".strip()


# =========================
# FALLBACK (engineered)
# =========================
def best_affordable_task(agent: AgentState, market: List[Task]) -> Optional[Task]:
    affordable = [t for t in market if t.cost <= agent.budget]
    if not affordable:
        return None
    # чуть подталкиваем к росту: ростовые задачи выгоднее по reward
    return max(affordable, key=lambda t: (t.roi, t.reward, t.tier == "growth", -t.cost))


def best_task_overall(market: List[Task]) -> Optional[Task]:
    return max(market, key=lambda t: (t.roi, t.reward, t.tier == "jackpot"), default=None)


def my_open_loans(loans: List[Loan], borrower_id: int) -> List[Loan]:
    return sorted(
        [l for l in loans if l.status == "open" and l.borrower_id == borrower_id],
        key=lambda x: x.due_tick,
    )


def memory_tail(mem: List[MemoryItem], k: int) -> List[MemoryItem]:
    return mem[-k:] if mem else []


def desired_rate(agent: AgentState, urgency: float) -> int:
    """
    max_rate_pct, который агент готов платить.
    Авантюрист (risk высокий) готов платить больше.
    Осторожный — меньше.
    urgency 0..1 увеличивает ставку.
    """
    base = 8 + int(agent.risk * 12)  # 8..20
    bump = int(urgency * 10)  # 0..10
    penalty = int(agent.debt_aversion * 6)  # 0..6
    return clamp(base + bump - penalty, 5, 30)


def fallback_action(
    *,
    tick: int,
    agent: AgentState,
    agents: List[AgentState],
    market: List[Task],
    loans: List[Loan],
    borrows: List[BorrowOrder],
    lends: List[LendOrder],
    board: List[str],
) -> Dict[str, Any]:
    # BANK: держит ликвидность
    if agent.role == "bank":
        # 1) если есть BORROW — закрываем в первую очередь
        open_b = [b for b in borrows if b.status == "open" and b.amount > 0 and b.borrower_id != agent.agent_id]
        if open_b and agent.budget >= 40:
            b = sorted(open_b, key=lambda x: (-x.amount, x.created_tick))[0]
            # банк хочет заработать, но не душит: 8..14%
            min_rate = clamp(9 + int((1 - min(1.0, agent.budget / 400)) * 4), 8, 14)
            amt = min(b.amount, max(40, int(agent.budget * 0.45)))
            return {
                "action": "lend",
                "amount": amt,
                "min_rate_pct": min_rate,
                "due_in": 4,
                "ttl": 5,
            }

        # 2) иначе всегда выставляем "standing liquidity" (это прям важно)
        if agent.budget >= 80:
            return {
                "action": "lend",
                "amount": 60,
                "min_rate_pct": 10,
                "due_in": 4,
                "ttl": 5,
            }

        # 3) если банк бедный — решает small, чтобы восстановить бюджет
        t = best_affordable_task(agent, market)
        if t and t.tier == "small":
            return {
                "action": "solve",
                "task_id": t.task_id,
                "answer": "HTTP — протокол обмена данными между клиентом и сервером.",
            }
        return {"action": "skip", "reason": "bank_wait"}

    # 1) repay: если скоро и можем
    my = my_open_loans(loans, agent.agent_id)
    if my:
        l0 = my[0]
        due_soon = tick >= l0.due_tick - 1
        if agent.budget >= l0.total_due() and (due_soon or agent.debt_aversion >= 0.6):
            return {"action": "repay", "loan_id": l0.loan_id}

    # 2) иногда выгодно быть "мини-кредитором" (богатые богатеют)
    open_b = [b for b in borrows if b.status == "open" and b.amount > 0 and b.borrower_id != agent.agent_id]
    if open_b and agent.budget >= 140 and random.random() < 0.25:
        b = sorted(open_b, key=lambda x: (-x.amount, x.created_tick))[0]
        rate = clamp(11 + int((1 - agent.risk) * 6), 10, 18)
        amt = min(b.amount, int(agent.budget * 0.25))
        return {
            "action": "lend",
            "amount": amt,
            "min_rate_pct": rate,
            "due_in": 4,
            "ttl": 4,
        }

    # 3) ВАЖНО: сначала пытаемся занять под дорогую цель (growth/jackpot), потом solve
    best = best_task_overall(market)
    if best and best.cost > agent.budget:
        # правило: осторожные занимают только под growth/jackpot, авантюристы — под любые
        need = best.cost - agent.budget
        urgency = 1.0 if best.tier == "jackpot" else 0.6 if best.tier == "growth" else 0.3
        allow = (best.tier in ("growth", "jackpot")) or (agent.risk >= 0.75)
        if allow and need > 0:
            tail = memory_tail(agent.memory, 8)
            tried = sum(1 for m in tail if m.action == "borrow")
            filled = sum(1 for m in tail if m.action == "borrow" and m.outcome == "filled")
            if tried >= 3 and filled == 0:
                msg = f"Нужен займ {need} для {best.task_id} [{best.tier}] c={best.cost} r={best.reward}."
                return {"action": "post", "message": msg[:220]}
            max_rate = desired_rate(agent, urgency)
            return {
                "action": "borrow",
                "amount": need,
                "max_rate_pct": max_rate,
                "ttl": 5,
            }

    # 4) solve лучшую доступную
    t = best_affordable_task(agent, market)
    if t:
        if "HTTP" in t.description:
            ans = "HTTP — протокол обмена данными между клиентом и сервером."
        elif "GET" in t.description:
            ans = "GET обычно получает данные, POST отправляет/создаёт данные на сервере."
        elif "pytest" in t.description:
            ans = "pytest проще и мощнее: меньше шаблонного кода и удобные фикстуры."
        else:
            ans = "RAG — ответы с опорой на внешние данные, чтобы быть точнее."
        return {"action": "solve", "task_id": t.task_id, "answer": ans[:180]}

    return {"action": "skip", "reason": "no_moves"}


# =========================
# APPLY ACTIONS
# =========================
def find_task(market: List[Task], task_id: str) -> Optional[Task]:
    return next((t for t in market if t.task_id == task_id), None)


def apply_action(
    *,
    tick: int,
    agent: AgentState,
    agents: List[AgentState],
    market: List[Task],
    loans: List[Loan],
    borrows: List[BorrowOrder],
    lends: List[LendOrder],
    action: Dict[str, Any],
    board: List[str],
) -> str:
    act = action.get("action", "skip")

    if act == "post":
        msg = (action.get("message") or "").strip()
        if msg:
            board.append(f"Тик {tick}: A{agent.agent_id} POST: {msg}")
            return "ok"
        agent.reputation -= 1
        return "rejected"

    if act == "solve":
        t = find_task(market, str(action.get("task_id", "")).strip())
        if not t:
            agent.reputation -= 1
            board.append(f"Тик {tick}: A{agent.agent_id} invalid task.")
            return "rejected"
        if agent.budget < t.cost:
            agent.reputation -= 2
            board.append(f"Тик {tick}: A{agent.agent_id} no budget for {t.task_id}.")
            return "rejected"
        ans = (action.get("answer") or "").strip()
        if not ans:
            agent.reputation -= 1
            board.append(f"Тик {tick}: A{agent.agent_id} empty answer.")
            return "rejected"
        agent.budget -= t.cost
        agent.score += t.reward
        agent.reputation += 1
        market.remove(t)
        board.append(f"Тик {tick}: A{agent.agent_id} solved {t.task_id}[{t.tier}] +{t.reward}.")
        return "ok"

    if act == "borrow":
        amount = int(action.get("amount", 0))
        if amount <= 0:
            agent.reputation -= 1
            return "rejected"
        max_rate = int(action.get("max_rate_pct", 0))
        borrows.append(new_borrow_order(agent.agent_id, amount, max_rate, tick, int(action.get("ttl", 5))))
        board.append(f"Тик {tick}: A{agent.agent_id} BORROW amt={amount} max%={max_rate}.")
        return "ok"

    if act == "lend":
        amount = int(action.get("amount", 0))
        if amount <= 0 or agent.budget < amount:
            agent.reputation -= 1
            return "rejected"
        min_rate = int(action.get("min_rate_pct", 0))
        lends.append(
            new_lend_order(
                agent.agent_id,
                amount,
                min_rate,
                int(action.get("due_in", 4)),
                tick,
                int(action.get("ttl", 5)),
            )
        )
        board.append(f"Тик {tick}: A{agent.agent_id} LEND amt={amount} min%={min_rate}.")
        return "ok"

    if act == "repay":
        loan_id = str(action.get("loan_id", "")).strip()
        loan = next((l for l in loans if l.loan_id == loan_id and l.status == "open"), None)
        if not loan or loan.borrower_id != agent.agent_id:
            agent.reputation -= 1
            return "rejected"
        total = loan.total_due()
        if agent.budget < total:
            agent.reputation -= 1
            return "rejected"
        lender = find_agent(agents, loan.lender_id)
        if not lender:
            agent.reputation -= 1
            return "rejected"
        agent.budget -= total
        lender.budget += total
        loan.status = "paid"
        agent.reputation += 2
        lender.reputation += 1
        board.append(f"Тик {tick}: A{agent.agent_id} repaid {loan_id} total={total}.")
        return "ok"

    return "ok"


# =========================
# LOGGING
# =========================
@dataclass
class TickMetrics:
    tick: int
    llm_latency_s: float
    llm_status: str
    credit_deals: int
    deal_volume: int
    open_loans: int
    defaults_total: int
    gini_budget: float
    board_len: int


def write_csv_row(w: csv.DictWriter, agent: AgentState, action: Dict[str, Any], m: TickMetrics) -> None:
    w.writerow(
        {
            "tick": m.tick,
            "agent_id": agent.agent_id,
            "role": agent.role,
            "budget": agent.budget,
            "score": agent.score,
            "reputation": agent.reputation,
            "action": action.get("action"),
            "task_id": action.get("task_id", ""),
            "amount": action.get("amount", ""),
            "rate_pct": action.get("min_rate_pct", action.get("max_rate_pct", "")),
            "loan_id": action.get("loan_id", ""),
            "llm_latency_s": round(m.llm_latency_s, 3),
            "llm_status": m.llm_status,
            "credit_deals": m.credit_deals,
            "deal_volume": m.deal_volume,
            "open_loans": m.open_loans,
            "defaults_total": m.defaults_total,
            "gini_budget": round(m.gini_budget, 4),
            "board_len": m.board_len,
        }
    )


# =========================
# MAIN
# =========================
def main() -> None:
    print("=== MULTIAGENT CREDIT ECONOMY (portfolio) ===")
    print(f"MODEL={CFG.model}")
    print(f"BASE_URL={CFG.base_url}")
    print(
        f"TICK={CFG.tick_timeout_s}s  REQ_CAP={CFG.ollama_req_timeout_s}s  num_predict={CFG.num_predict}  num_ctx={CFG.num_ctx}"
    )
    print(f"MAX_TICKS={CFG.max_ticks}  SEED={CFG.seed}  LLM_EVERY_N={CFG.llm_every_n_ticks}")
    print()

    # 8 agents: 1 банк + разные характеры (монополия-стайл)
    agents: List[AgentState] = [
        AgentState(agent_id=1, role="agent", budget=110, risk=0.15, debt_aversion=0.90),  # осторожный
        AgentState(agent_id=2, role="agent", budget=110, risk=0.35, debt_aversion=0.70),  # аккуратный
        AgentState(agent_id=3, role="bank", budget=320, risk=0.20, debt_aversion=0.95),  # банк
        AgentState(agent_id=4, role="agent", budget=110, risk=0.65, debt_aversion=0.45),  # рост
        AgentState(agent_id=5, role="agent", budget=110, risk=0.85, debt_aversion=0.20),  # авантюрист
        AgentState(agent_id=6, role="agent", budget=110, risk=0.55, debt_aversion=0.35),  # баланс
        AgentState(agent_id=7, role="agent", budget=110, risk=0.75, debt_aversion=0.30),  # охотник за jackpot
        AgentState(agent_id=8, role="agent", budget=110, risk=0.25, debt_aversion=0.60),  # средний
    ]

    market: List[Task] = [gen_task() for _ in range(7)]
    loans: List[Loan] = []
    borrows: List[BorrowOrder] = []
    lends: List[LendOrder] = []
    board: List[str] = []

    llm_timeout_soft = max(12, min(CFG.ollama_req_timeout_s, int(CFG.tick_timeout_s - 2)))

    with open(CFG.log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tick",
                "agent_id",
                "role",
                "budget",
                "score",
                "reputation",
                "action",
                "task_id",
                "amount",
                "rate_pct",
                "loan_id",
                "llm_latency_s",
                "llm_status",
                "credit_deals",
                "deal_volume",
                "open_loans",
                "defaults_total",
                "gini_budget",
                "board_len",
            ],
        )
        writer.writeheader()

        try:
            for tick in range(1, CFG.max_ticks + 1):
                tick_start = now_s()
                deadline = tick_start + float(CFG.tick_timeout_s)

                # tasks
                for _ in range(CFG.new_tasks_per_tick):
                    market.append(gen_task())
                if len(market) > CFG.market_size_cap:
                    market[:] = sorted(market, key=lambda t: (t.ttl, -t.roi), reverse=True)[: CFG.market_size_cap]

                agent = agents[(tick - 1) % len(agents)]
                llm_latency = 0.0
                llm_status = "skip"
                action: Dict[str, Any] = {"action": "skip", "reason": "strategic"}

                # cooldown
                if agent.llm_cooldown > 0:
                    agent.llm_cooldown -= 1
                    llm_status = f"cooldown({agent.llm_cooldown})"

                remaining = deadline - now_s()
                usable = remaining - 1.0  # reserve for apply/clearing/log

                # LLM call not every tick (экономим, и меньше фризов)
                allow_llm = tick % CFG.llm_every_n_ticks == 0

                if agent.llm_cooldown == 0 and allow_llm and usable >= 5.0:
                    prompt = build_prompt(
                        tick=tick,
                        agent=agent,
                        agents=agents,
                        market=market,
                        loans=loans,
                        borrows=borrows,
                        lends=lends,
                        board=board,
                    )
                    llm_timeout = int(min(llm_timeout_soft, usable))
                    t0 = now_s()
                    raw, status = OLLAMA.generate(prompt, timeout_s=llm_timeout)
                    llm_latency = now_s() - t0
                    llm_status = status

                    if status == "ok":
                        obj = safe_json_loads(raw) or {}
                        action = normalize_action(obj)
                    else:
                        # LLM flaked — cooldown, but simulation lives
                        agent.llm_cooldown = 2
                        action = {"action": "skip", "reason": f"llm_{status}"}
                        board.append(f"Тик {tick}: [WARN] LLM {status} for A{agent.agent_id}")

                # fallback when skip/unclear
                if action.get("action") in ("skip", "", None):
                    action = fallback_action(
                        tick=tick,
                        agent=agent,
                        agents=agents,
                        market=market,
                        loans=loans,
                        borrows=borrows,
                        lends=lends,
                        board=board,
                    )

                outcome = apply_action(
                    tick=tick,
                    agent=agent,
                    agents=agents,
                    market=market,
                    loans=loans,
                    borrows=borrows,
                    lends=lends,
                    action=action,
                    board=board,
                )

                deals, vol = clear_credit_market(
                    tick=tick,
                    agents=agents,
                    borrows=borrows,
                    lends=lends,
                    loans=loans,
                    board=board,
                )

                expire_orders(tick, borrows, lends, board)
                defaults_added = expire_market_and_loans(tick, market, loans, agents, board)

                # memory
                agent.memory.append(
                    MemoryItem(
                        tick=tick,
                        action=str(action.get("action")),
                        outcome=("filled" if deals > 0 and action.get("action") == "borrow" else outcome),
                    )
                )
                if len(agent.memory) > 80:
                    agent.memory[:] = agent.memory[-60:]

                # cap board
                if len(board) > 800:
                    board[:] = board[-500:]

                metrics = TickMetrics(
                    tick=tick,
                    llm_latency_s=llm_latency,
                    llm_status=llm_status,
                    credit_deals=deals,
                    deal_volume=vol,
                    open_loans=sum(1 for l in loans if l.status == "open"),
                    defaults_total=count_defaults(loans),
                    gini_budget=gini([a.budget for a in agents]),
                    board_len=len(board),
                )
                write_csv_row(writer, agent, action, metrics)
                f.flush()

                sleep_left = deadline - now_s()
                print(
                    f"tick={tick:02d} {agent.short()} act={action.get('action')} out={outcome} "
                    f"llm={llm_status} lat={llm_latency:.2f}s deals={deals} vol={vol} open={metrics.open_loans} "
                    f"def+={defaults_added} gini={metrics.gini_budget:.2f} sleep={max(0.0, sleep_left):.2f}s"
                )
                if sleep_left > 0:
                    time.sleep(sleep_left)

        except KeyboardInterrupt:
            print("\n[STOP] KeyboardInterrupt — сохраняю CSV и выхожу аккуратно...")

    print("\n[OK] CSV saved:", CFG.log_csv)


if __name__ == "__main__":
    main()
