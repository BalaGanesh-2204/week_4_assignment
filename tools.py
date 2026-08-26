"""
Tool layer for the e-commerce support agent.

15 tools exposed to the LLM through TOOL_REGISTRY (OpenAI-style
function schemas understood by Groq). Handlers read seeded mock
data, the knowledge base (hybrid search), persistent memory and
the reminder queue.

Chaining happens naturally: each step's tool results are fed back
into the conversation so later calls can depend on earlier ones.
Sensitive tools are gated (e.g. refunds require a prior successful
eligibility check within the same turn).
"""

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Set

import hybrid_search
import keyword_search  # noqa: F401  (ensures store exists before search)
import memory_store
import reminders

from config import (
    MOCK_DATA_DIR,
    DATA_DIR,
    RETURN_WINDOW_DAYS,
    ESCALATE_REFUND_THRESHOLD,
    RESTOCKING_FEE_RATE,
    SUPPORT_LEAD,
)


# ---------------------------------------------------------
# RUNTIME CONTEXT (per turn)
# ---------------------------------------------------------

@dataclass
class ToolContext:
    """
    Carries per-turn state between tool calls.

    Enables chaining rules like "refund requires prior eligibility".
    """

    customer_email: Optional[str] = None
    eligible_order_ids: Set[str] = field(default_factory=set)
    refund_requests_created: List[Dict] = field(default_factory=list)
    escalations_raised: List[str] = field(default_factory=list)

    @property
    def customer_key(self) -> str:
        return (self.customer_email or "guest").strip().lower()


ELECTRONICS_KEYWORDS = [
    "headphone", "charger", "laptop", "camera", "smartwatch",
    "speaker", "keyboard", "mouse", "buds", "cable", "watch",
]

FAULTY_REASON_KEYWORDS = [
    "damage", "damaged", "defect", "defective", "broken",
    "faulty", "dead", "not working", "stopped working",
    "missing", "wrong item", "incorrect",
]

SHIPPING_COSTS = {
    1: {"standard": 4.99, "express": 12.99, "overnight": 24.99},
    2: {"standard": 4.99, "express": 12.99, "overnight": 24.99},
    3: {"standard": 5.99, "express": 14.99, "overnight": None},
    4: {"standard": 7.99, "express": 17.99, "overnight": None},
    5: {"standard": 7.99, "express": 17.99, "overnight": None},
    6: {"standard": 9.99, "express": 21.99, "overnight": None},
    7: {"standard": 9.99, "express": 21.99, "overnight": None},
    8: {"standard": 9.99, "express": 21.99, "overnight": None},
    9: {"standard": 9.99, "express": 21.99, "overnight": None},
}


# ---------------------------------------------------------
# DATA LOADING HELPERS
# ---------------------------------------------------------

def _load_json(filename: str):
    path = MOCK_DATA_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


REFUNDS_LEDGER_FILE = DATA_DIR / "refunds_ledger.json"
MEMORIES_FILE = DATA_DIR / "memories.json"


def _load_memories() -> List[Dict]:
    if MEMORIES_FILE.exists():
        return json.loads(MEMORIES_FILE.read_text(encoding="utf-8"))
    return []


def _save_memories(memories: List[Dict]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORIES_FILE.write_text(
        json.dumps(memories, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_refund_ledger(entry: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ledger = []

    if REFUNDS_LEDGER_FILE.exists():
        ledger = json.loads(
            REFUNDS_LEDGER_FILE.read_text(encoding="utf-8")
        )

    ledger.append(entry)

    REFUNDS_LEDGER_FILE.write_text(
        json.dumps(ledger, indent=2),
        encoding="utf-8",
    )


def _find_customer(identifier: str) -> Optional[Dict]:
    identifier = identifier.strip().lower()

    for customer in _load_json("customers.json"):

        if (
            identifier == customer["email"].lower()
            or identifier == customer["customer_id"].lower()
        ):
            return customer

    return None


def _find_order(order_id: str) -> Optional[Dict]:
    order_id = order_id.strip().upper()

    for order in _load_json("orders.json"):

        if order["order_id"].upper() == order_id:
            return order

    return None


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


# ---------------------------------------------------------
# TOOL HANDLERS
# ---------------------------------------------------------

def _search_knowledge_base(
    query: str,
    context: ToolContext,
) -> Dict:
    hits = hybrid_search.hybrid_search(query)

    return {
        "query": query,
        "results": [
            {
                "text": hit["text"][:900],
                "source": hit["source"],
                "heading": hit["heading"],
                "path": hit["path"],
            }
            for hit in hits
        ],
    }


def _get_customer_profile(
    email_or_id: str,
    context: ToolContext,
) -> Dict:
    customer = _find_customer(email_or_id)

    if not customer:
        return {
            "error": f"No customer found for '{email_or_id}'."
        }

    return {"customer": customer}


def _list_customer_orders(
    customer_id_or_email: str,
    context: ToolContext,
) -> Dict:
    customer = _find_customer(customer_id_or_email)

    if not customer:
        return {
            "error": f"No customer found for '{customer_id_or_email}'."
        }

    cid = customer["customer_id"]

    orders = [
        {
            "order_id": o["order_id"],
            "status": o["status"],
            "placed_at": o["ordered_at"],
            "total": o["total"],
            "items": [
                f"{i['qty']}x {i['name']}" for i in o["items"]
            ],
        }
        for o in _load_json("orders.json")
        if o["customer_id"] == cid
    ]

    return {"customer_id": cid, "orders": orders}


def _get_order_status(
    order_id: str,
    context: ToolContext,
) -> Dict:
    order = _find_order(order_id)

    if not order:
        return {"error": f"Order '{order_id}' not found."}

    info = {
        "order_id": order["order_id"],
        "status": order["status"],
        "ordered_at": order["ordered_at"],
        "tracking_number": order.get("tracking_number"),
        "estimated_delivery": order.get("estimated_delivery"),
        "delivered_at": order.get("delivered_at"),
        "items": order["items"],
        "total": order["total"],
        "notes": order.get("notes", ""),
    }

    return {"order": info}


def _check_return_eligibility(
    order_id: str,
    context: ToolContext,
) -> Dict:
    order = _find_order(order_id)

    if not order:
        return {"error": f"Order '{order_id}' not found."}

    if order["status"] == "cancelled":
        return {
            "order_id": order_id,
            "eligible": False,
            "reason": (
                "Order was cancelled. Cancellations are refunded "
                "to source automatically."
            ),
        }

    if order["status"] != "delivered" or not order.get("delivered_at"):
        return {
            "order_id": order_id,
            "eligible": False,
            "reason": (
                "Order is not delivered yet. Returns open only "
                "after delivery."
            ),
        }

    delivered = _parse_date(order["delivered_at"])
    days_since = (date.today() - delivered).days

    notes = (order.get("notes") or "").lower()

    if "final sale" in notes:
        return {
            "order_id": order_id,
            "eligible": False,
            "reason": (
                "This order contains final-sale items which cannot "
                "be returned per policy."
            ),
        }

    if days_since > RETURN_WINDOW_DAYS:
        return {
            "order_id": order_id,
            "eligible": False,
            "reason": (
                f"Delivered {days_since} days ago; the return window "
                f"is {RETURN_WINDOW_DAYS} days."
            ),
        }

    context.eligible_order_ids.add(order["order_id"])

    return {
        "order_id": order_id,
        "eligible": True,
        "days_since_delivery": days_since,
        "days_left_in_window": RETURN_WINDOW_DAYS - days_since,
        "reason": "Within the return window.",
        "note": (
            "You may now estimate the refund amount or create the "
            "request for this order."
        ),
    }


def _estimate_refund_amount(
    order_id: str,
    reason: str = "",
    context: ToolContext = None,
) -> Dict:
    order = _find_order(order_id)

    if not order:
        return {"error": f"Order '{order_id}' not found."}

    if order_id not in context.eligible_order_ids:
        return {
            "error": (
                "Run check_return_eligibility for this order first; "
                "estimates require an eligibility confirmation."
            )
        }

    subtotal = sum(i["price"] * i["qty"] for i in order["items"])

    lowered_reason = (reason or "").lower()
    faulty = any(
        word in lowered_reason for word in FAULTY_REASON_KEYWORDS
    )

    electronics = any(
        keyword in i["name"].lower() for i in order["items"]
        for keyword in ELECTRONICS_KEYWORDS
    )

    restocking_fee = 0.0

    if electronics and not faulty:
        restocking_fee = round(subtotal * RESTOCKING_FEE_RATE, 2)

    refund_amount = round(subtotal - restocking_fee, 2)

    return {
        "order_id": order_id,
        "subtotal": subtotal,
        "restocking_fee": restocking_fee,
        "estimated_refund": refund_amount,
        "faulty_claim": faulty,
        "policy_note": (
            "10% restocking fee applies only to opened electronics "
            "with no fault."
        ),
        "requires_manager_review": (
            refund_amount >= ESCALATE_REFUND_THRESHOLD
        ),
    }


def _create_refund_request(
    order_id: str,
    reason: str,
    context: ToolContext,
) -> Dict:
    if order_id not in context.eligible_order_ids:
        return {
            "error": (
                "Refusal: eligibility has not been confirmed for this "
                "order this turn. Run check_return_eligibility first."
            )
        }

    estimate = _estimate_refund_amount(
        order_id, reason, context
    )

    amount = estimate["estimated_refund"]

    entry = {
        "rma_ref": f"RFD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "order_id": order_id,
        "customer_key": context.customer_key,
        "reason": reason,
        "amount": amount,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    _append_refund_ledger(entry)
    context.refund_requests_created.append(entry)

    escalated = False

    if amount >= ESCALATE_REFUND_THRESHOLD:

        escalated = True
        context.escalations_raised.append(entry["rma_ref"])

        reminders.raise_escalation(
            summary=(
                f"High-value refund request {entry['rma_ref']} for "
                f"{order_id} (${amount:.2f}) needs approval."
            ),
            reason=reason,
            meta={"order_id": order_id, "amount": amount},
        )

    return {
        "refund_request": entry,
        "status": (
            "logged - pending manager approval"
            if escalated
            else "logged - approved automatically"
        ),
        "manager_review_required": escalated,
        "message": (
            f"The ${amount:.2f} refund request is recorded and routed "
            f"to {SUPPORT_LEAD} for review."
            if escalated
            else f"Refund of ${amount:.2f} approved and recorded."
        ),
    }


def _create_return_label(
    order_id: str,
    context: ToolContext,
) -> Dict:
    if order_id not in context.eligible_order_ids:
        return {
            "error": (
                "Refusal: run check_return_eligibility first; labels "
                "are issued only for eligible orders."
            )
        }

    rma = f"RMA-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return {
        "order_id": order_id,
        "rma_number": rma,
        "label_url": f"https://labels.shopkart.example/{rma}.pdf",
        "instructions": (
            "Print the prepaid label and drop the parcel at any "
            "partner pickup point within 14 days."
        ),
    }


def _check_stock_status(
    product_name: str,
    context: ToolContext,
) -> Dict:
    needle = product_name.strip().lower()

    matches = []

    for product in _load_json("stock.json"):

        haystack = (
            f"{product['name']} {product['sku']}".lower()
        )

        if needle in haystack or needle == product["sku"].lower():
            matches.append(product)

    if not matches:
        return {"error": f"No product matching '{product_name}'."}

    return {
        "products": [
            {
                "name": m["name"],
                "sku": m["sku"],
                "in_stock": m["stock"] > 0,
                "units_available": m["stock"],
                "restock_date": m.get("restock_date"),
            }
            for m in matches[:5]
        ]
    }


def _validate_promo_code(
    code: str,
    cart_total: float = 0.0,
    context: ToolContext = None,
) -> Dict:
    code_clean = code.strip().upper()

    for promo in _load_json("promos.json"):

        if promo["code"].upper() != code_clean:
            continue

        result = {
            "code": promo["code"],
            "exists": True,
            "active": promo["active"],
            "discount_type": promo["discount_type"],
            "discount_value": promo["discount_value"],
            "min_spend": promo["min_spend"],
            "expires_at": promo["expires_at"],
            "note": promo["note"],
        }

        today = date.today()
        expires = _parse_date(promo["expires_at"])

        if not promo["active"]:
            result["valid"] = False
            result["problem"] = "This code is inactive."

        elif expires < today:
            result["valid"] = False
            result["problem"] = (
                f"This code expired on {promo['expires_at']}."
            )

        elif cart_total and cart_total < promo["min_spend"]:
            result["valid"] = False
            result["problem"] = (
                f"Cart total ${cart_total:.2f} is below the "
                f"${promo['min_spend']:.2f} minimum spend."
            )

        else:
            result["valid"] = True

        return result

    return {
        "code": code_clean,
        "exists": False,
        "valid": False,
        "problem": "Unknown promo code.",
    }


def _get_shipping_estimate(
    postcode: str,
    method: str = "standard",
    context: ToolContext = None,
) -> Dict:
    method_clean = method.strip().lower()

    if method_clean not in ("standard", "express", "overnight"):
        return {
            "error": "method must be standard, express or overnight."
        }

    digits = "".join(ch for ch in postcode if ch.isdigit())

    if not digits:
        return {"error": f"'{postcode}' is not a valid postcode."}

    zone_digit = int(digits[0])

    costs = SHIPPING_COSTS.get(zone_digit, SHIPPING_COSTS[9])

    cost = costs[method_clean]

    if cost is None:
        return {
            "postcode": postcode,
            "zone": zone_digit,
            "method": method_clean,
            "available": False,
            "note": (
                f"{method_clean.title()} delivery is not available "
                f"for this zone."
            ),
        }

    eta_days = {
        (1, "standard"): 3,
        (1, "express"): 1,
        (1, "overnight"): 1,
        (2, "standard"): 5,
        (2, "express"): 2,
        (3, "standard"): 6,
        (3, "express"): 3,
    }.get((zone_digit, method_clean))

    if eta_days is None:
        eta_days = 10 if method_clean == "standard" else 5

    free_shipping = (
        method_clean == "standard"
    )

    return {
        "postcode": postcode,
        "zone": zone_digit,
        "method": method_clean,
        "cost_usd": cost,
        "estimated_business_days": eta_days,
        "free_standard_over_75": free_shipping,
        "note": (
            "Standard shipping is free on orders above $75."
            if free_shipping
            else ""
        ),
    }


def _remember_fact(
    key: str,
    value: str,
    context: ToolContext,
) -> Dict:
    memory_store.remember_fact(context.customer_key, key, value)

    return {
        "saved": True,
        "customer_key": context.customer_key,
        "key": key,
        "value": value,
    }


def _recall_memory(
    key: str = "",
    context: ToolContext = None,
) -> Dict:
    facts = memory_store.recall_facts(
        context.customer_key,
        key if key else None,
    )

    return {
        "customer_key": context.customer_key,
        "facts": facts,
    }


def _add_reminder(
    message: str,
    urgency: str = "medium",
    context: ToolContext = None,
) -> Dict:
    entry = reminders.add_reminder(message, urgency)

    return {
        "added": True,
        "reminder_id": entry["id"],
        "urgency": entry["urgency"],
        "message": entry["message"],
    }


def _escalate_to_balaganesh(
    summary: str,
    reason: str = "",
    context: ToolContext = None,
) -> Dict:
    entry = reminders.raise_escalation(
        summary=summary,
        reason=reason,
        meta={"customer_key": context.customer_key},
    )

    context.escalations_raised.append(entry["id"])

    return {
        "escalated": True,
        "escalation_id": entry["id"],
        "urgency": "critical",
        "message": (
            f"A human colleague ({SUPPORT_LEAD}) has been alerted "
            "and will follow up. The customer should be told a "
            "person will contact them."
        ),
    }


def _store_issue(
    order_id: str,
    issue_type: str,
    description: str,
    context: ToolContext,
) -> Dict:
    memories = _load_memories()

    entry = {
        "id": f"ISS-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "order_id": order_id,
        "customer_key": context.customer_key,
        "issue_type": issue_type,
        "description": description,
        "status": "open",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    memories.append(entry)
    _save_memories(memories)

    return {
        "stored": True,
        "issue_id": entry["id"],
        "order_id": order_id,
        "issue_type": issue_type,
        "status": "open",
    }


def _recall_issues(
    order_id: str = "",
    context: ToolContext = None,
) -> Dict:
    memories = _load_memories()

    if order_id:
        issues = [m for m in memories if m.get("order_id") == order_id]
    else:
        issues = [m for m in memories if m.get("customer_key") == context.customer_key]

    return {
        "customer_key": context.customer_key,
        "order_filter": order_id or "all",
        "issues": issues,
        "count": len(issues),
    }


# ---------------------------------------------------------
# GROQ TOOL REGISTRY
# ---------------------------------------------------------

def _tool(name: str, description: str, properties: Dict, required: List[str]) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_REGISTRY = [
    _tool(
        "search_knowledge_base",
        "Hybrid search over store policy documents. Use for any "
        "policy/how-to question: shipping times and costs, returns, "
        "refunds, warranty, payments, promos, account help.",
        {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
        },
        ["query"],
    ),
    _tool(
        "get_customer_profile",
        "Look up a customer by email address or customer id "
        "(e.g. CUS-1001).",
        {
            "email_or_id": {"type": "string"},
        },
        ["email_or_id"],
    ),
    _tool(
        "list_customer_orders",
        "List all orders of a customer with status summaries.",
        {
            "customer_id_or_email": {"type": "string"},
        },
        ["customer_id_or_email"],
    ),
    _tool(
        "get_order_status",
        "Get full status of one order: items, tracking number, "
        "delivery estimate or delivery date.",
        {
            "order_id": {
                "type": "string",
                "description": "Order id like ORD-1042.",
            },
        },
        ["order_id"],
    ),
    _tool(
        "check_return_eligibility",
        "Check whether an order can be returned under the 30-day "
        "policy. MUST be called before refunds or labels.",
        {
            "order_id": {"type": "string"},
        },
        ["order_id"],
    ),
    _tool(
        "estimate_refund_amount",
        "Calculate the exact refund for an eligible order, applying "
        "restocking-fee rules. Requires prior eligibility check.",
        {
            "order_id": {"type": "string"},
            "reason": {
                "type": "string",
                "description": (
                    "Customer's stated reason, e.g. 'arrived damaged' "
                    "or 'changed my mind'."
                ),
            },
        },
        ["order_id"],
    ),
    _tool(
        "create_refund_request",
        "Create an official refund request for an eligible order. "
        "Refunds of $200+ are auto-routed to the human lead.",
        {
            "order_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        ["order_id", "reason"],
    ),
    _tool(
        "create_return_label",
        "Issue a prepaid return shipping label for an eligible "
        "order. Requires prior eligibility check.",
        {
            "order_id": {"type": "string"},
        },
        ["order_id"],
    ),
    _tool(
        "check_stock_status",
        "Check live stock for a product by name or SKU, including "
        "restock dates when out of stock.",
        {
            "product_name": {"type": "string"},
        },
        ["product_name"],
    ),
    _tool(
        "validate_promo_code",
        "Validate a promo code: active, expiry, minimum spend.",
        {
            "code": {"type": "string"},
            "cart_total": {
                "type": "number",
                "description": "Optional cart total to test min spend.",
            },
        },
        ["code"],
    ),
    _tool(
        "get_shipping_estimate",
        "Estimate shipping cost and days for a postcode and method "
        "(standard/express/overnight).",
        {
            "postcode": {"type": "string"},
            "method": {
                "type": "string",
                "enum": ["standard", "express", "overnight"],
            },
        },
        ["postcode"],
    ),
    _tool(
        "remember_fact",
        "Persist a durable fact about the current customer across "
        "sessions (e.g. preferred contact, product interest).",
        {
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        ["key", "value"],
    ),
    _tool(
        "recall_memory",
        "Recall stored facts about the current customer. Omit key "
        "to list all facts.",
        {
            "key": {"type": "string"},
        },
        [],
    ),
    _tool(
        "add_reminder",
        "Leave a note on the human support lead's reminder board "
        "(follow-ups, promises made to customers).",
        {
            "message": {"type": "string"},
            "urgency": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
            },
        },
        ["message"],
    ),
    _tool(
        "escalate_to_balaganesh",
        "Escalate to the human support lead: high-value approvals, "
        "security issues, angry customers, anything beyond agent "
        "authority, or explicit customer request.",
        {
            "summary": {"type": "string"},
            "reason": {"type": "string"},
        },
        ["summary", "reason"],
    ),
    _tool(
        "store_issue",
        "Record a customer-reported issue (damaged item, wrong product, "
        "missing item, etc.) linked to an order. Stores in memories.json.",
        {
            "order_id": {"type": "string"},
            "issue_type": {
                "type": "string",
                "enum": [
                    "damaged",
                    "wrong_item",
                    "missing_item",
                    "not_working",
                    "late_delivery",
                    "other",
                ],
                "description": "Category of the issue.",
            },
            "description": {
                "type": "string",
                "description": "Brief description of the problem.",
            },
        },
        ["order_id", "issue_type", "description"],
    ),
    _tool(
        "recall_issues",
        "Retrieve previously reported issues. Filter by order_id or "
        "list all issues for the current customer.",
        {
            "order_id": {"type": "string"},
        },
        [],
    ),
]


# ---------------------------------------------------------
# DISPATCH TABLE
# ---------------------------------------------------------

HANDLERS = {
    "search_knowledge_base": _search_knowledge_base,
    "get_customer_profile": _get_customer_profile,
    "list_customer_orders": _list_customer_orders,
    "get_order_status": _get_order_status,
    "check_return_eligibility": _check_return_eligibility,
    "estimate_refund_amount": _estimate_refund_amount,
    "create_refund_request": _create_refund_request,
    "create_return_label": _create_return_label,
    "check_stock_status": _check_stock_status,
    "validate_promo_code": _validate_promo_code,
    "get_shipping_estimate": _get_shipping_estimate,
    "remember_fact": _remember_fact,
    "recall_memory": _recall_memory,
    "add_reminder": _add_reminder,
    "escalate_to_balaganesh": _escalate_to_balaganesh,
    "store_issue": _store_issue,
    "recall_issues": _recall_issues,
}


def execute_tool(name: str, arguments: Dict, context: ToolContext) -> Dict:
    """
    Execute one tool safely. Never raises - errors become results
    the model can read and react to.
    """

    handler = HANDLERS.get(name)

    if handler is None:
        return {"error": f"Unknown tool '{name}'."}

    kwargs = {}

    import inspect

    signature = inspect.signature(handler)

    for param_name, param in signature.parameters.items():

        if param_name == "context":
            kwargs["context"] = context
            continue

        if param_name in arguments:
            kwargs[param_name] = arguments[param_name]

        elif param.default is inspect.Parameter.empty:
            return {
                "error": f"Missing required argument '{param_name}'."
            }

    try:
        return handler(**kwargs)
    except Exception as exc:
        return {"error": f"Tool '{name}' failed: {exc}"}
