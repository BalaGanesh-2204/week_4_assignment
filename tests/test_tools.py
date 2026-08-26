import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

for mod_name in ("pinecone", "groq", "google", "google.genai"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from tools import (
    ToolContext,
    _find_customer,
    _find_order,
    _get_customer_profile,
    _list_customer_orders,
    _get_order_status,
    _check_return_eligibility,
    _estimate_refund_amount,
    _check_stock_status,
    _validate_promo_code,
    _get_shipping_estimate,
    execute_tool,
)


@pytest.fixture
def context():
    return ToolContext(customer_email="anita.sharma@example.com")


@pytest.fixture
def empty_context():
    return ToolContext()


class TestFindCustomer:
    def test_finds_by_email(self):
        customer = _find_customer("anita.sharma@example.com")
        assert customer is not None
        assert customer["email"] == "anita.sharma@example.com"

    def test_finds_by_id(self):
        customer = _find_customer("CUS-1001")
        assert customer is not None
        assert customer["customer_id"] == "CUS-1001"

    def test_case_insensitive(self):
        customer = _find_customer("ANITA.SHARMA@EXAMPLE.COM")
        assert customer is not None

    def test_not_found(self):
        customer = _find_customer("nobody@example.com")
        assert customer is None


class TestFindOrder:
    def test_finds_by_id(self):
        order = _find_order("ORD-1041")
        assert order is not None
        assert order["order_id"] == "ORD-1041"

    def test_case_insensitive(self):
        order = _find_order("ord-1041")
        assert order is not None

    def test_not_found(self):
        order = _find_order("ORD-9999")
        assert order is None


class TestGetCustomerProfile:
    def test_existing_customer(self, context):
        result = _get_customer_profile("anita.sharma@example.com", context)
        assert "customer" in result
        assert result["customer"]["email"] == "anita.sharma@example.com"

    def test_nonexistent_customer(self, context):
        result = _get_customer_profile("nobody@example.com", context)
        assert "error" in result


class TestListCustomerOrders:
    def test_existing_customer(self, context):
        result = _list_customer_orders("anita.sharma@example.com", context)
        assert "orders" in result
        assert "customer_id" in result
        assert len(result["orders"]) > 0

    def test_nonexistent_customer(self, context):
        result = _list_customer_orders("nobody@example.com", context)
        assert "error" in result


class TestGetOrderStatus:
    def test_existing_order(self, context):
        result = _get_order_status("ORD-1041", context)
        assert "order" in result
        assert result["order"]["order_id"] == "ORD-1041"

    def test_nonexistent_order(self, context):
        result = _get_order_status("ORD-9999", context)
        assert "error" in result


class TestCheckReturnEligibility:
    def test_delivered_order(self, context):
        result = _check_return_eligibility("ORD-1041", context)
        assert "eligible" in result

    def test_nonexistent_order(self, context):
        result = _check_return_eligibility("ORD-9999", context)
        assert "error" in result

    def test_eligible_order_adds_to_context(self, context):
        result = _check_return_eligibility("ORD-1041", context)
        if result.get("eligible"):
            assert "ORD-1041" in context.eligible_order_ids


class TestEstimateRefundAmount:
    def test_requires_eligibility_check(self, context):
        result = _estimate_refund_amount("ORD-1041", "damaged", context)
        assert "error" in result

    def test_with_eligibility(self, context):
        context.eligible_order_ids.add("ORD-1041")
        result = _estimate_refund_amount("ORD-1041", "damaged", context)
        assert "estimated_refund" in result
        assert result["estimated_refund"] > 0


class TestCheckStockStatus:
    def test_existing_product(self, context):
        result = _check_stock_status("headphone", context)
        assert "products" in result
        assert len(result["products"]) > 0

    def test_nonexistent_product(self, context):
        result = _check_stock_status("xyznonexistent", context)
        assert "error" in result


class TestValidatePromoCode:
    def test_valid_code(self, empty_context):
        result = _validate_promo_code("SAVE5", 100.0, empty_context)
        assert result["exists"] is True
        assert result["valid"] is True

    def test_invalid_code(self, empty_context):
        result = _validate_promo_code("FAKECODE", 100.0, empty_context)
        assert result["exists"] is False
        assert result["valid"] is False


class TestGetShippingEstimate:
    def test_valid_postcode(self, empty_context):
        result = _get_shipping_estimate("10001", "standard", empty_context)
        assert "cost_usd" in result or "available" in result

    def test_invalid_method(self, empty_context):
        result = _get_shipping_estimate("10001", "teleport", empty_context)
        assert "error" in result

    def test_invalid_postcode(self, empty_context):
        result = _get_shipping_estimate("abc", "standard", empty_context)
        assert "error" in result


class TestExecuteTool:
    def test_unknown_tool(self, context):
        result = execute_tool("nonexistent_tool", {}, context)
        assert "error" in result

    def test_missing_required_arg(self, context):
        result = execute_tool("get_customer_profile", {}, context)
        assert "error" in result

    def test_tool_exception_safety(self, context):
        result = execute_tool(
            "get_order_status", {"order_id": "ORD-9999"}, context
        )
        assert isinstance(result, dict)
