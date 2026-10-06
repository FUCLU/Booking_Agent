"""
tests/conftest.py - Pytest fixtures dùng chung cho toàn bộ test suite.

Cung cấp:
  - db: MockAirline đã có dữ liệu cơ bản
  - customer / guest: UserCtx khách hàng / khách vãng lai
  - harness: Harness BẬT enforce với customer
  - intent_book / intent_cancel / intent_rebook: Intent mẫu
"""
from __future__ import annotations

import pytest

from harness import Harness, Intent, UserCtx
from scenarios import base_db
from tools import MockAirline


# --- Users ---
@pytest.fixture
def customer() -> UserCtx:
    return UserCtx("u1", "Nguyen Van A", {"flights:read", "flights:book", "bookings:cancel"})


@pytest.fixture
def guest() -> UserCtx:
    """Khách vãng lai, chỉ có quyền đọc."""
    return UserCtx("u3", "Khach Vang Lai", {"flights:read"})


@pytest.fixture
def no_scope_user() -> UserCtx:
    """User hoàn toàn không có quyền gì."""
    return UserCtx("u99", "No Access", set())


# --- DB ---
@pytest.fixture
def db() -> MockAirline:
    """DB cơ bản với 4 chuyến bay (giống base_db trong scenarios)."""
    return base_db()


@pytest.fixture
def empty_db() -> MockAirline:
    """DB rỗng, không có chuyến bay nào."""
    return MockAirline()


# --- Harness ---
@pytest.fixture
def harness(db, customer) -> Harness:
    """Harness BẬT enforce, dùng customer."""
    return Harness(db, customer, enforce=True)


@pytest.fixture
def harness_no_enforce(db, customer) -> Harness:
    """Harness TẮT enforce (ablation mode)."""
    return Harness(db, customer, enforce=False)


# --- Intent ---
@pytest.fixture
def intent_book() -> Intent:
    return Intent(kind="book_cheapest", origin="SGN", destination="HAN",
                  depart_date="2026-10-20", passenger="Nguyen Van A")


@pytest.fixture
def intent_cancel() -> Intent:
    return Intent(kind="cancel", booking_id="BK0001", passenger="Nguyen Van A")


@pytest.fixture
def intent_rebook() -> Intent:
    return Intent(kind="rebook", origin="SGN", destination="HAN",
                  depart_date="2026-10-20", booking_id="BK0001", passenger="Nguyen Van A")

