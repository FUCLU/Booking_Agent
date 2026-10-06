from __future__ import annotations

import json

from config import TODAY
from harness import Intent, Obs, ToolResult
from tools import TOOLS

# 1. Prompt hệ thống (dùng chung)
SYSTEM_PROMPT = """\
Bạn là agent đặt vé máy bay, làm việc thay mặt khách hàng. Hôm nay là {today}.

Bạn làm việc bằng cách GỌI TOOL. Quy tắc bắt buộc:
1. flight_id CHỈ được lấy từ kết quả search_flights, tuyệt đối không tự nghĩ ra.
2. Ngày luôn dạng YYYY-MM-DD; mã sân bay là mã IATA 3 chữ in hoa (SGN, HAN, ...).
3. "Rẻ nhất" nghĩa là giá thấp nhất trong các chuyến còn chỗ (seats_left > 0).
4. Đổi vé = huỷ booking cũ trước, rồi mới đặt vé mới.
5. Nếu tool báo lỗi, đọc thông báo lỗi để sửa tham số hoặc đổi cách làm (ví dụ SOLD_OUT thì search lại).
6. Nếu search không có chuyến nào, hoặc bạn không thể tiếp tục an toàn (thiếu quyền, cần người duyệt): \
gọi handoff_to_human.
7. Chỉ gọi finish khi mục tiêu đã thực sự hoàn thành. Nếu nhận COMPLETION_CHECK_FAILED, đọc lý do và làm tiếp.\
"""


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(today=TODAY.isoformat())


# 2. Prompt cho parse 
PARSE_PROMPT = """\
Trích mục tiêu có cấu trúc từ yêu cầu của khách. Hôm nay là {today}.
kind:
- book_cheapest: đặt vé rẻ nhất cho một tuyến và ngày
- cancel: huỷ một booking
- rebook: huỷ booking cũ rồi đặt vé rẻ nhất mới
Ngày dạng YYYY-MM-DD, mã sân bay IATA 3 chữ in hoa. Trường nào khách không nói thì để trống.\
"""

def parse_prompt() -> str:
    return PARSE_PROMPT.format(today=TODAY.isoformat())

PARSE_USER = "Khách tên '{name}' nói: {task}"

def parse_user(name: str, task: str) -> str:
    return PARSE_USER.format(name=name, task=task)


# 3. Mục tiêu của khách (tin nhắn đầu tiên gửi cho model)
GOAL_PROMPT = "Mục tiêu của khách (đã cấu trúc hoá): {intent}"


def goal_prompt(it: Intent) -> str:
    return GOAL_PROMPT.format(intent=it.model_dump_json(exclude_none=True))


# 4. Prompt cho plan 
PLAN_PROMPT = """\
Hãy lập kế hoạch các bước CÒN LẠI để hoàn thành mục tiêu. Trả về danh sách rỗng nếu mục tiêu đã xong.

Các tool có thể dùng trong kế hoạch:
{catalog}
- handoff: chuyển cho nhân viên khi không thể tiếp tục an toàn (ghi lý do vào reason).

Với mỗi bước:
- Điền sẵn tham số nếu đã biết chắc từ mục tiêu (origin, destination, depart_date, booking_id).
- Bước book_flight KHÔNG điền gì: flight_id phụ thuộc kết quả search_flights nên sẽ được điền sau.

Các bước đã làm:
{history}\
"""


def plan_prompt(hist: list[Obs]) -> str:
    catalog = "\n".join(f"- {s.name}: {s.func.__doc__}" for s in TOOLS.values())
    return PLAN_PROMPT.format(catalog=catalog, history=render_history(hist))


# 5. Prompt cho fill_args 
FILL_ARGS_PROMPT = "Hãy gọi tool `{tool}` với tham số đúng, dựa trên mục tiêu và các kết quả/lỗi ở trên."


def fill_args_prompt(tool: str) -> str:
    return FILL_ARGS_PROMPT.format(tool=tool)


# 6. Cách hiển thị kết quả tool cho model 
def format_result(res: ToolResult) -> str:
    """Nội dung ToolMessage gửi lại cho model."""
    if res.ok:
        return json.dumps(res.data, ensure_ascii=False)[:2000]
    return f"LỖI {res.code}: {res.handoff or res.error}"


def render_history(hist: list[Obs]) -> str:
    """Lịch sử dạng văn bản (dùng cho plan, vì lời gọi plan không gắn tool)."""
    if not hist:
        return "(chưa làm gì)"
    return "\n".join(f"- {o.tool}({json.dumps(o.args, ensure_ascii=False)}) -> {format_result(o.res)}"
                     for o in hist)