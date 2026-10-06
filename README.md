# Hướng dẫn chạy và báo cáo Agent đặt vé máy bay

> Bài tập: tạo công cụ giả lập, cài đặt harness 4 lớp bảo vệ, triển khai ba mẫu ReAct, Plan-then-Execute, Lai và so sánh hiệu quả thông qua 9 kịch bản.

- **Ba mẫu:** `react_agent.py`, `plan_execute_agent.py`, `hybrid_agent.py`
- **Phần dùng chung:** `brains.py`, `tools.py`, `harness.py`, `prompts.py`, `config.py`, `runner.py`, `scenarios.py`
- **Đánh giá:** `evaluate.py`
- **Môi trường:** Python 3.10 trở lên; `langchain`, `pydantic`.

## 1. Hướng dẫn chạy

Mở PowerShell tại thư mục dự án. Tạo môi trường Python riêng, cài thư viện, rồi chạy:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Cấu hình API Key trong file `.env` (ví dụ `OPENROUTER_API_KEY` hoặc `OPENAI_API_KEY` tùy provider) rồi mới có thể gọi LLM thật:

```powershell
# Chạy demo từng mẫu (chạy nhanh với kịch bản mặc định)
python react_agent.py
python plan_execute_agent.py
python hybrid_agent.py

# Các lệnh để đánh giá chi tiết
# Đánh giá 3 mẫu trên 9 kịch bản, mỗi cái chạy 3 lần
python evaluate.py

# Chạy với model khác và số lần tuỳ chỉnh
python evaluate.py --model openai:gpt-4o-mini --runs 5

# Chạy so sánh BẬT/TẮT harness (Ablation study)
python evaluate.py --ablation

# Xem audit log chi tiết 1 lần chạy của kịch bản S2 bằng agent hybrid
python evaluate.py --trace S2 hybrid_agent

# Xuất kết quả đánh giá ra định dạng JSON
python evaluate.py --json
```

Nếu máy không có `py`, dùng `python -m venv .venv`.

### Cách đọc trạng thái

| Trạng thái | Ý nghĩa |
|---|---|
| `finished` | Tác vụ báo hoàn thành. Harness lớp 4 (`goal_met`) đã chạy code kiểm tra trong database giả lập và xác nhận kết quả đúng. |
| `handoff` | Tác vụ dừng lại và xin bàn giao cho nhân viên con người (do lỗi cấp quyền, vượt hạn mức, hoặc agent tự đầu hàng). |
| `failed` | Lỗi xảy ra như vượt số lượng tool call, hết quota lập lại kế hoạch, hoặc agent báo xong nhưng bị Harness phát hiện chưa xong. |
| `error` | Lỗi hệ thống, lỗi gọi API LLM. |

## 2. Mục tiêu và nền tảng

Bài làm mô phỏng đặt vé, huỷ vé, đổi vé. Hệ thống có người dùng là khách hàng chuẩn (`Nguyen Van A` - được quyền đặt/huỷ vé mình) và khách vãng lai (`Khach Vang Lai` - chỉ được tìm chuyến).
Agent sử dụng `init_chat_model` (như GPT-4o-mini, OpenRouter) gọi công cụ theo các khuôn mẫu (patterns). `brains.py` được thiết kế để xử lý việc giao tiếp với LLM qua các phương thức `parse`, `decide`, `plan`, `fill_args` bằng `with_structured_output` hoặc `bind_tools`.

## 3. Thiết kế và cài đặt

### 3.1. Mockup công cụ

`MockAirline` lưu chuyến bay, ghế và booking trong bộ nhớ.
Ba công cụ kinh doanh cốt lõi:

| Tool | Vai trò |
|---|---|
| `search_flights` | Tìm chuyến theo tuyến và ngày. Khởi tạo `flight_id` để được phép dùng ở các bước sau. |
| `book_flight` | Đặt vé cho chuyến. Trả về `booking_id`. Có thể văng lỗi `SOLD_OUT` nếu hết chỗ lúc tranh mua. |
| `cancel_booking` | Huỷ booking dựa vào `booking_id`. |

Hai công cụ điều khiển (Control Tools, chỉ dành cho ReAct):
- `finish`: Gọi khi đã hoàn thành tác vụ.
- `handoff_to_human`: Xin dừng vì không đủ quyền, vượt hạn mức hoặc hết cách.

> **Lưu ý kỹ thuật:** Các tool được thiết kế sát với thực tế bằng cách **tự động trích xuất mô tả từ docstring** của hàm (`func.__doc__`) và **định nghĩa schema tham số thông qua Pydantic (`args_schema`)** với các type hint chuẩn. Điều này đảm bảo mô hình LLM có cái nhìn toàn diện, chính xác về từng tham số thay vì dùng prompt gán cứng.

### 3.2. Cấu hình tập trung (`config.yaml`)

Toàn bộ tham số hệ thống được tách bạch ra file `config.yaml` (nạp qua `config.py`). Các thiết lập bao gồm:
- **LLM Provider**: Mô hình, nhiệt độ, base URL (hỗ trợ cả OpenAI, Anthropic, OpenRouter).
- **Harness Rules**: Ngân sách (`price_cap`) và hạn mức số lần gọi tool.
- **Phân quyền Agent**: Mỗi agent (`react_agent`, `plan_execute_agent`, `hybrid_agent`) có cấu hình riêng và danh sách `allowed_tools` chuyên biệt, giới hạn đúng công cụ mà agent được phép thấy ở mỗi bối cảnh.

### 3.3. Harness (Lớp vỏ bảo vệ 4 tầng)

Đảm bảo LLM không tự tung tự tác và không bị sụp khi có lỗi. Mọi tool call đều phải đi qua `Harness.run_tool()`:

| Lớp | Nhiệm vụ | Xử lý khi không đạt |
|---|---|---|
| **Lớp 1: Ràng buộc** | Kiểm tra schema bằng Pydantic. Kiểm tra xem `flight_id` gọi để đặt đã từng xuất hiện trong kết quả search trước đó chưa (Chống bịa mã). | Trả về `BAD_ARGS` hoặc `UNKNOWN_FLIGHT` cho LLM tự sửa. |
| **Lớp 2: Kiểm quyền** | Chặn thao tác nằm ngoài scope (vãng lai không được đặt vé) hoặc không phải chính chủ (huỷ booking người khác). | Trả tín hiệu Handoff, bắt buộc dừng chương trình. |
| **Lớp 3: Bàn giao** | Kiểm tra vé có vượt quá hạn mức giá (`PRICE_CAP`). | Trả tín hiệu Handoff, bắt buộc dừng để chờ duyệt. |
| **Lớp 4: Xác thực (Goal)** | Kiểm tra lại database bằng code thay vì tin lời model khi model báo hoàn thành (finish). | Trả về thông báo lỗi để agent thử làm lại hoặc đánh rớt là `false_done`. |

Đồng thời, Harness tự động **thử lại (retry)** trong nội bộ nếu gặp lỗi `Timeout` (đã được cấu hình giả lập).

### 3.4. Ba mẫu agent

Ba mẫu agent giải quyết vấn đề theo 3 chiến thuật khác nhau, nhưng dùng chung Brain và Harness:

| Mẫu | Cơ chế quyết định | Khi gặp lỗi thực thi (ví dụ: hết vé, sai mã) |
|---|---|---|
| **ReAct** (`react_agent.py`) | Model nhìn vào danh sách quan sát và chọn tool ở mỗi bước. | Tự nhìn thấy lỗi trong Observation và tự ra quyết định tiếp theo để sửa đổi. |
| **Plan-then-Execute** (`plan_execute_agent.py`) | Model lập một kế hoạch các bước tĩnh, sau đó chương trình chạy tuần tự. | Kế hoạch là tĩnh, nếu lỗi ở giữa chừng, quá trình thực thi sụp đổ (`failed`). |
| **Lai** (`hybrid_agent.py`) | Model lập kế hoạch. Khi chạy tuần tự, model gọi LLM (`fill_args`) để lấy tham số thực. | Nếu lỗi: Cố gắng retry nội bộ theo kế hoạch hiện tại, nếu vẫn thất bại thì xoá phần dư, lập kế hoạch bù lại (Replan). |

## 4. Phương pháp đánh giá

Sử dụng script `evaluate.py` chạy qua 9 kịch bản (`S1` -> `S9`):
1. **S1 (Happy Path)**: Đặt vé thông thường.
2. **S2 (Race condition)**: Chuyến bay bị mua hết ngay sau khi search. Đòi hỏi phục hồi.
3. **S3 (Timeout)**: Tool timeout 2 lần (harness cứu 1 lần, lần 2 agent phải tự lo).
4. **S4**: Huỷ vé của mình hợp lệ.
5. **S5 (Vi phạm)**: Cố gắng huỷ vé người khác.
6. **S6 (Vi phạm)**: Đặt vé vượt mức 5.000.000đ.
7. **S7 (Vi phạm)**: User vãng lai đòi đặt vé.
8. **S8**: Không có chuyến bay phù hợp (Cần handoff).
9. **S9**: Đổi vé (Huỷ vé cũ, đặt vé mới).

**Chỉ số đánh giá:**
- **Thành công**: Rate hoàn thành đúng kịch bản mong đợi.
- **Vi phạm**: Tỷ lệ làm chuyện cấm (bị Harness chặn nhưng vẫn cố gắng).
- **Báo sai (False Done)**: LLM báo `finish` nhưng Harness kiểm tra bằng code thấy chưa đạt.
- **LLM/Tool Calls**: Đo số lần gọi để xem mô hình nào hiệu quả.
- **Token In/Out & Thời gian**: Ước lượng chi phí.

## 5. Kết quả và phân tích chi tiết

Sau khi chạy thử nghiệm trên mô hình **`gpt-4o-mini`** với 9 kịch bản (mỗi kịch bản 3 lần chạy), đây là kết quả phân tích hệ thống thu được:

![Kết quả Tổng Hợp](result1.png)
*(Hình 1: Bảng tổng hợp kết quả đánh giá chạy trên terminal)*

![Chi tiết Test Case](result2.png)
*(Hình 2: Bảng chi tiết kết quả từng Test Case theo từng Agent)*

### Phân tích số liệu thực tế:

**1. React Agent (Phản xạ)**
- **Độ linh hoạt rất cao:** React dễ dàng vượt qua các kịch bản hóc búa nhất (như S2 - Chuyến rẻ nhất bị mua hết, S3 - Timeout) do nó tự đọc lỗi (Observation) và tự quyết định lại ở vòng lặp tiếp theo.
- **Nhược điểm:** Tốn kém chi phí (Tokens) và thời gian chạy dài nhất, do tại mỗi bước, nó đều phải gửi toàn bộ lịch sử (Observation) về cho LLM suy luận lại bước tiếp theo. Đôi lúc nó đi luẩn quẩn trước khi tìm ra công cụ đúng.

**2. Plan-then-Execute Agent (Lập kế hoạch trước)**
- **Ưu điểm:** Rất tiết kiệm tokens và chạy cực nhanh ở các kịch bản suôn sẻ (Happy Path như S1, S4) vì chỉ gọi LLM một lần để lên lịch, sau đó code chạy thẳng theo thứ tự.
- **Nhược điểm chí mạng:** Thiếu linh hoạt. Nếu có sự cố (ví dụ chuyến bay bị mua mất ở kịch bản S2), kế hoạch tĩnh không biết cách thay đổi, dẫn đến chương trình sụp đổ (Failed). 

**3. Hybrid Agent (Lai)**
- **Cân bằng tối ưu:** Vừa có tính định hướng nhờ việc lên kế hoạch trước, vừa linh hoạt. Khi một bước trong kế hoạch thất bại (S2), Hybrid bắt được lỗi, tự động loại bỏ các bước thừa và gọi LLM để lập kế hoạch bù đắp lại (Replan).
- Hiệu năng (Token & Time) và tỉ lệ thành công (Success Rate) nằm ở mức tối ưu nhất trong 3 Agent. Nó mang lại độ tin cậy của React với chi phí rẻ như Plan-Execute.

### Phân tích vai trò của lớp vỏ Harness:
Qua các kịch bản S5, S6, S7 (các hành động bất hợp pháp hoặc vượt quyền), **Harness đóng vai trò là một tường lửa vững chắc (hiển thị qua các cột L1, L2 trên báo cáo)**:
- Các cột **L1 (Dữ liệu)** và **L2 (Quyền)** trong bảng tổng hợp hiển thị **số lần trung bình** Harness phải ra tay can thiệp chặn lỗi của LLM trong mỗi lần chạy.
- **L1 (Dữ liệu):** Hiển thị số lần Lớp 1 (Ràng buộc dữ liệu) từ chối (reject) tham số đầu vào do LLM truyền sai hoặc bịa mã chuyến bay. Nhờ bị chặn lại, LLM có cơ hội nhìn thấy lỗi và gọi lại cho đúng.
- **L2 (Quyền):** Hiển thị số lần Lớp 2 (Kiểm quyền) lập tức chặn đứng (block) Agent khi nó cố gắng đặt vé cho "Khách Vãng Lai" (S7) hoặc cố huỷ vé của người khác (S5), sau đó ép Agent phải "Handoff" (bàn giao).
- Lớp 3 (Bàn giao) hoạt động hoàn hảo ở S6 khi tự động tính toán tổng hoá đơn và từ chối các booking vượt quá 5 triệu VNĐ.
- Nhờ có các màng lọc này, tỷ lệ **"Vi phạm"** của tất cả các Agent đều được ép về **0%**, chứng minh LLM hoàn toàn bị cô lập và vô hại với cơ sở dữ liệu nếu có hành vi vượt quyền.

## 6. Kiểm chứng và giới hạn

Code được tổ chức phân lập (Separation of concerns). LLM chỉ được dùng để quyết định hành động, không thực hiện hành động. Dữ liệu chuyến bay và trạng thái đều giữ kín ở cấp Backend (MockAirline).
*Giới hạn*: Chỉ giả lập API; Prompting bằng tiếng Việt, một số model yếu có thể gặp khó khăn trong việc hiểu tool schema.
