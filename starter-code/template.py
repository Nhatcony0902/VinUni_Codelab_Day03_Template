"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple

from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

AIRPORT_CODES = ["SGN", "HAN", "DAD"]
FAQ_ANSWER = (
    "Chính sách Vinpearl: vé máy bay trong combo Vinpearl được đổi/hoàn theo điều kiện "
    "hạng vé của hãng bay; vui lòng liên hệ hotline Vinpearl để được hỗ trợ chi tiết."
)


class ChatbotBaseline:
    """Baseline LLM Chatbot without ReAct Loop or Tools"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn du lịch. Hãy trả lời KHÔNG dùng tool hay internet: {user_input}"
                )
                return {
                    "answer": response.text,
                    "status": "success",
                    "tool_calls": []
                }
            except Exception as e:
                return {
                    "answer": f"[Chatbot Baseline] Lỗi gọi Gemini: {e}",
                    "status": "error",
                    "tool_calls": []
                }

        # Không có API key: chatbot không có dữ liệu thật nên chỉ trả lời chung chung
        return {
            "answer": (
                f"[Chatbot Baseline] Trả lời cho: {user_input}\n"
                "Tôi không có quyền truy cập dữ liệu chuyến bay hay thời tiết thời gian thực, "
                "nên không thể cung cấp thông tin chính xác."
            ),
            "status": "success",
            "tool_calls": []
        }


class ReActAgent:
    """Production-grade ReAct Agent with Tool Registry and Safeguards"""
    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    def parse_city_code(self, text: str) -> str:
        text_upper = text.upper()
        for code in AIRPORT_CODES:
            if code in text_upper:
                return code
        if "HÀ NỘI" in text_upper:
            return "HAN"
        if "HỒ CHÍ MINH" in text_upper or "SÀI GÒN" in text_upper:
            return "SGN"
        if "ĐÀ NẴNG" in text_upper:
            return "DAD"
        return "SGN"

    def parse_max_price(self, text: str) -> int:
        """Đọc ngân sách: '2 triệu' -> 2000000, '1.5 triệu' -> 1500000, '500k' -> 500000."""
        text_lower = text.lower()
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*triệu", text_lower)
        if match:
            return int(float(match.group(1).replace(",", ".")) * 1_000_000)
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*k\b", text_lower)
        if match:
            return int(float(match.group(1).replace(",", ".")) * 1_000)
        return 5_000_000

    def detect_tool_plan(self, user_input: str) -> List[Dict[str, Any]]:
        """Thought: xác định các Action cần gọi theo thứ tự."""
        text_upper = user_input.upper()
        plan = []

        codes = re.findall(r"\b(" + "|".join(AIRPORT_CODES) + r")\b", text_upper)
        if "CHUYẾN BAY" in text_upper and len(codes) >= 2:
            plan.append({
                "name": "get_flight_info",
                "args": {
                    "origin": codes[0],
                    "destination": codes[1],
                    "max_price": self.parse_max_price(user_input)
                }
            })

        if "THỜI TIẾT" in text_upper or "MẶC GÌ" in text_upper:
            weather_part = text_upper.split("THỜI TIẾT", 1)[-1]
            plan.append({
                "name": "get_weather_forecast",
                "args": {"city_code": self.parse_city_code(weather_part)}
            })
        return plan

    def execute_action(self, action_text: str) -> Any:
        """Parse Action JSON và gọi tool trong TOOL_MAP (Trap 1 + Trap 2)."""
        try:
            action = json.loads(action_text)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format"}

        tool_name = str(action.get("name", "")).strip().lower()
        tool_fn = TOOL_MAP.get(tool_name)
        if tool_fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return tool_fn(**action.get("args", {}))
        except TypeError as e:
            return {"error": f"Invalid arguments for {tool_name}: {e}"}

    def format_observation(self, tool_name: str, observation: Any) -> str:
        """Chuyển Observation thành 1 mục trong Final Answer (tiêu đề + các dòng '   - ')."""
        if tool_name == "get_flight_info":
            title = "Thông tin chuyến bay:"
            if isinstance(observation, dict) and "error" in observation:
                lines = [f"Không lấy được dữ liệu chuyến bay ({observation['error']})."]
            elif not observation:
                lines = ["Không tìm thấy chuyến bay phù hợp với ngân sách."]
            else:
                lines = [
                    f"{fl['airline']} ({fl['flight_number']}): {fl['departure_time']} "
                    f"- Giá: {fl['price_vnd']:,} VNĐ"
                    for fl in observation
                ]
        elif tool_name == "get_weather_forecast":
            title = "Thông tin thời tiết & trang phục:"
            if "error" in observation:
                lines = [f"Không lấy được dữ liệu thời tiết ({observation['error']})."]
            else:
                lines = [
                    f"Thời tiết tại {observation['city']}: "
                    f"{observation['temperature_c']}°C ({observation['condition']}).",
                    f"Gợi ý trang phục: {observation['recommendation']}"
                ]
        else:
            title = f"Kết quả {tool_name}:"
            lines = [json.dumps(observation, ensure_ascii=False)]
        return title + "\n" + "\n".join(f"   - {line}" for line in lines)

    def build_final_answer(self) -> str:
        sections = [
            self.format_observation(step["action"]["name"], step["observation"])
            for step in self.trace
            if step.get("action")
        ]
        if len(sections) == 1:
            return sections[0]
        return "\n\n".join(f"{i}. {section}" for i, section in enumerate(sections, start=1))

    def build_thought(self, action: Dict[str, Any]) -> str:
        args = action["args"]
        if action["name"] == "get_flight_info":
            return (
                f"Tôi cần tìm chuyến bay từ {args['origin']} đến {args['destination']} "
                f"với giá dưới {args['max_price']:,} VNĐ."
            )
        if action["name"] == "get_weather_forecast":
            return f"Tôi cần kiểm tra thông tin thời tiết tại {args['city_code']}."
        return f"Tôi cần gọi {action['name']}."

    def plan_and_execute_step(self, user_input: str, iteration: int) -> Tuple[str, bool]:
        """
        Thực hiện 1 bước Thought -> Action -> Observation.
        Trả về (nội dung, is_final). Với câu hỏi cần nhiều tool, bước cuối là bước tổng hợp
        riêng; câu hỏi chỉ cần 0-1 tool thì trả Final Answer ngay trong bước đó.
        """
        plan = self.detect_tool_plan(user_input)
        step_index = iteration - 1

        # Không cần tool: trả lời FAQ trực tiếp
        if not plan:
            self.trace.append({
                "iteration": iteration,
                "thought": "Câu hỏi không cần tra cứu dữ liệu, tôi có thể trả lời trực tiếp.",
                "final_answer": FAQ_ANSWER
            })
            return FAQ_ANSWER, True

        # Còn Action chưa thực hiện
        if step_index < len(plan):
            action = plan[step_index]
            action_text = json.dumps(action, ensure_ascii=False)
            observation = self.execute_action(action_text)
            step = {
                "iteration": iteration,
                "thought": self.build_thought(action),
                "action": action,
                "observation": observation
            }
            self.trace.append(step)
            if len(plan) == 1:
                step["final_answer"] = self.build_final_answer()
                return step["final_answer"], True
            return json.dumps(observation, ensure_ascii=False), False

        # Đã có đủ Observation: tổng hợp Final Answer
        final_answer = self.build_final_answer()
        self.trace.append({
            "iteration": iteration,
            "thought": "Tôi đã thu thập đủ thông tin để trả lời khách hàng.",
            "final_answer": final_answer
        })
        return final_answer, True

    def run(self, user_input: str) -> Dict[str, Any]:
        self.trace = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            content, is_final = self.plan_and_execute_step(user_input, iteration)
            if is_final:
                return {
                    "status": "completed",
                    "answer": content,
                    "iterations": iteration,
                    "trace": self.trace
                }

        return {
            "status": "max_iterations_reached",
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "iterations": iteration,
            "trace": self.trace
        }


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query)["answer"])

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
