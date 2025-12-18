"""
调用本地 FastAPI 反欺诈检测接口的示例脚本
"""
import json
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"
API_URL = f"{BASE_URL}/api/v1/fraud/detect"


def main():
    # 构造一个测试样例, 模拟一笔较大金额、设备/IP 略可疑的交易
    payload = {
        "transaction_id": "test_api_001",
        "user_id": "user_10086",
        "amount": 12888.0,
        "transaction_type": "payment",
        "merchant_id": "merchant_2001",
        "merchant_category": "MCC_5999",
        "device_id": "device_abc123",
        "ip_address": "10.23.45.67",
        "location": "CN",
        "detection_mode": "llm"
    }

    print("请求 URL:", API_URL)
    print("请求报文:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    try:
        resp = requests.post(API_URL, json=payload, timeout=30)
    except Exception as e:
        print("请求失败:", e)
        return

    print("\nHTTP 状态码:", resp.status_code)
    try:
        data = resp.json()
    except Exception:
        print("响应文本:", resp.text)
        return

    print("响应 JSON:")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
