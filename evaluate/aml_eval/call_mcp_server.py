# 用来测试接口能不能跑通

import requests
import json

BASE_URL = "http://localhost:8000"

def get_model_info():
    """调用 get_model_info 工具"""
    payload = {
        "method": "tools/call",
        "params": {
            "name": "get_model_info",
            "arguments": {},
            "_meta": {
                "progressToken": 2
            }
        }
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(f"{BASE_URL}/messages", json=payload, headers=headers)
        print("✅ get_model_info 成功结果:", response.json())
        print()
    except Exception as e:
        print("❌ get_model_info 失败结果: ", e)
        print()

def predict_customer_risk():
    """调用 predict_customer_risk 工具"""
    payload = {
        "method": "tools/call",
        "params": {
            "name": "predict_customer_risk",
            "arguments": {
                "customer_data": {
                    "customer_id": "CUST123",
                    "name": "John Doe",
                    "age": 45,
                    "nationality": "United States",
                    "occupation": "Business Owner",
                    "account_opening_date": "2020-01-15",
                    "pep_status": 0,
                    "sanctions_match": 0,
                    "address_change_count": 1
                },
                "transactions": [
                    {
                        "transaction_id": "TXN123",
                        "customer_id": "CUST123",
                        "transaction_date": "2022-01-01",
                        "amount": 1000,
                        "transaction_type": "deposit",  # "deposit", "withdrawal", "transfer"
                        "is_cash_transaction": 0,
                        "is_cross_border": 0,
                        "country_code": "US"
                    }
                ]
            },
            "_meta": {
                "progressToken": 2
            }
        }
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(f"{BASE_URL}/messages", json=payload, headers=headers)
        print("✅ predict_customer_risk 成功结果:", response.json())
        print()
    except Exception as e:
        print("❌ predict_customer_risk 失败结果: ", e)
        print()

def pridict_batch_customers():
    """调用 pridict_batch_customers 工具"""
    payload = {
        "method": "tools/call",
        "params": {
            "name": "pridict_batch_customers",
            "arguments": {
                "customer_data": [
                    {
                        "customer_id": "CUST123",
                        "name": "John Doe",
                        "age": 45,
                        "nationality": "United States",
                        "occupation": "Business Owner",
                        "account_opening_date": "2020-01-15",
                        "pep_status": 0,
                        "sanctions_match": 0,
                        "address_change_count": 1
                    }
                ]
            },
            "_meta": {
                "progressToken": 2
            }
        }
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(f"{BASE_URL}/messages", json=payload, headers=headers)
        print("✅ pridict_batch_customers 成功结果:", response.json())
        print()
    except Exception as e:
        print("❌ pridict_batch_customers 失败结果: ", e)
        print()
        return {
                "error": "Failed to call pridict_batch_customers tool"
            }
    
def caclulate_transaction_risk():
    """调用 caclulate_transaction_risk 工具"""
    payload = {
        "method": "tools/call",
        "params": {
            "name": "caclulate_transaction_risk",
            "arguments": {
                "transactions": [
                    {
                        "transaction_id": "TXN123",
                        "customer_id": "CUST123",
                        "transaction_date": "2022-01-01",
                        "amount": 1000,
                        "transaction_type": "deposit",  # "deposit", "withdrawal", "transfer"
                        "is_cash_transaction": 0,
                        "is_cross_border": 0,
                        "country_code": "US"
                    }
                ]
            },
            "_meta": {
                "progressToken": 2
            }
        }
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(f"{BASE_URL}/messages", json=payload, headers=headers)
        print("✅ caclulate_transaction_risk 成功结果:", response.json())
        print()
    except Exception as e:
        print("❌ caclulate_transaction_risk 失败结果: ", e)
        print()
        return {
                "error": "Failed to call caclulate_transaction_risk tool"
            }