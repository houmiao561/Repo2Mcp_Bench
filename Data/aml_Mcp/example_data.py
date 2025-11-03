"""
示例数据 - 用于测试预测器
"""

# 示例客户数据
EXAMPLE_CUSTOMER = {
    'customer_id': 'CUST000001',
    'name': 'John Doe',
    'age': 45,
    'nationality': 'United States',
    'occupation': 'Business Owner',
    'account_opening_date': '2020-01-15',
    'pep_status': 0,
    'sanctions_match': 0,
    'address_change_count': 1
}

# 示例可疑客户数据
EXAMPLE_SUSPICIOUS_CUSTOMER = {
    'customer_id': 'CUST000002',
    'name': 'Jane Smith',
    'age': 52,
    'nationality': 'Switzerland',
    'occupation': 'Business Owner',
    'account_opening_date': '2021-06-10',
    'pep_status': 1,  # PEP身份
    'sanctions_match': 0,
    'address_change_count': 5  # 频繁更换地址
}

# 示例正常交易
EXAMPLE_NORMAL_TRANSACTIONS = [
    {
        'transaction_id': 'TXN00000001',
        'customer_id': 'CUST000001',
        'transaction_date': '2024-01-10 10:30:00',
        'amount': 500.00,
        'transaction_type': 'deposit',
        'is_cash_transaction': 0,
        'is_cross_border': 0,
        'country_code': 'US'
    },
    {
        'transaction_id': 'TXN00000002',
        'customer_id': 'CUST000001',
        'transaction_date': '2024-01-15 14:20:00',
        'amount': 200.00,
        'transaction_type': 'withdrawal',
        'is_cash_transaction': 0,
        'is_cross_border': 0,
        'country_code': 'US'
    },
    {
        'transaction_id': 'TXN00000003',
        'customer_id': 'CUST000001',
        'transaction_date': '2024-01-20 09:15:00',
        'amount': 1000.00,
        'transaction_type': 'transfer',
        'is_cash_transaction': 0,
        'is_cross_border': 0,
        'country_code': 'US'
    }
]

# 示例可疑交易
EXAMPLE_SUSPICIOUS_TRANSACTIONS = [
    {
        'transaction_id': 'TXN00000004',
        'customer_id': 'CUST000002',
        'transaction_date': '2024-01-05 11:00:00',
        'amount': 15000.00,  # 大额
        'transaction_type': 'deposit',
        'is_cash_transaction': 1,  # 现金交易
        'is_cross_border': 0,
        'country_code': 'CH'
    },
    {
        'transaction_id': 'TXN00000005',
        'customer_id': 'CUST000002',
        'transaction_date': '2024-01-06 15:30:00',
        'amount': 8000.00,
        'transaction_type': 'transfer',
        'is_cash_transaction': 0,
        'is_cross_border': 1,  # 跨境
        'country_code': 'KY'  # 开曼群岛
    },
    {
        'transaction_id': 'TXN00000006',
        'customer_id': 'CUST000002',
        'transaction_date': '2024-01-07 10:00:00',
        'amount': 12000.00,
        'transaction_type': 'withdrawal',
        'is_cash_transaction': 1,  # 现金取款
        'is_cross_border': 0,
        'country_code': 'CH'
    },
    {
        'transaction_id': 'TXN00000007',
        'customer_id': 'CUST000002',
        'transaction_date': '2024-01-08 16:45:00',
        'amount': 20000.00,
        'transaction_type': 'transfer',
        'is_cash_transaction': 0,
        'is_cross_border': 1,
        'country_code': 'PA'  # 巴拿马
    }
]

# 国家风险映射示例
EXAMPLE_COUNTRY_RISK_MAPPING = {
    'US': 0,   # low risk
    'GB': 0,   # low risk
    'CA': 0,   # low risk
    'FR': 0,   # low risk
    'DE': 0,   # low risk
    'CN': 1,   # medium risk
    'RU': 1,   # medium risk
    'BR': 1,   # medium risk
    'CH': 1,   # medium risk
    'AF': 2,   # high risk
    'KP': 2,   # high risk
    'SY': 2,   # high risk
    'IR': 2,   # high risk
    'KY': 2,   # high risk (开曼群岛 - 避税天堂)
    'PA': 2,   # high risk (巴拿马 - 避税天堂)
}

# 批量测试数据
BATCH_CUSTOMERS = [
    EXAMPLE_CUSTOMER,
    EXAMPLE_SUSPICIOUS_CUSTOMER,
    {
        'customer_id': 'CUST000003',
        'name': 'Bob Johnson',
        'age': 35,
        'nationality': 'Canada',
        'occupation': 'Engineer',
        'account_opening_date': '2019-03-20',
        'pep_status': 0,
        'sanctions_match': 0,
        'address_change_count': 0
    }
]

BATCH_TRANSACTIONS = EXAMPLE_NORMAL_TRANSACTIONS + EXAMPLE_SUSPICIOUS_TRANSACTIONS + [
    {
        'transaction_id': 'TXN00000008',
        'customer_id': 'CUST000003',
        'transaction_date': '2024-01-12 12:00:00',
        'amount': 300.00,
        'transaction_type': 'payment',
        'is_cash_transaction': 0,
        'is_cross_border': 0,
        'country_code': 'CA'
    }
]


