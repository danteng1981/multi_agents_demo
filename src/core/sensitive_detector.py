"""
Sensitive Data Detection and Masking
"""
import re
from typing import Dict, List


class SensitiveDataDetector:
    """敏感信息检测器"""
    
    # 敏感信息正则表达式模式
    PATTERNS = {
        'phone': r'1[3-9]\d{9}',
        'id_card': r'\d{17}[\dXx]',
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'credit_card': r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
        'ip_address': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
        'api_key': r'(api[_-]?key|token|secret)["\']?\s*[: =]\s*["\']? ([a-zA-Z0-9_-]{20,})',
        'password': r'(password|passwd|pwd)["\']?\s*[:=]\s*["\']?([^\s"\']{6,})',
        'bank_card': r'\d{16,19}',
    }
    
    def detect(self, text: str) -> Dict[str, List[str]]:
        """
        检测文本中的敏感信息
        
        Args:
            text: 要检测的文本
            
        Returns:
            检测到的敏感信息字典 {类型: [匹配列表]}
        """
        detected = {}
        
        for data_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # 处理元组匹配（如api_key返回元组）
                if isinstance(matches[0], tuple):
                    matches = [m[1] if len(m) > 1 else m[0] for m in matches]
                detected[data_type] = matches
        
        return detected
    
    def mask(self, text:  str, mask_char: str = '*') -> str:
        """
        脱敏处理
        
        Args: 
            text: 要脱敏的文本
            mask_char: 掩码字符
            
        Returns:
            脱敏后的文本
        """
        masked_text = text
        
        # 手机号脱敏：138****5678
        masked_text = re.sub(
            r'(1[3-9]\d)\d{4}(\d{4})',
            r'\1****\2',
            masked_text
        )
        
        # 身份证脱敏：110***********1234
        masked_text = re. sub(
            r'(\d{3})\d{11}(\d{4})',
            r'\1***********\2',
            masked_text
        )
        
        # 邮箱脱敏：abc***@example.com
        masked_text = re.sub(
            r'([a-zA-Z0-9._%+-]{1,3})[a-zA-Z0-9._%+-]*(@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            r'\1***\2',
            masked_text
        )
        
        # 信用卡脱敏：1234 **** **** 5678
        masked_text = re.sub(
            r'(\d{4})[-\s]?\d{4}[-\s]?\d{4}[-\s]? (\d{4})',
            r'\1 **** **** \2',
            masked_text
        )
        
        # 银行卡脱敏
        masked_text = re.sub(
            r'(\d{4})\d{8,11}(\d{4})',
            r'\1********\2',
            masked_text
        )
        
        # IP地址脱敏：192.168.***. ***
        masked_text = re.sub(
            r'(\d{1,3}\.\d{1,3}\. )\d{1,3}\.\d{1,3}',
            r'\1***.  ***',
            masked_text
        )
        
        # API Key脱敏：只显示前后4位
        def mask_api_key(match):
            key = match.group(2)
            if len(key) > 8:
                return f"{match.group(1)}={key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"
            return match.group(0)
        
        masked_text = re.sub(
            r'(api[_-]? key|token|secret)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})',
            mask_api_key,
            masked_text,
            flags=re.IGNORECASE
        )
        
        return masked_text
    
    def is_sensitive(self, text: str) -> bool:
        """
        检查文本是否包含敏感信息
        
        Args:
            text:  要检查的文本
            
        Returns:
            是否包含敏感信息
        """
        detected = self.detect(text)
        return len(detected) > 0
    
    def get_sensitivity_report(self, text: str) -> dict:
        """
        生成敏感信息报告
        
        Args:
            text: 要分析的文本
            
        Returns:
            敏感信息报告
        """
        detected = self.detect(text)
        
        return {
            "is_sensitive": len(detected) > 0,
            "detected_types": list(detected.keys()),
            "total_matches": sum(len(matches) for matches in detected.values()),
            "details": detected,
            "masked_text": self.mask(text) if detected else text
        }


# 全局检测器实例
sensitive_detector = SensitiveDataDetector()
