"""Sensitive Data Detector Module

This module provides functionality to detect and mask sensitive information
such as phone numbers, ID cards, emails, credit cards, IP addresses, API keys,
passwords, and bank cards.
"""

import re
from typing import Dict, List, Tuple, Any


class SensitiveDataDetector:
    """Detector for identifying and masking sensitive data in text."""
    
    # Regular expression patterns for different types of sensitive data
    PATTERNS = {
        'phone': [
            r'1[3-9]\d{9}',  # Chinese mobile phone
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # International format
        ],
        'id_card': [
            r'[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]',  # Chinese ID card
            r'\b\d{3}-\d{2}-\d{4}\b',  # US SSN format
        ],
        'email': [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        ],
        'credit_card': [
            r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',  # 16-digit credit card
            r'\b\d{13,19}\b',  # Generic card number
        ],
        'ip_address': [
            r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b',  # IPv4
            r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b',  # IPv6
        ],
        'api_key': [
            r'\b[Aa][Pp][Ii][-_]?[Kk][Ee][Yy]\s*[:=]\s*[\'"]?([A-Za-z0-9_\-]{20,})[\'"\s]?',
            r'\b[Aa]ccess[-_]?[Kk]ey\s*[:=]\s*[\'"]?([A-Za-z0-9_\-]{20,})[\'"\s]?',
            r'\b[Ss]ecret[-_]?[Kk]ey\s*[:=]\s*[\'"]?([A-Za-z0-9_\-]{20,})[\'"\s]?',
        ],
        'password': [
            r'\b[Pp]assword\s*[:=]\s*[\'"]?([^\s\'"]{6,})[\'"\s]?',
            r'\b[Pp]wd\s*[:=]\s*[\'"]?([^\s\'"]{6,})[\'"\s]?',
            r'\b[Pp]ass\s*[:=]\s*[\'"]?([^\s\'"]{6,})[\'"\s]?',
        ],
        'bank_card': [
            r'\b\d{16,19}\b',  # Bank card 16-19 digits
            r'\b(?:\d{4}[-.\s]?){3,4}\d{1,4}\b',  # Formatted bank card
        ],
    }
    
    def __init__(self):
        """Initialize the sensitive data detector."""
        # Compile all patterns for better performance
        self.compiled_patterns = {}
        for data_type, patterns in self.PATTERNS.items():
            self.compiled_patterns[data_type] = [
                re.compile(pattern) for pattern in patterns
            ]
    
    def detect(self, text: str) -> Dict[str, List[str]]:
        """Detect all sensitive data in the given text.
        
        Args:
            text: The text to scan for sensitive data
            
        Returns:
            Dictionary mapping data types to lists of detected values
        """
        if not text:
            return {}
        
        detected = {}
        
        for data_type, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(text)
                if found:
                    # Handle tuple results from groups
                    for match in found:
                        if isinstance(match, tuple):
                            matches.extend([m for m in match if m])
                        else:
                            matches.append(match)
            
            if matches:
                # Remove duplicates while preserving order
                detected[data_type] = list(dict.fromkeys(matches))
        
        return detected
    
    def mask(self, text: str, data_type: str, value: str) -> str:
        """Mask a specific sensitive value in text according to its type.
        
        Args:
            text: The text containing the sensitive data
            data_type: The type of sensitive data
            value: The sensitive value to mask
            
        Returns:
            The text with the sensitive value masked
        """
        if not value or not text:
            return text
        
        masked_value = self._get_masked_value(data_type, value)
        return text.replace(value, masked_value)
    
    def _get_masked_value(self, data_type: str, value: str) -> str:
        """Get the masked version of a sensitive value.
        
        Args:
            data_type: The type of sensitive data
            value: The original sensitive value
            
        Returns:
            The masked version of the value
        """
        if data_type == 'phone':
            # Format: 138****5678
            if len(value) >= 11:
                return f"{value[:3]}****{value[-4:]}"
            elif len(value) >= 7:
                return f"{value[:3]}****{value[-4:]}"
            else:
                return "***" + value[-3:] if len(value) > 3 else "***"
        
        elif data_type == 'id_card':
            # Format: 110***********1234
            if len(value) >= 18:
                return f"{value[:3]}***********{value[-4:]}"
            elif len(value) >= 10:
                return f"{value[:3]}***{value[-4:]}"
            else:
                return "***" + value[-4:] if len(value) > 4 else "***"
        
        elif data_type == 'email':
            # Format: abc***@example.com
            if '@' in value:
                local, domain = value.split('@', 1)
                if len(local) <= 3:
                    masked_local = local[0] + "***"
                else:
                    masked_local = local[:3] + "***"
                return f"{masked_local}@{domain}"
            return "***@***.com"
        
        elif data_type == 'credit_card':
            # Format: 1234 **** **** 5678
            clean = value.replace(' ', '').replace('-', '').replace('.', '')
            if len(clean) >= 13:
                return f"{clean[:4]} **** **** {clean[-4:]}"
            return "**** **** **** ****"
        
        elif data_type == 'ip_address':
            # Format: 192.168.***.***
            parts = value.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.***. ***"
            elif ':' in value:  # IPv6
                return f"{value.split(':')[0]}:***:***:***"
            return "***.***.***.***"
        
        elif data_type == 'api_key':
            # Format: sk_***...(last 4 chars)
            if len(value) > 8:
                return f"{value[:3]}***{value[-4:]}"
            return "***" + value[-4:] if len(value) > 4 else "******"
        
        elif data_type == 'password':
            # Format: ********
            return "*" * min(len(value), 8)
        
        elif data_type == 'bank_card':
            # Format: 6222 **** **** 1234
            clean = value.replace(' ', '').replace('-', '').replace('.', '')
            if len(clean) >= 16:
                return f"{clean[:4]} **** **** {clean[-4:]}"
            return "**** **** **** ****"
        
        else:
            # Default masking
            if len(value) > 8:
                return f"{value[:2]}***{value[-2:]}"
            return "***"
    
    def mask_all(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Mask all detected sensitive data in the text.
        
        Args:
            text: The text to mask
            
        Returns:
            Tuple of (masked_text, count_by_type)
        """
        if not text:
            return text, {}
        
        detected = self.detect(text)
        masked_text = text
        count_by_type = {}
        
        for data_type, values in detected.items():
            count = 0
            for value in values:
                masked_text = self.mask(masked_text, data_type, value)
                count += masked_text.count(self._get_masked_value(data_type, value))
            count_by_type[data_type] = len(values)
        
        return masked_text, count_by_type
    
    def is_sensitive(self, text: str) -> bool:
        """Check if the text contains any sensitive data.
        
        Args:
            text: The text to check
            
        Returns:
            True if sensitive data is detected, False otherwise
        """
        if not text:
            return False
        
        detected = self.detect(text)
        return len(detected) > 0
    
    def get_sensitivity_report(self, text: str) -> Dict[str, Any]:
        """Generate a comprehensive sensitivity report for the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        if not text:
            return {
                'is_sensitive': False,
                'detected_types': [],
                'total_count': 0,
                'details': {},
                'risk_level': 'none'
            }
        
        detected = self.detect(text)
        total_count = sum(len(values) for values in detected.values())
        
        # Calculate risk level
        risk_level = 'none'
        if total_count > 0:
            high_risk_types = {'credit_card', 'bank_card', 'password', 'api_key', 'id_card'}
            has_high_risk = any(dt in detected for dt in high_risk_types)
            
            if has_high_risk or total_count >= 5:
                risk_level = 'high'
            elif total_count >= 3:
                risk_level = 'medium'
            else:
                risk_level = 'low'
        
        # Build detailed report
        details = {}
        for data_type, values in detected.items():
            details[data_type] = {
                'count': len(values),
                'samples': [self._get_masked_value(data_type, v) for v in values[:3]]  # Show up to 3 samples
            }
        
        return {
            'is_sensitive': len(detected) > 0,
            'detected_types': list(detected.keys()),
            'total_count': total_count,
            'details': details,
            'risk_level': risk_level,
            'recommendations': self._get_recommendations(detected)
        }
    
    def _get_recommendations(self, detected: Dict[str, List[str]]) -> List[str]:
        """Generate recommendations based on detected sensitive data.
        
        Args:
            detected: Dictionary of detected sensitive data
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if 'password' in detected:
            recommendations.append("Remove or mask password information immediately")
        
        if 'api_key' in detected:
            recommendations.append("Rotate API keys and store them securely (e.g., environment variables)")
        
        if 'credit_card' in detected or 'bank_card' in detected:
            recommendations.append("Ensure PCI DSS compliance when handling payment card data")
        
        if 'id_card' in detected:
            recommendations.append("Implement strict access controls for personal identification data")
        
        if 'email' in detected or 'phone' in detected:
            recommendations.append("Consider GDPR/privacy law compliance for personal contact information")
        
        if detected:
            recommendations.append("Enable encryption for data at rest and in transit")
            recommendations.append("Implement audit logging for sensitive data access")
        
        return recommendations


# Global instance for easy access
sensitive_detector = SensitiveDataDetector()


if __name__ == "__main__":
    # Example usage and testing
    test_text = """
    Contact: phone=13812345678, email=user@example.com
    ID Card: 110101199003071234
    Credit Card: 4532-1234-5678-9010
    IP Address: 192.168.1.100
    API Key: sk_test_abcdefghijklmnopqrstuvwxyz123456
    Password: MySecret123!
    Bank Card: 6222021234567890123
    """
    
    print("=== Sensitive Data Detection Demo ===")
    print("\nOriginal Text:")
    print(test_text)
    
    print("\n=== Detection Results ===")
    detected = sensitive_detector.detect(test_text)
    for data_type, values in detected.items():
        print(f"\n{data_type.upper()}:")
        for value in values:
            print(f"  - {value}")
    
    print("\n=== Masked Text ===")
    masked_text, counts = sensitive_detector.mask_all(test_text)
    print(masked_text)
    print(f"\nMasked counts: {counts}")
    
    print("\n=== Sensitivity Report ===")
    report = sensitive_detector.get_sensitivity_report(test_text)
    print(f"Is Sensitive: {report['is_sensitive']}")
    print(f"Risk Level: {report['risk_level']}")
    print(f"Total Count: {report['total_count']}")
    print(f"\nDetected Types: {', '.join(report['detected_types'])}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
