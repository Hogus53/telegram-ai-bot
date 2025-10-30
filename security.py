#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔐 نظام الأمان والتشفير القوي
Encryption & Security System
"""

import os
import hashlib
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import base64
from typing import Optional, Dict, List
import json
from datetime import datetime, timedelta
import secrets

class SecurityManager:
    """مدير الأمان والتشفير"""
    
    def __init__(self, master_password: Optional[str] = None):
        """
        تهيئة مدير الأمان
        
        Args:
            master_password: كلمة المرور الرئيسية
        """
        self.master_password = master_password or os.environ.get("MASTER_PASSWORD", "default_secure_password")
        self.encryption_key = self._generate_key()
        self.cipher = Fernet(self.encryption_key)
        
    def _generate_key(self) -> bytes:
        """توليد مفتاح التشفير"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'telegram_ai_bot_salt_2024',
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_password.encode()))
        return key
    
    def encrypt(self, data: str) -> str:
        """تشفير البيانات"""
        try:
            encrypted = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            raise Exception(f"خطأ في التشفير: {str(e)}")
    
    def decrypt(self, encrypted_data: str) -> str:
        """فك تشفير البيانات"""
        try:
            encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            raise Exception(f"خطأ في فك التشفير: {str(e)}")
    
    def hash_password(self, password: str) -> str:
        """تجزئة كلمة المرور"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hash_value: str) -> bool:
        """التحقق من كلمة المرور"""
        return self.hash_password(password) == hash_value
    
    def generate_token(self, length: int = 32) -> str:
        """توليد رمز آمن"""
        return secrets.token_urlsafe(length)
    
    def create_signature(self, data: str) -> str:
        """إنشاء توقيع رقمي"""
        return hmac.new(
            self.master_password.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_signature(self, data: str, signature: str) -> bool:
        """التحقق من التوقيع الرقمي"""
        return hmac.compare_digest(
            self.create_signature(data),
            signature
        )

class OwnerVerification:
    """نظام التحقق من المالك"""
    
    def __init__(self, owner_id: str, security_manager: SecurityManager):
        """
        تهيئة نظام التحقق من المالك
        
        Args:
            owner_id: معرف المالك
            security_manager: مدير الأمان
        """
        self.owner_id = owner_id
        self.security_manager = security_manager
        self.verified_sessions: Dict[str, Dict] = {}
        self.access_logs: List[Dict] = []
        
    def create_session(self, user_id: str, password: str) -> Optional[str]:
        """إنشاء جلسة تحقق"""
        if user_id != self.owner_id:
            self._log_access(user_id, "failed", "غير مصرح")
            return None
        
        # التحقق من كلمة المرور
        owner_password_hash = os.environ.get("OWNER_PASSWORD_HASH")
        if not owner_password_hash:
            owner_password_hash = self.security_manager.hash_password("default_owner_password")
        
        if not self.security_manager.verify_password(password, owner_password_hash):
            self._log_access(user_id, "failed", "كلمة مرور خاطئة")
            return None
        
        # إنشاء رمز جلسة
        session_token = self.security_manager.generate_token()
        self.verified_sessions[session_token] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=24),
            "permissions": ["full_control"]
        }
        
        self._log_access(user_id, "success", "تم التحقق بنجاح")
        return session_token
    
    def verify_session(self, session_token: str) -> bool:
        """التحقق من صحة الجلسة"""
        if session_token not in self.verified_sessions:
            return False
        
        session = self.verified_sessions[session_token]
        if datetime.now() > session["expires_at"]:
            del self.verified_sessions[session_token]
            return False
        
        return True
    
    def get_user_permissions(self, session_token: str) -> List[str]:
        """الحصول على صلاحيات المستخدم"""
        if not self.verify_session(session_token):
            return []
        
        return self.verified_sessions[session_token].get("permissions", [])
    
    def has_permission(self, session_token: str, permission: str) -> bool:
        """التحقق من وجود صلاحية معينة"""
        permissions = self.get_user_permissions(session_token)
        return "full_control" in permissions or permission in permissions
    
    def _log_access(self, user_id: str, status: str, reason: str):
        """تسجيل محاولة الوصول"""
        self.access_logs.append({
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "reason": reason
        })
        
        # الاحتفاظ بآخر 1000 سجل فقط
        if len(self.access_logs) > 1000:
            self.access_logs = self.access_logs[-1000:]
    
    def get_access_logs(self) -> List[Dict]:
        """الحصول على سجلات الوصول"""
        return self.access_logs

class DataProtection:
    """حماية البيانات"""
    
    def __init__(self, security_manager: SecurityManager):
        """تهيئة حماية البيانات"""
        self.security_manager = security_manager
        
    def protect_sensitive_data(self, data: Dict) -> Dict:
        """حماية البيانات الحساسة"""
        protected_data = {}
        
        for key, value in data.items():
            if key in ["password", "token", "api_key", "secret"]:
                protected_data[key] = self.security_manager.encrypt(str(value))
            else:
                protected_data[key] = value
        
        return protected_data
    
    def unprotect_sensitive_data(self, data: Dict) -> Dict:
        """فك حماية البيانات الحساسة"""
        unprotected_data = {}
        
        for key, value in data.items():
            if key in ["password", "token", "api_key", "secret"]:
                try:
                    unprotected_data[key] = self.security_manager.decrypt(value)
                except:
                    unprotected_data[key] = value
            else:
                unprotected_data[key] = value
        
        return unprotected_data
    
    def sanitize_input(self, user_input: str) -> str:
        """تنظيف مدخلات المستخدم"""
        # إزالة الأحرف الخطرة
        dangerous_chars = ["<", ">", "&", "\"", "'", ";", "\\"]
        sanitized = user_input
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")
        
        return sanitized

class AuditLog:
    """سجل التدقيق"""
    
    def __init__(self):
        """تهيئة سجل التدقيق"""
        self.logs: List[Dict] = []
        
    def log_action(self, user_id: str, action: str, details: Dict, status: str = "success"):
        """تسجيل إجراء"""
        log_entry = {
            "user_id": user_id,
            "action": action,
            "details": details,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logs.append(log_entry)
        
        # الاحتفاظ بآخر 10000 سجل
        if len(self.logs) > 10000:
            self.logs = self.logs[-10000:]
    
    def get_logs(self, user_id: Optional[str] = None) -> List[Dict]:
        """الحصول على السجلات"""
        if user_id:
            return [log for log in self.logs if log["user_id"] == user_id]
        return self.logs
    
    def export_logs(self, filepath: str):
        """تصدير السجلات"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

# إنشاء مثيلات عامة
security_manager = SecurityManager()
audit_log = AuditLog()
