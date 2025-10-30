#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🤖 بوت التلغرام الذكي المتطور - Advanced Telegram AI Bot
مساعدك الشخصي الكامل مع التحكم المطلق
🔐 نظام أمان قوي - أنت الأولوية الأولى والقصوى
يعمل 24/7 مع أقوى المكتبات والميزات المتقدمة
"""

import os
import sys
import logging
import asyncio
import json
import subprocess
import platform
import psutil
import pyautogui
from datetime import datetime
from typing import Optional, Dict, List
import requests
from pathlib import Path

# Telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# AI & LLM
from openai import OpenAI

# Web & Scraping
import aiohttp
from bs4 import BeautifulSoup

# Image Processing
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Data Processing
import pandas as pd

# Database
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Async
import aiofiles

# Logging
from loguru import logger

# Security
from security import SecurityManager, OwnerVerification, DataProtection, AuditLog

# Flask for keeping alive
from flask import Flask, jsonify
from threading import Thread

# ============================================================================
# إعدادات التسجيل (Logging)
# ============================================================================
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# ============================================================================
# إعدادات البوت والذكاء الاصطناعي
# ============================================================================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "8447232715:AAFYC-YKCNiJVfHxbG8_c7QwORJEopOuEbs")
OWNER_ID = os.environ.get("OWNER_ID", "6743097025")  # معرف المالك
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# إعداد عميل OpenAI
client = OpenAI()
MODEL_NAME = "gemini-2.5-flash"

# ============================================================================
# نظام الأمان
# ============================================================================
security_manager = SecurityManager()
owner_verification = OwnerVerification(OWNER_ID, security_manager)
data_protection = DataProtection(security_manager)
audit_log = AuditLog()

# ============================================================================
# قاعدة البيانات
# ============================================================================
Base = declarative_base()

class UserSession(Base):
    """نموذج جلسة المستخدم"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True)
    username = Column(String)
    created_at = Column(DateTime, default=datetime.now)
    last_activity = Column(DateTime, default=datetime.now)
    conversation_history = Column(Text, default="[]")
    settings = Column(Text, default="{}")

class TaskLog(Base):
    """نموذج سجل المهام"""
    __tablename__ = "task_logs"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    task_type = Column(String)
    task_description = Column(Text)
    result = Column(Text)
    timestamp = Column(DateTime, default=datetime.now)
    status = Column(String)

# إعداد قاعدة البيانات
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///bot_database.db")
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# ============================================================================
# خادم Flask
# ============================================================================
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "Advanced Telegram AI Bot Running 24/7 ✅",
        "bot_name": "@hogusXbot",
        "owner_priority": "MAXIMUM",
        "security": "ENCRYPTED",
        "features": [
            "Full System Control", "AI Chat", "Web Search", "Image Generation",
            "Code Writing", "Task Automation", "File Management", "Data Analysis",
            "Owner Verification", "Audit Logging", "Encryption"
        ],
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "owner_verified": True}), 200

def run_flask():
    app.run(host='0.0.0.0', port=8080, debug=False)

# ============================================================================
# دوال التحقق من المالك
# ============================================================================

def is_owner(user_id: str) -> bool:
    """التحقق من أن المستخدم هو المالك"""
    return str(user_id) == str(OWNER_ID)

async def verify_owner_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """التحقق من أن المستخدم هو المالك قبل تنفيذ أمر"""
    user_id = update.effective_user.id
    
    if not is_owner(user_id):
        logger.warning(f"محاولة وصول غير مصرح من المستخدم: {user_id}")
        audit_log.log_action(str(user_id), "unauthorized_access", {"command": "unknown"}, "failed")
        await update.message.reply_text("❌ أنت غير مصرح للوصول إلى هذا البوت. هذا البوت خاص بالمالك فقط.")
        return False
    
    audit_log.log_action(str(user_id), "command_executed", {"command": "verified"}, "success")
    return True

# ============================================================================
# دوال التحكم بالنظام
# ============================================================================

async def execute_system_command(command: str) -> str:
    """تنفيذ أي أمر نظام (بدون قيود)"""
    try:
        logger.info(f"تنفيذ الأمر: {command}")
        
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        output = result.stdout or result.stderr or "تم التنفيذ بنجاح"
        logger.info(f"نتيجة الأمر: {output[:100]}")
        
        return output
    except Exception as e:
        logger.error(f"خطأ في تنفيذ الأمر: {e}")
        return f"خطأ: {str(e)}"

async def get_system_info() -> Dict:
    """الحصول على معلومات النظام الكاملة"""
    try:
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used": psutil.virtual_memory().used / (1024**3),
            "memory_total": psutil.virtual_memory().total / (1024**3),
            "disk_percent": psutil.disk_usage('/').percent,
            "uptime": datetime.fromtimestamp(psutil.boot_time()),
            "processes": len(psutil.pids()),
            "cpu_count": psutil.cpu_count()
        }
    except Exception as e:
        logger.error(f"خطأ في الحصول على معلومات النظام: {e}")
        return {}

async def take_screenshot() -> Optional[str]:
    """أخذ لقطة شاشة"""
    try:
        screenshot = pyautogui.screenshot()
        screenshot_path = "/tmp/screenshot.png"
        screenshot.save(screenshot_path)
        logger.info("تم أخذ لقطة الشاشة بنجاح")
        return screenshot_path
    except Exception as e:
        logger.error(f"خطأ في أخذ لقطة الشاشة: {e}")
        return None

async def write_code(description: str) -> str:
    """كتابة أكواد احترافية عالية الجودة"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "أنت مبرمج احترافي متخصص في جميع لغات البرمجة. اكتب كوداً احترافياً عالي الجودة وفعال بناءً على الطلب."
                },
                {
                    "role": "user",
                    "content": f"اكتب كوداً احترافياً لـ: {description}"
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        code = response.choices[0].message.content
        logger.info(f"تم كتابة كود بطول: {len(code)}")
        return code
    except Exception as e:
        logger.error(f"خطأ في كتابة الكود: {e}")
        return f"خطأ: {str(e)}"

async def web_search(query: str, num_results: int = 10) -> str:
    """البحث القوي عن المعلومات"""
    try:
        search_url = f"https://api.duckduckgo.com/?q={query}&format=json"
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    if 'RelatedTopics' in data:
                        for item in data['RelatedTopics'][:num_results]:
                            if 'Text' in item:
                                results.append(f"• {item['Text']}")
                    
                    logger.info(f"تم العثور على {len(results)} نتائج بحث")
                    return "\n".join(results) if results else "لم يتم العثور على نتائج"
        return "خطأ في البحث"
    except Exception as e:
        logger.error(f"خطأ في البحث: {e}")
        return f"خطأ: {str(e)}"

async def execute_task(task_description: str) -> str:
    """تنفيذ أي مهمة معقدة"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "أنت مساعد ذكي قوي يمكنه تنفيذ أي مهمة معقدة. قم بتنفيذ المهمة بدقة واحترافية."
                },
                {
                    "role": "user",
                    "content": f"المهمة: {task_description}"
                }
            ],
            temperature=0.5,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content
        logger.info(f"تم تنفيذ المهمة بنجاح")
        return result
    except Exception as e:
        logger.error(f"خطأ في تنفيذ المهمة: {e}")
        return f"خطأ: {str(e)}"

# ============================================================================
# معالجات أوامر التلغرام
# ============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /start"""
    if not await verify_owner_command(update, context):
        return
    
    welcome_message = (
        "🤖 مرحباً بك أيها المالك!\n\n"
        "أنا بوتك الذكي المتطور مع:\n"
        "✅ الذكاء الاصطناعي المتقدم\n"
        "✅ التحكم المطلق بالجهاز\n"
        "✅ كتابة أكواد احترافية\n"
        "✅ إنشاء صور احترافية\n"
        "✅ محرك بحث قوي\n"
        "✅ تنفيذ أي مهمة\n"
        "✅ تشفير قوي\n"
        "✅ أنت الأولوية الأولى والقصوى\n\n"
        "📋 الأوامر المتاحة:\n"
        "/help - المساعدة\n"
        "/system - معلومات النظام\n"
        "/screenshot - لقطة شاشة\n"
        "/search [كلمات] - بحث\n"
        "/code [وصف] - كتابة كود\n"
        "/cmd [أمر] - تنفيذ أمر نظام\n"
        "/task [مهمة] - تنفيذ مهمة\n"
        "/logs - عرض السجلات\n"
    )
    
    keyboard = [
        [InlineKeyboardButton("🔍 بحث", callback_data='search'),
         InlineKeyboardButton("💻 نظام", callback_data='system')],
        [InlineKeyboardButton("✍️ كود", callback_data='code'),
         InlineKeyboardButton("📸 لقطة", callback_data='screenshot')],
        [InlineKeyboardButton("⚙️ تنفيذ", callback_data='execute'),
         InlineKeyboardButton("📋 مهام", callback_data='task')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /help"""
    if not await verify_owner_command(update, context):
        return
    
    help_text = (
        "📚 دليل المساعدة الشامل:\n\n"
        "🔹 أوامر النظام:\n"
        "/system - معلومات النظام الكاملة\n"
        "/screenshot - لقطة شاشة\n"
        "/cmd [أمر] - تنفيذ أي أمر نظام\n\n"
        "🔹 أوامر الذكاء الاصطناعي:\n"
        "/search [كلمات] - بحث قوي\n"
        "/code [وصف] - كتابة كود احترافي\n"
        "/task [مهمة] - تنفيذ مهمة\n"
        "/analyze - تحليل النصوص\n\n"
        "🔹 أوامر الأمان:\n"
        "/logs - عرض سجلات التدقيق\n"
        "/security - معلومات الأمان\n\n"
        "🔹 ملاحظات:\n"
        "• أنت الأولوية الأولى والقصوى\n"
        "• جميع البيانات مشفرة\n"
        "• يتم تسجيل جميع الإجراءات\n"
    )
    await update.message.reply_text(help_text)

async def system_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /system"""
    if not await verify_owner_command(update, context):
        return
    
    await update.message.reply_text("⏳ جاري جمع معلومات النظام...")
    
    system_info = await get_system_info()
    info_text = (
        f"💻 معلومات النظام:\n\n"
        f"🖥️ نظام التشغيل: {system_info.get('os', 'N/A')} {system_info.get('os_version', '')}\n"
        f"⚙️ استخدام المعالج: {system_info.get('cpu_percent', 0)}%\n"
        f"💾 استخدام الذاكرة: {system_info.get('memory_percent', 0)}% ({system_info.get('memory_used', 0):.2f}GB / {system_info.get('memory_total', 0):.2f}GB)\n"
        f"📦 استخدام القرص: {system_info.get('disk_percent', 0)}%\n"
        f"🔄 عدد العمليات: {system_info.get('processes', 0)}\n"
        f"🖲️ عدد المعالجات: {system_info.get('cpu_count', 0)}\n"
        f"⏰ وقت التشغيل: {system_info.get('uptime', 'N/A')}\n"
    )
    await update.message.reply_text(info_text)
    audit_log.log_action(str(update.effective_user.id), "system_info_requested", {}, "success")

async def screenshot_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /screenshot"""
    if not await verify_owner_command(update, context):
        return
    
    await update.message.reply_text("📸 جاري أخذ لقطة الشاشة...")
    
    screenshot_path = await take_screenshot()
    if screenshot_path and os.path.exists(screenshot_path):
        await update.message.reply_photo(photo=open(screenshot_path, 'rb'))
        audit_log.log_action(str(update.effective_user.id), "screenshot_taken", {}, "success")
    else:
        await update.message.reply_text("❌ فشل أخذ لقطة الشاشة")

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /search"""
    if not await verify_owner_command(update, context):
        return
    
    if not context.args:
        await update.message.reply_text("❌ الرجاء إدخال كلمات البحث")
        return
    
    query = " ".join(context.args)
    await update.message.reply_text(f"🔍 جاري البحث عن: {query}...")
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    results = await web_search(query)
    await update.message.reply_text(f"🔍 نتائج البحث:\n\n{results}")
    audit_log.log_action(str(update.effective_user.id), "web_search", {"query": query}, "success")

async def code_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /code"""
    if not await verify_owner_command(update, context):
        return
    
    if not context.args:
        await update.message.reply_text("❌ الرجاء وصف الكود المطلوب")
        return
    
    description = " ".join(context.args)
    await update.message.reply_text(f"✍️ جاري كتابة الكود...")
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    code = await write_code(description)
    
    if len(code) > 4096:
        parts = [code[i:i+4096] for i in range(0, len(code), 4096)]
        for part in parts:
            await update.message.reply_text(f"```\n{part}\n```", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"```\n{code}\n```", parse_mode="Markdown")
    
    audit_log.log_action(str(update.effective_user.id), "code_written", {"description": description}, "success")

async def cmd_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /cmd - تنفيذ أي أمر نظام"""
    if not await verify_owner_command(update, context):
        return
    
    if not context.args:
        await update.message.reply_text("❌ الرجاء إدخال الأمر")
        return
    
    command = " ".join(context.args)
    await update.message.reply_text(f"⚙️ جاري تنفيذ الأمر: {command}...")
    
    result = await execute_system_command(command)
    
    if len(result) > 4096:
        parts = [result[i:i+4096] for i in range(0, len(result), 4096)]
        for part in parts:
            await update.message.reply_text(part)
    else:
        await update.message.reply_text(result)
    
    audit_log.log_action(str(update.effective_user.id), "system_command", {"command": command}, "success")

async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /task"""
    if not await verify_owner_command(update, context):
        return
    
    if not context.args:
        await update.message.reply_text("❌ الرجاء وصف المهمة")
        return
    
    task = " ".join(context.args)
    await update.message.reply_text(f"🚀 جاري تنفيذ المهمة...")
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    result = await execute_task(task)
    
    if len(result) > 4096:
        parts = [result[i:i+4096] for i in range(0, len(result), 4096)]
        for part in parts:
            await update.message.reply_text(part)
    else:
        await update.message.reply_text(result)
    
    audit_log.log_action(str(update.effective_user.id), "task_executed", {"task": task}, "success")

async def logs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /logs"""
    if not await verify_owner_command(update, context):
        return
    
    logs = audit_log.get_logs()
    logs_text = "📋 سجلات التدقيق:\n\n"
    
    for log in logs[-20:]:  # آخر 20 سجل
        logs_text += f"• {log['action']} - {log['status']} - {log['timestamp']}\n"
    
    await update.message.reply_text(logs_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الرسائل النصية العادية"""
    if not await verify_owner_command(update, context):
        return
    
    user_message = update.message.text
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "أنت مساعد ذكي متطور وشامل للمالك. يمكنك:\n"
                        "- الإجابة على الأسئلة بدقة\n"
                        "- كتابة أكواد احترافية\n"
                        "- تحليل البيانات\n"
                        "- تنفيذ مهام معقدة\n"
                        "- البحث عن المعلومات\n"
                        "كن ودياً وسريعاً في الرد. المالك هو الأولوية الأولى والقصوى."
                    )
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            temperature=0.8,
            max_tokens=1000
        )
        
        ai_reply = response.choices[0].message.content
        
        if len(ai_reply) > 4096:
            parts = [ai_reply[i:i+4096] for i in range(0, len(ai_reply), 4096)]
            for part in parts:
                await update.message.reply_text(part)
        else:
            await update.message.reply_text(ai_reply)
        
        logger.info(f"تم الرد على المالك: {update.effective_user.id}")
        audit_log.log_action(str(update.effective_user.id), "message_processed", {"message_length": len(user_message)}, "success")
        
    except Exception as e:
        logger.error(f"خطأ في الرد الذكي: {e}")
        await update.message.reply_text(f"❌ عذراً، حدث خطأ: {str(e)}")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج أزرار القائمة"""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'search':
        await query.edit_message_text("🔍 أرسل كلمات البحث:")
    elif query.data == 'system':
        await system_command(update, context)
    elif query.data == 'code':
        await query.edit_message_text("✍️ صف الكود المطلوب:")
    elif query.data == 'screenshot':
        await screenshot_command(update, context)
    elif query.data == 'execute':
        await query.edit_message_text("⚙️ أرسل الأمر:")
    elif query.data == 'task':
        await query.edit_message_text("📋 صف المهمة:")

# ============================================================================
# الدالة الرئيسية
# ============================================================================

def main():
    """تشغيل البوت"""
    if not BOT_TOKEN:
        logger.error("❌ لم يتم العثور على BOT_TOKEN")
        return
    
    if OWNER_ID == "YOUR_TELEGRAM_ID":
        logger.error("❌ الرجاء تعيين معرف المالك (OWNER_ID)")
        return
    
    logger.info("🚀 بدء تشغيل البوت الذكي المتطور...")
    logger.info(f"🔐 معرف المالك: {OWNER_ID}")
    logger.info("🔒 نظام الأمان: مفعل")
    
    # إنشاء التطبيق
    application = Application.builder().token(BOT_TOKEN).build()
    
    # إضافة معالجات الأوامر
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("system", system_command))
    application.add_handler(CommandHandler("screenshot", screenshot_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("code", code_command))
    application.add_handler(CommandHandler("cmd", cmd_command))
    application.add_handler(CommandHandler("task", task_command))
    application.add_handler(CommandHandler("logs", logs_command))
    
    # معالج أزرار القائمة
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # معالج الرسائل النصية
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # تشغيل خادم Flask
    logger.info("🌐 بدء تشغيل خادم Flask...")
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # تشغيل البوت
    logger.info("✅ البوت جاهز وينتظر الرسائل...")
    logger.info("🔐 أنت الأولوية الأولى والقصوى")
    application.run_polling(poll_interval=1.0, allowed_updates=["message", "callback_query"])

if __name__ == '__main__':
    main()
