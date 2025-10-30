#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🤖 بوت التلغرام الذكي المتطور - Advanced Telegram AI Bot
مساعدك الشخصي الكامل مع التحكم الكامل بالجهاز
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
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# إعداد عميل OpenAI
client = OpenAI()
MODEL_NAME = "gemini-2.5-flash"

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
        "features": [
            "AI Chat", "Web Search", "Image Generation",
            "System Control", "Code Writing", "Task Automation",
            "File Management", "Data Analysis"
        ],
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

def run_flask():
    app.run(host='0.0.0.0', port=8080, debug=False)

# ============================================================================
# دوال التحكم بالنظام
# ============================================================================

async def execute_system_command(command: str) -> str:
    """تنفيذ أوامر النظام"""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        return result.stdout or result.stderr or "تم التنفيذ بنجاح"
    except Exception as e:
        logger.error(f"خطأ في تنفيذ الأمر: {e}")
        return f"خطأ: {str(e)}"

async def get_system_info() -> Dict:
    """الحصول على معلومات النظام"""
    try:
        return {
            "os": platform.system(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "uptime": psutil.boot_time(),
            "processes": len(psutil.pids())
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
        return screenshot_path
    except Exception as e:
        logger.error(f"خطأ في أخذ لقطة الشاشة: {e}")
        return None

async def write_code(description: str) -> str:
    """كتابة أكواد احترافية"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "أنت مبرمج احترافي متخصص. اكتب كوداً احترافياً عالي الجودة بناءً على الطلب."
                },
                {
                    "role": "user",
                    "content": f"اكتب كوداً احترافياً لـ: {description}"
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"خطأ في كتابة الكود: {e}")
        return f"خطأ: {str(e)}"

async def generate_image(prompt: str) -> Optional[str]:
    """إنشاء صور احترافية"""
    try:
        # استخدام DALL-E أو Stable Diffusion
        # هنا مثال بسيط - يمكن تحسينه
        logger.info(f"جاري إنشاء صورة: {prompt}")
        return "تم إنشاء الصورة بنجاح (يتطلب API مدفوع)"
    except Exception as e:
        logger.error(f"خطأ في إنشاء الصورة: {e}")
        return None

async def web_search(query: str, num_results: int = 5) -> str:
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
                    
                    return "\n".join(results) if results else "لم يتم العثور على نتائج"
        return "خطأ في البحث"
    except Exception as e:
        logger.error(f"خطأ في البحث: {e}")
        return f"خطأ: {str(e)}"

async def analyze_image(image_path: str) -> str:
    """تحليل الصور"""
    try:
        # قراءة الصورة
        image = Image.open(image_path)
        
        # يمكن إضافة تحليل متقدم هنا
        return f"تم تحليل الصورة: {image.size}"
    except Exception as e:
        logger.error(f"خطأ في تحليل الصورة: {e}")
        return f"خطأ: {str(e)}"

async def execute_task(task_description: str) -> str:
    """تنفيذ مهام معقدة"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "أنت مساعد ذكي يمكنه تنفيذ مهام معقدة. قم بتنفيذ المهمة بدقة."
                },
                {
                    "role": "user",
                    "content": f"المهمة: {task_description}"
                }
            ],
            temperature=0.5,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"خطأ في تنفيذ المهمة: {e}")
        return f"خطأ: {str(e)}"

# ============================================================================
# معالجات أوامر التلغرام
# ============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /start"""
    welcome_message = (
        "🤖 مرحباً بك في بوتك الذكي المتطور!\n\n"
        "أنا مساعدك الشخصي الكامل مع:\n"
        "✅ الذكاء الاصطناعي المتقدم\n"
        "✅ التحكم الكامل بالجهاز\n"
        "✅ كتابة أكواد احترافية\n"
        "✅ إنشاء صور احترافية\n"
        "✅ محرك بحث قوي\n"
        "✅ تنفيذ مهام معقدة\n"
        "✅ تحليل البيانات\n\n"
        "📋 الأوامر المتاحة:\n"
        "/help - المساعدة\n"
        "/system - معلومات النظام\n"
        "/screenshot - لقطة شاشة\n"
        "/search [كلمات] - بحث\n"
        "/code [وصف] - كتابة كود\n"
        "/execute [أمر] - تنفيذ أمر\n"
        "/task [مهمة] - تنفيذ مهمة\n"
    )
    
    keyboard = [
        [InlineKeyboardButton("🔍 بحث", callback_data='search'),
         InlineKeyboardButton("💻 نظام", callback_data='system')],
        [InlineKeyboardButton("✍️ كود", callback_data='code'),
         InlineKeyboardButton("🖼️ صورة", callback_data='image')],
        [InlineKeyboardButton("⚙️ تنفيذ", callback_data='execute'),
         InlineKeyboardButton("📸 لقطة", callback_data='screenshot')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /help"""
    help_text = (
        "📚 دليل المساعدة الشامل:\n\n"
        "🔹 أوامر النظام:\n"
        "/system - معلومات النظام\n"
        "/screenshot - لقطة شاشة\n"
        "/cmd [أمر] - تنفيذ أمر نظام\n\n"
        "🔹 أوامر الذكاء الاصطناعي:\n"
        "/search [كلمات] - بحث قوي\n"
        "/code [وصف] - كتابة كود احترافي\n"
        "/image [وصف] - إنشاء صورة\n"
        "/analyze - تحليل النصوص\n"
        "/task [مهمة] - تنفيذ مهمة\n\n"
        "🔹 أوامر إضافية:\n"
        "/info - معلومات البوت\n"
        "/help - هذه الرسالة\n"
    )
    await update.message.reply_text(help_text)

async def system_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /system"""
    await update.message.reply_text("⏳ جاري جمع معلومات النظام...")
    
    system_info = await get_system_info()
    info_text = (
        f"💻 معلومات النظام:\n\n"
        f"🖥️ نظام التشغيل: {system_info.get('os', 'N/A')}\n"
        f"⚙️ استخدام المعالج: {system_info.get('cpu_percent', 0)}%\n"
        f"💾 استخدام الذاكرة: {system_info.get('memory_percent', 0)}%\n"
        f"📦 استخدام القرص: {system_info.get('disk_percent', 0)}%\n"
        f"🔄 عدد العمليات: {system_info.get('processes', 0)}\n"
    )
    await update.message.reply_text(info_text)

async def screenshot_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /screenshot"""
    await update.message.reply_text("📸 جاري أخذ لقطة الشاشة...")
    
    screenshot_path = await take_screenshot()
    if screenshot_path and os.path.exists(screenshot_path):
        await update.message.reply_photo(photo=open(screenshot_path, 'rb'))
    else:
        await update.message.reply_text("❌ فشل أخذ لقطة الشاشة")

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /search"""
    if not context.args:
        await update.message.reply_text("❌ الرجاء إدخال كلمات البحث")
        return
    
    query = " ".join(context.args)
    await update.message.reply_text(f"🔍 جاري البحث عن: {query}...")
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    results = await web_search(query)
    await update.message.reply_text(f"🔍 نتائج البحث:\n\n{results}")

async def code_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /code"""
    if not context.args:
        await update.message.reply_text("❌ الرجاء وصف الكود المطلوب")
        return
    
    description = " ".join(context.args)
    await update.message.reply_text(f"✍️ جاري كتابة الكود...")
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    code = await write_code(description)
    
    # تقسيم الرد إذا كان طويلاً
    if len(code) > 4096:
        parts = [code[i:i+4096] for i in range(0, len(code), 4096)]
        for part in parts:
            await update.message.reply_text(f"```\n{part}\n```", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"```\n{code}\n```", parse_mode="Markdown")

async def execute_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /execute"""
    if not context.args:
        await update.message.reply_text("❌ الرجاء إدخال الأمر")
        return
    
    command = " ".join(context.args)
    await update.message.reply_text(f"⚙️ جاري تنفيذ الأمر...")
    
    result = await execute_system_command(command)
    
    if len(result) > 4096:
        parts = [result[i:i+4096] for i in range(0, len(result), 4096)]
        for part in parts:
            await update.message.reply_text(part)
    else:
        await update.message.reply_text(result)

async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الأمر /task"""
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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج الرسائل النصية العادية"""
    user_message = update.message.text
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "أنت مساعد ذكي متطور وشامل. يمكنك:\n"
                        "- الإجابة على الأسئلة بدقة\n"
                        "- كتابة أكواد احترافية\n"
                        "- تحليل البيانات\n"
                        "- تنفيذ مهام معقدة\n"
                        "- البحث عن المعلومات\n"
                        "كن ودياً وسريعاً في الرد."
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
        
        logger.info(f"تم الرد على المستخدم: {update.effective_user.id}")
        
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
    elif query.data == 'image':
        await query.edit_message_text("🖼️ صف الصورة المطلوبة:")
    elif query.data == 'execute':
        await query.edit_message_text("⚙️ أرسل الأمر:")
    elif query.data == 'screenshot':
        await screenshot_command(update, context)

# ============================================================================
# الدالة الرئيسية
# ============================================================================

def main():
    """تشغيل البوت"""
    if not BOT_TOKEN:
        logger.error("❌ لم يتم العثور على BOT_TOKEN")
        return
    
    logger.info("🚀 بدء تشغيل البوت الذكي المتطور...")
    
    # إنشاء التطبيق
    application = Application.builder().token(BOT_TOKEN).build()
    
    # إضافة معالجات الأوامر
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("system", system_command))
    application.add_handler(CommandHandler("screenshot", screenshot_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("code", code_command))
    application.add_handler(CommandHandler("execute", execute_command_handler))
    application.add_handler(CommandHandler("task", task_command))
    
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
    application.run_polling(poll_interval=1.0, allowed_updates=["message", "callback_query"])

if __name__ == '__main__':
    main()
