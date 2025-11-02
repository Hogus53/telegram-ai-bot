# استخدام صورة Python الرسمية
FROM python:3.11-slim

# تعيين مجلد العمل
WORKDIR /app

# نسخ ملف المتطلبات
COPY requirements.txt .

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

# نسخ جميع الملفات
COPY . .

# تعيين متغيرات البيئة
ENV PYTHONUNBUFFERED=1

# تشغيل البوت
CMD ["python3", "main.py"]
