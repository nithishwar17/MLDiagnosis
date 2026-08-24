import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_FILE = "app/healthcare_app.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create History Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patient_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_added TEXT,
        doctor_name TEXT,
        patient_name TEXT,
        age INTEGER,
        gender TEXT,
        symptoms TEXT,
        prediction TEXT,
        confidence REAL,
        severity TEXT,
        feedback TEXT
    )
    ''')
    conn.commit()
    conn.close()

def add_record(doctor, patient, age, gender, symptoms, prediction, confidence, severity):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    date_added = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
    INSERT INTO patient_history (date_added, doctor_name, patient_name, age, gender, symptoms, prediction, confidence, severity, feedback)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (date_added, doctor, patient, age, gender, symptoms, prediction, confidence, severity, "Pending"))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def update_feedback(record_id, feedback):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('UPDATE patient_history SET feedback = ? WHERE id = ?', (feedback, record_id))
    conn.commit()
    conn.close()

def get_all_history():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM patient_history ORDER BY id DESC", conn)
    conn.close()
    return df
