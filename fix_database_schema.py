"""
Fix Database Schema
Fix the database schema error 'no such column: field_name'
"""

import sqlite3
import os
from pathlib import Path

def fix_database_schema():
    """Fix the database schema by recreating it with the correct structure"""
    
    db_path = "document_extractions.db"
    
    print("🔧 Fixing database schema...")
    
    # Check if database exists
    if not os.path.exists(db_path):
        print("❌ Database file not found. Creating new database...")
        return
    
    # Backup the old database
    backup_path = f"{db_path}.backup"
    if os.path.exists(backup_path):
        os.remove(backup_path)
    
    print(f"📦 Creating backup: {backup_path}")
    os.rename(db_path, backup_path)
    
    # Create new database with correct schema
    print("🆕 Creating new database with correct schema...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            original_filename TEXT,
            file_path TEXT,
            file_size INTEGER,
            upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            processing_time REAL,
            status TEXT DEFAULT 'completed',
            document_type TEXT,
            confidence_score REAL,
            document_category TEXT,
            processing_method TEXT,
            accuracy_mode TEXT,
            accuracy_boost REAL
        )
    ''')
    
    # Create extracted_fields table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extracted_fields (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            field_name TEXT NOT NULL,
            field_value TEXT,
            confidence_score REAL,
            validation_status TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    ''')
    
    # Create financial_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            data_type TEXT NOT NULL,
            data_value TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    ''')
    
    # Create validation_results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS validation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            is_valid BOOLEAN DEFAULT FALSE,
            warnings TEXT,
            errors TEXT,
            field_scores TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    ''')
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_timestamp ON documents (upload_timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_type ON documents (document_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_extracted_fields_doc_id ON extracted_fields (document_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_financial_data_doc_id ON financial_data (document_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_validation_results_doc_id ON validation_results (document_id)')
    
    conn.commit()
    conn.close()
    
    print("✅ Database schema fixed successfully!")
    print(f"📦 Old database backed up as: {backup_path}")
    print("🆕 New database created with correct schema")
    
    # Try to migrate data from backup if possible
    try:
        migrate_data_from_backup(backup_path, db_path)
    except Exception as e:
        print(f"⚠️ Could not migrate data from backup: {e}")
        print("   You may need to re-upload documents to populate the new database")

def migrate_data_from_backup(backup_path: str, new_db_path: str):
    """Migrate data from backup database to new database"""
    
    print("🔄 Attempting to migrate data from backup...")
    
    # Connect to both databases
    backup_conn = sqlite3.connect(backup_path)
    new_conn = sqlite3.connect(new_db_path)
    
    backup_cursor = backup_conn.cursor()
    new_cursor = new_conn.cursor()
    
    try:
        # Get all documents from backup
        backup_cursor.execute('SELECT * FROM documents')
        documents = backup_cursor.fetchall()
        
        print(f"📄 Found {len(documents)} documents in backup")
        
        # Get column names from backup
        backup_cursor.execute('PRAGMA table_info(documents)')
        backup_columns = [row[1] for row in backup_cursor.fetchall()]
        
        # Insert documents into new database
        for doc in documents:
            doc_dict = dict(zip(backup_columns, doc))
            
            # Insert with only the columns that exist in new schema
            new_cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (id, filename, original_filename, file_path, file_size, 
                 upload_timestamp, processing_time, status, document_type, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_dict.get('id'),
                doc_dict.get('filename'),
                doc_dict.get('original_filename'),
                doc_dict.get('file_path'),
                doc_dict.get('file_size'),
                doc_dict.get('upload_timestamp'),
                doc_dict.get('processing_time'),
                doc_dict.get('status'),
                doc_dict.get('document_type'),
                doc_dict.get('confidence_score')
            ))
        
        new_conn.commit()
        print(f"✅ Migrated {len(documents)} documents successfully")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        new_conn.rollback()
    finally:
        backup_conn.close()
        new_conn.close()

if __name__ == "__main__":
    fix_database_schema()
