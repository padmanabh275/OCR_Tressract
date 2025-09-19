"""
Database setup for AI Document Extraction System
Creates SQLite database to store extracted fields linked to uploads
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class DocumentDatabase:
    """Database manager for document extraction results"""
    
    def __init__(self, db_path: str = "document_extractions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
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
                confidence_score REAL
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
                amount REAL,
                year INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        # Create validation_results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                is_valid BOOLEAN,
                warnings TEXT,
                errors TEXT,
                field_scores TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_timestamp ON documents(upload_timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fields_document ON extracted_fields(document_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_financial_document ON financial_data(document_id)')
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")
    
    def save_document(self, document_data: Dict) -> str:
        """Save document and extracted fields to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert document record
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (id, filename, original_filename, file_path, file_size, 
                 processing_time, status, document_type, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document_data['id'],
                document_data['filename'],
                document_data.get('original_filename', document_data['filename']),
                document_data.get('file_path', ''),
                document_data.get('file_size', 0),
                document_data.get('processing_time', 0.0),
                document_data.get('status', 'completed'),
                document_data.get('document_type', 'unknown'),
                document_data.get('confidence_score', 0.0)
            ))
            
            # Insert extracted fields
            extracted_data = document_data.get('extracted_data', {})
            field_mapping = {
                'first_name': 'First Name',
                'last_name': 'Last Name',
                'date_of_birth': 'Date of Birth',
                'marriage_date': 'Marriage Date',
                'birth_city': 'Birth City',
                'ssn': 'SSN',
                'current_address': 'Current Address'
            }
            
            for field_key, field_display in field_mapping.items():
                field_value = extracted_data.get(field_key)
                if field_value:
                    cursor.execute('''
                        INSERT INTO extracted_fields 
                        (document_id, field_name, field_value, confidence_score)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        document_data['id'],
                        field_display,
                        str(field_value),
                        document_data.get('confidence_score', 0.0)
                    ))
            
            # Insert financial data
            financial_data = extracted_data.get('financial_data', {})
            if financial_data and isinstance(financial_data, dict):
                for data_type, data_value in financial_data.items():
                    if data_value:
                        cursor.execute('''
                            INSERT INTO financial_data 
                            (document_id, data_type, data_value)
                            VALUES (?, ?, ?)
                        ''', (
                            document_data['id'],
                            data_type,
                            str(data_value)
                        ))
            
            # Insert validation results
            validation = extracted_data.get('validation', {})
            if validation and isinstance(validation, dict):
                cursor.execute('''
                    INSERT INTO validation_results 
                    (document_id, is_valid, warnings, errors, field_scores)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    document_data['id'],
                    validation.get('is_valid', True),
                    json.dumps(validation.get('warnings', [])),
                    json.dumps(validation.get('errors', [])),
                    json.dumps(validation.get('field_scores', {}))
                ))
            
            conn.commit()
            print(f"✅ Document {document_data['id']} saved to database")
            return document_data['id']
            
        except Exception as e:
            print(f"❌ Error saving document: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """Get document and all extracted fields"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get document info
            cursor.execute('SELECT * FROM documents WHERE id = ?', (document_id,))
            doc_row = cursor.fetchone()
            if not doc_row:
                return None
            
            # Get extracted fields
            cursor.execute('''
                SELECT field_name, field_value, confidence_score 
                FROM extracted_fields 
                WHERE document_id = ?
            ''', (document_id,))
            fields = cursor.fetchall()
            
            # Get financial data
            cursor.execute('''
                SELECT data_type, data_value, amount, year 
                FROM financial_data 
                WHERE document_id = ?
            ''', (document_id,))
            financial = cursor.fetchall()
            
            # Get validation results
            cursor.execute('''
                SELECT is_valid, warnings, errors, field_scores 
                FROM validation_results 
                WHERE document_id = ?
            ''', (document_id,))
            validation_row = cursor.fetchone()
            
            # Build result
            result = {
                'id': doc_row[0],
                'filename': doc_row[1],
                'original_filename': doc_row[2],
                'file_path': doc_row[3],
                'file_size': doc_row[4],
                'upload_timestamp': doc_row[5],
                'processing_time': doc_row[6],
                'status': doc_row[7],
                'document_type': doc_row[8],
                'confidence_score': doc_row[9],
                'extracted_fields': {field[0]: field[1] for field in fields},
                'financial_data': {data[0]: data[1] for data in financial},
                'validation': {
                    'is_valid': validation_row[0] if validation_row else True,
                    'warnings': json.loads(validation_row[1]) if validation_row and validation_row[1] else [],
                    'errors': json.loads(validation_row[2]) if validation_row and validation_row[2] else [],
                    'field_scores': json.loads(validation_row[3]) if validation_row and validation_row[3] else {}
                } if validation_row else {}
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error getting document: {e}")
            return None
        finally:
            conn.close()
    
    def get_all_documents(self, limit: int = 100) -> List[Dict]:
        """Get all documents with full extracted data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, filename, document_type, confidence_score, 
                       upload_timestamp, processing_time, status, file_size, file_path
                FROM documents 
                ORDER BY upload_timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            documents = []
            for row in cursor.fetchall():
                doc_id = row[0]
                
                # Get extracted fields for this document
                extracted_data = self._get_extracted_fields_for_document(conn, doc_id)
                
                documents.append({
                    'id': doc_id,
                    'filename': row[1],
                    'document_type': row[2],
                    'confidence_score': row[3],
                    'upload_timestamp': row[4],
                    'processing_time': row[5],
                    'status': row[6],
                    'file_size': row[7] or 0,
                    'file_path': row[8] or '',
                    'extracted_data': extracted_data
                })
            
            return documents
            
        except Exception as e:
            print(f"❌ Error getting documents: {e}")
            return []
        finally:
            conn.close()
    
    def _get_extracted_fields_for_document(self, conn, document_id: str) -> Dict:
        """Get all extracted fields for a specific document"""
        cursor = conn.cursor()
        
        try:
            # Get basic extracted fields
            cursor.execute('''
                SELECT field_name, field_value FROM extracted_fields 
                WHERE document_id = ?
            ''', (document_id,))
            
            extracted_data = {}
            for field_name, field_value in cursor.fetchall():
                # Convert field names back to the expected format
                field_mapping = {
                    'First Name': 'first_name',
                    'Last Name': 'last_name',
                    'Date of Birth': 'date_of_birth',
                    'Marriage Date': 'marriage_date',
                    'Birth City': 'birth_city',
                    'SSN': 'ssn',
                    'Current Address': 'current_address'
                }
                
                if field_name in field_mapping:
                    extracted_data[field_mapping[field_name]] = field_value
            
            # Get financial data
            cursor.execute('''
                SELECT field_name, field_value FROM financial_data 
                WHERE document_id = ?
            ''', (document_id,))
            
            financial_data = {}
            for field_name, field_value in cursor.fetchall():
                financial_data[field_name] = field_value
            
            if financial_data:
                extracted_data['financial_data'] = financial_data
            
            return extracted_data
            
        except Exception as e:
            print(f"❌ Error getting extracted fields: {e}")
            return {}
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its associated data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete from all related tables
            cursor.execute('DELETE FROM extracted_fields WHERE document_id = ?', (document_id,))
            cursor.execute('DELETE FROM financial_data WHERE document_id = ?', (document_id,))
            cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
            return False
        finally:
            conn.close()
    
    def search_documents(self, field_name: str, field_value: str) -> List[Dict]:
        """Search documents by extracted field value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT DISTINCT d.id, d.filename, d.document_type, d.confidence_score, 
                       d.upload_timestamp, ef.field_value
                FROM documents d
                JOIN extracted_fields ef ON d.id = ef.document_id
                WHERE ef.field_name = ? AND ef.field_value LIKE ?
                ORDER BY d.upload_timestamp DESC
            ''', (field_name, f'%{field_value}%'))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'filename': row[1],
                    'document_type': row[2],
                    'confidence_score': row[3],
                    'upload_timestamp': row[4],
                    'field_value': row[5]
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
        finally:
            conn.close()
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Total documents
            cursor.execute('SELECT COUNT(*) FROM documents')
            total_docs = cursor.fetchone()[0]
            
            # Document types
            cursor.execute('''
                SELECT document_type, COUNT(*) 
                FROM documents 
                GROUP BY document_type
            ''')
            doc_types = dict(cursor.fetchall())
            
            # Average confidence
            cursor.execute('SELECT AVG(confidence_score) FROM documents')
            avg_confidence = cursor.fetchone()[0] or 0
            
            # Recent uploads (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM documents 
                WHERE upload_timestamp > datetime('now', '-1 day')
            ''')
            recent_uploads = cursor.fetchone()[0]
            
            return {
                'total_documents': total_docs,
                'document_types': doc_types,
                'average_confidence': round(avg_confidence, 2),
                'recent_uploads': recent_uploads
            }
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {}
        finally:
            conn.close()

def migrate_existing_data():
    """Migrate existing JSON files to database"""
    db = DocumentDatabase()
    results_dir = Path("results")
    
    migrated_count = 0
    for json_file in results_dir.glob("*_results.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            db.save_document(data)
            migrated_count += 1
            print(f"✅ Migrated {json_file.name}")
            
        except Exception as e:
            print(f"❌ Error migrating {json_file.name}: {e}")
    
    print(f"🎉 Migration complete! Migrated {migrated_count} documents to database")

if __name__ == "__main__":
    # Initialize database
    db = DocumentDatabase()
    
    # Migrate existing data
    print("🔄 Migrating existing JSON data to database...")
    migrate_existing_data()
    
    # Show statistics
    stats = db.get_statistics()
    print(f"\n📊 Database Statistics:")
    print(f"Total Documents: {stats.get('total_documents', 0)}")
    print(f"Average Confidence: {stats.get('average_confidence', 0)}%")
    print(f"Document Types: {stats.get('document_types', {})}")
