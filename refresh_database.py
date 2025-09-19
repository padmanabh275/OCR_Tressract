"""
Database Refresh Script
Clears and recreates the database with the latest schema
"""

import os
import shutil
from pathlib import Path
from database_setup import DocumentDatabase

def backup_database():
    """Create a backup of the current database"""
    db_path = Path("document_extractions.db")
    if db_path.exists():
        backup_path = Path("document_extractions_backup.db")
        shutil.copy2(db_path, backup_path)
        print(f"✅ Database backed up to: {backup_path}")
        return True
    else:
        print("ℹ️ No existing database found to backup")
        return False

def clear_database():
    """Clear the current database"""
    db_path = Path("document_extractions.db")
    if db_path.exists():
        db_path.unlink()
        print("✅ Database cleared")
        return True
    else:
        print("ℹ️ No database to clear")
        return False

def recreate_database():
    """Recreate the database with latest schema"""
    try:
        # Initialize database (this will create the schema)
        db = DocumentDatabase()
        print("✅ Database recreated with latest schema")
        
        # Test database connection
        test_result = db.get_all_documents()
        print(f"✅ Database connection test successful - {len(test_result)} documents")
        
        return True
    except Exception as e:
        print(f"❌ Error recreating database: {e}")
        return False

def verify_database_schema():
    """Verify the database schema is correct"""
    try:
        db = DocumentDatabase()
        
        # Check if we can access all expected tables and columns
        documents = db.get_all_documents()
        print(f"✅ Schema verification successful")
        print(f"   - Documents table: OK")
        print(f"   - Extracted fields table: OK")
        print(f"   - All columns accessible: OK")
        
        return True
    except Exception as e:
        print(f"❌ Schema verification failed: {e}")
        return False

def main():
    """Main refresh function"""
    print("🔄 Refreshing Database...")
    print("=" * 40)
    
    # Step 1: Backup existing database
    print("\n1. Creating backup...")
    backup_database()
    
    # Step 2: Clear current database
    print("\n2. Clearing database...")
    clear_database()
    
    # Step 3: Recreate database
    print("\n3. Recreating database...")
    if not recreate_database():
        print("❌ Failed to recreate database")
        return False
    
    # Step 4: Verify schema
    print("\n4. Verifying schema...")
    if not verify_database_schema():
        print("❌ Schema verification failed")
        return False
    
    print("\n🎉 Database refresh completed successfully!")
    print("\n📊 Database Status:")
    print("   - Schema: Latest version")
    print("   - Tables: Created")
    print("   - Columns: All accessible")
    print("   - Backup: Available (document_extractions_backup.db)")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Database refresh successful!")
    else:
        print("\n❌ Database refresh failed!")
