"""
Database Management Script for AI Document Extraction System
"""

import json
from pathlib import Path
from database_setup import DocumentDatabase

def show_database_stats():
    """Show database statistics"""
    print("📊 Database Statistics")
    print("=" * 40)
    
    db = DocumentDatabase()
    stats = db.get_statistics()
    
    print(f"Total Documents: {stats.get('total_documents', 0)}")
    print(f"Average Confidence: {stats.get('average_confidence', 0)}%")
    print(f"Recent Uploads (24h): {stats.get('recent_uploads', 0)}")
    
    print("\nDocument Types:")
    for doc_type, count in stats.get('document_types', {}).items():
        print(f"  {doc_type}: {count}")

def show_recent_documents(limit=10):
    """Show recent documents"""
    print(f"\n📄 Recent Documents (Last {limit})")
    print("=" * 40)
    
    db = DocumentDatabase()
    documents = db.get_all_documents(limit)
    
    for doc in documents:
        print(f"ID: {doc['id']}")
        print(f"File: {doc['filename']}")
        print(f"Type: {doc['document_type']}")
        print(f"Confidence: {doc['confidence_score']:.2f}")
        print(f"Uploaded: {doc['upload_timestamp']}")
        print("-" * 30)

def search_documents():
    """Search documents by field"""
    print("\n🔍 Search Documents")
    print("=" * 40)
    
    field_name = input("Enter field name (e.g., 'First Name', 'SSN'): ")
    field_value = input("Enter search value: ")
    
    db = DocumentDatabase()
    results = db.search_documents(field_name, field_value)
    
    print(f"\nFound {len(results)} documents:")
    for result in results:
        print(f"ID: {result['id']}")
        print(f"File: {result['filename']}")
        print(f"Type: {result['document_type']}")
        print(f"Value: {result['field_value']}")
        print("-" * 30)

def show_document_details():
    """Show detailed information for a specific document"""
    print("\n📋 Document Details")
    print("=" * 40)
    
    document_id = input("Enter document ID: ")
    
    db = DocumentDatabase()
    document = db.get_document(document_id)
    
    if document:
        print(f"ID: {document['id']}")
        print(f"Filename: {document['filename']}")
        print(f"Document Type: {document['document_type']}")
        print(f"Confidence Score: {document['confidence_score']:.2f}")
        print(f"Upload Time: {document['upload_timestamp']}")
        print(f"Processing Time: {document['processing_time']:.2f}s")
        
        print("\nExtracted Fields:")
        for field, value in document.get('extracted_fields', {}).items():
            print(f"  {field}: {value}")
        
        print("\nFinancial Data:")
        for data_type, value in document.get('financial_data', {}).items():
            print(f"  {data_type}: {value}")
        
        print("\nValidation:")
        validation = document.get('validation', {})
        print(f"  Valid: {validation.get('is_valid', 'Unknown')}")
        if validation.get('warnings'):
            print(f"  Warnings: {validation['warnings']}")
        if validation.get('errors'):
            print(f"  Errors: {validation['errors']}")
    else:
        print("Document not found!")

def export_database():
    """Export database to JSON"""
    print("\n📤 Export Database")
    print("=" * 40)
    
    db = DocumentDatabase()
    documents = db.get_all_documents(1000)  # Get all documents
    
    export_data = {
        "export_timestamp": str(datetime.now()),
        "total_documents": len(documents),
        "documents": []
    }
    
    for doc in documents:
        full_doc = db.get_document(doc['id'])
        if full_doc:
            export_data["documents"].append(full_doc)
    
    export_file = f"database_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(export_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"✅ Database exported to {export_file}")
    print(f"Exported {len(export_data['documents'])} documents")

def migrate_json_files():
    """Migrate existing JSON files to database"""
    print("\n🔄 Migrate JSON Files to Database")
    print("=" * 40)
    
    db = DocumentDatabase()
    results_dir = Path("results")
    
    json_files = list(results_dir.glob("*_results.json"))
    print(f"Found {len(json_files)} JSON files to migrate")
    
    migrated = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            db.save_document(data)
            migrated += 1
            print(f"✅ Migrated {json_file.name}")
            
        except Exception as e:
            print(f"❌ Error migrating {json_file.name}: {e}")
    
    print(f"\n🎉 Migration complete! Migrated {migrated} documents")

def main():
    """Main menu"""
    while True:
        print("\n🤖 AI Document Extraction System - Database Manager")
        print("=" * 60)
        print("1. Show Database Statistics")
        print("2. Show Recent Documents")
        print("3. Search Documents")
        print("4. Show Document Details")
        print("5. Export Database")
        print("6. Migrate JSON Files")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == "1":
            show_database_stats()
        elif choice == "2":
            limit = input("Enter number of documents to show (default 10): ")
            show_recent_documents(int(limit) if limit.isdigit() else 10)
        elif choice == "3":
            search_documents()
        elif choice == "4":
            show_document_details()
        elif choice == "5":
            export_database()
        elif choice == "6":
            migrate_json_files()
        elif choice == "7":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    from datetime import datetime
    main()
