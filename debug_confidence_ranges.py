"""
Debug Confidence Ranges
Check what confidence scores exist in the database
"""

import sqlite3
import json

def debug_confidence_ranges():
    """Debug confidence score ranges in the database"""
    
    print("🔍 Debugging Confidence Score Ranges...")
    
    # Connect to database
    conn = sqlite3.connect("document_extractions.db")
    cursor = conn.cursor()
    
    try:
        # Get all documents with confidence scores
        cursor.execute('''
            SELECT id, filename, document_type, confidence_score 
            FROM documents 
            ORDER BY confidence_score DESC
        ''')
        
        documents = cursor.fetchall()
        
        print(f"📊 Total documents in database: {len(documents)}")
        print("\n📈 Confidence Score Distribution:")
        print("=" * 50)
        
        # Group by confidence ranges
        high_count = 0      # 0.7-1.0
        medium_count = 0    # 0.4-0.69
        low_count = 0       # 0.1-0.39
        very_low_count = 0  # 0.0-0.09
        zero_count = 0      # exactly 0.0
        
        for doc in documents:
            doc_id, filename, doc_type, confidence = doc
            confidence = confidence or 0.0
            
            print(f"  {filename[:30]:<30} | {doc_type:<15} | {confidence:.3f}")
            
            if confidence >= 0.7:
                high_count += 1
            elif confidence >= 0.4:
                medium_count += 1
            elif confidence >= 0.1:
                low_count += 1
            elif confidence > 0.0:
                very_low_count += 1
            else:
                zero_count += 1
        
        print("\n📊 Range Summary:")
        print("=" * 50)
        print(f"High (0.7-1.0):     {high_count:>3} documents")
        print(f"Medium (0.4-0.69):  {medium_count:>3} documents")
        print(f"Low (0.1-0.39):     {low_count:>3} documents")
        print(f"Very Low (0.01-0.09): {very_low_count:>3} documents")
        print(f"Zero (0.0):         {zero_count:>3} documents")
        
        # Check if there are any documents in the low range
        if low_count == 0:
            print("\n⚠️  No documents found in Low (0.1-0.39) range!")
            print("   This explains why clicking 'Low' shows no results.")
            
            # Suggest alternative ranges
            if very_low_count > 0:
                print(f"\n💡 Suggestion: Consider adding a 'Very Low (0.01-0.09)' filter for {very_low_count} documents")
            if zero_count > 0:
                print(f"\n💡 Suggestion: Consider adding a 'Zero (0.0)' filter for {zero_count} documents")
        
        # Get confidence score statistics
        confidences = [doc[3] or 0.0 for doc in documents]
        if confidences:
            print(f"\n📈 Statistics:")
            print(f"   Min confidence: {min(confidences):.3f}")
            print(f"   Max confidence: {max(confidences):.3f}")
            print(f"   Avg confidence: {sum(confidences)/len(confidences):.3f}")
            
            # Show unique confidence values
            unique_confidences = sorted(set(confidences))
            print(f"\n🎯 Unique confidence scores:")
            for conf in unique_confidences[:10]:  # Show first 10
                print(f"   {conf:.3f}")
            if len(unique_confidences) > 10:
                print(f"   ... and {len(unique_confidences) - 10} more")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    debug_confidence_ranges()
