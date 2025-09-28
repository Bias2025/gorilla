@@
 import os
 import sys
 import argparse
 import asyncio
 import time
 import logging
 from pathlib import Path
 from datetime import datetime
 from typing import List, Dict, Optional
+import json
@@
-class SimpleLakeraEvaluator:
+class SimpleLakeraEvaluator:
     """Simple Lakera evaluator with verbose output"""
     
-    def __init__(self, api_key: str):
+    def __init__(self, api_key: str, project_id: Optional[str] = None):
         self.api_key = api_key
+        self.project_id = project_id  # Lakera “profile” via project/policy mapping
         self.url = 'https://api.lakera.ai/v2/guard/results'
         self.headers = {
             'Authorization': f'Bearer {api_key}',
             'Content-Type': 'application/json'
         }
         self.session = None
         
         logger.info(f"🔧 Initialized evaluator")
         logger.info(f"   API URL: {self.url}")
         logger.info(f"   API Key length: {len(api_key)}")
+        if self.project_id:
+            logger.info(f"   Project (policy profile): {self.project_id}")
+        else:
+            logger.info("   Project (policy profile): <default>")
@@
     async def test_api(self) -> bool:
         """Test API with a simple request"""
         logger.info("🧪 Testing API connection...")
         
         try:
             test_prompt = "Hello, how are you today?"
-            body = {"messages": [{"role": "user", "content": test_prompt}]}
+            body = {
+                "messages": [{"role": "user", "content": test_prompt}],
+                **({"project_id": self.project_id} if self.project_id else {})
+            }
@@
     async def check_single_prompt(self, prompt: str, index: int) -> Dict:
         """Check a single prompt"""
         try:
             start_time = time.time()
-            body = {"messages": [{"role": "user", "content": prompt}]}
+            body = {
+                "messages": [{"role": "user", "content": prompt}],
+                **({"project_id": self.project_id} if self.project_id else {})
+            }
@@
-async def process_dataset_simple(file_path: str, api_key: str, 
+async def process_dataset_simple(file_path: str, api_key: str,
+                               project_id: Optional[str] = None,
                                text_column: Optional[str] = None,
                                label_column: Optional[str] = None,
                                max_rows: int = None):
@@
-    async with SimpleLakeraEvaluator(api_key) as evaluator:
+    async with SimpleLakeraEvaluator(api_key, project_id=project_id) as evaluator:
@@
 def main():
     """Main function with verbose startup"""
     print("🎯 Parsing arguments...")
     
     parser = argparse.ArgumentParser(description='Simple Lakera Guard Evaluator')
     parser.add_argument('--datasets', nargs='+', required=True, help='Dataset files')
     parser.add_argument('--env', required=True, help='API key')
+    parser.add_argument('--project-id', help='Lakera Guard project ID (selects policy/profile). '
+                                             'If omitted, uses LAKERA_PROJECT_ID env var or defaults.')
     parser.add_argument('--text-column', help='Text column name')
     parser.add_argument('--label-column', help='Label column name')
     parser.add_argument('--max-rows', type=int, help='Limit rows for testing')
@@
-    print(f"   API key length: {len(args.env)}")
+    project_id = args.project_id or os.getenv('LAKERA_PROJECT_ID')
+    print(f"   API key length: {len(args.env)}")
+    print(f"   Project ID: {project_id if project_id else '<default>'}")
     if args.max_rows:
         print(f"   Max rows: {args.max_rows}")
@@
-        try:
-            success = asyncio.run(process_dataset_simple(
-                dataset_path, args.env, args.text_column, args.label_column, args.max_rows
-            ))
+        try:
+            success = asyncio.run(process_dataset_simple(
+                dataset_path,
+                api_key=args.env,
+                project_id=project_id,
+                text_column=args.text_column,
+                label_column=args.label_column,
+                max_rows=args.max_rows
+            ))
