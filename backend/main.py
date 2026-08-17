import io
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

# Add the root directory to sys.path to allow importing from 'module'
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from module.pipeline import InvoicePipeline

app = FastAPI(title="Invoice AI Extraction API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev. Change in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing AI Pipeline (CUDA:0)...")
# Initialize the pipeline globally
pipeline = InvoicePipeline(device='cuda:0')
print("Pipeline Ready.")

# Serve React frontend build in production (Docker)
frontend_dist = os.path.join(root_dir, "frontend", "dist")
if os.path.isdir(frontend_dist):
    from starlette.staticfiles import StaticFiles
    from starlette.responses import FileResponse
    
    @app.get("/")
    async def serve_spa():
        return FileResponse(os.path.join(frontend_dist, "index.html"))
    
    app.mount("/", StaticFiles(directory=frontend_dist), name="static")

class ExtractionResponse(BaseModel):
    success: bool
    data: list[dict]
    annotated_image_base64: str | None = None
    message: str = ""

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "AI Pipeline is running."}

@app.post("/api/v1/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only PNG, JPG, JPEG are allowed.")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
            
        # Process via pipeline
        print(f"Processing image: {file.filename}")
        results = pipeline.process(img, output_image_path=None)
        
        formatted_data = []
        grouped_data = {}
        
        vis_img = img.copy()
        colors = {
            'B-SELLER': (0, 0, 255), 'I-SELLER': (0, 0, 255),
            'B-ADDRESS': (0, 255, 0), 'I-ADDRESS': (0, 255, 0),
            'B-TIMESTAMP': (255, 0, 0), 'I-TIMESTAMP': (255, 0, 0),
            'B-TOTAL_COST': (0, 255, 255), 'I-TOTAL_COST': (0, 255, 255),
            'B-ITEM': (255, 0, 255), 'I-ITEM': (255, 0, 255),
            'B-QTY': (255, 255, 0), 'I-QTY': (255, 255, 0),
            'B-PRICE': (0, 165, 255), 'I-PRICE': (0, 165, 255),
            'O': (200, 200, 200)
        }
        
        h, w = vis_img.shape[:2]
        
        for item in results.get('layoutlm_labels', []):
            label = item['label']
            if label == 'O': 
                continue
            
            bbox = item['bbox']
            color = colors.get(label, (0, 255, 0))
            # bbox is normalized [0-1000], convert back to pixel coords
            x1 = int(bbox[0] * w / 1000)
            y1 = int(bbox[1] * h / 1000)
            x2 = int(bbox[2] * w / 1000)
            y2 = int(bbox[3] * h / 1000)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, label[2:] if '-' in label else label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            label_name = label[2:] if '-' in label else label
            if label_name not in grouped_data:
                grouped_data[label_name] = []
            grouped_data[label_name].append(item['text'])
            
        for k, v in grouped_data.items():
            formatted_data.append({
                "label": k,
                "text": " ".join(v)
            })
            
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', vis_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return ExtractionResponse(
            success=True,
            data=formatted_data,
            annotated_image_base64=f"data:image/jpeg;base64,{img_base64}",
            message="Extraction successful."
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)



