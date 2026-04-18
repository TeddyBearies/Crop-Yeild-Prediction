from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Crop Yield Prediction API")

# Setup templates (assuming run from project root)
templates = Jinja2Templates(directory="app/templates")

# Load model artifacts
try:
    print("Loading model artifacts...")
    model = joblib.load('models/final_pipeline_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_cols = joblib.load('models/feature_columns.joblib')
    print("Artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Pass some context variables for the dropdowns
    states = [col.replace('State_', '') for col in feature_cols if col.startswith('State_')]
    # Add states not explicitly in dummies (since drop_first=True, 'Andhra Pradesh' is typically dropped)
    # Actually, we can just use the states present in the dummies as the main options, but let's provide a clean list.
    # We'll just provide the raw lists we expect
    seasons = ['Kharif     ', 'Rabi       ', 'Summer     ', 'Whole Year ', 'Winter     ']
    zones = ['Eastern', 'North-Eastern', 'Northern', 'Southern', 'Western']
    crops = ['Rice', 'Wheat']
    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "states": sorted(states),
            "seasons": seasons,
            "zones": sorted(zones),
            "crops": crops
        }
    )

@app.post("/predict")
async def predict(
    request: Request,
    crop: str = Form(...),
    season: str = Form(...),
    state: str = Form(...),
    zone: str = Form(...),
    year: int = Form(...),
    area: float = Form(...),
    production: float = Form(...),
    rainfall: float = Form(...),
    fertilizer: float = Form(...),
    pesticide: float = Form(...)
):
    try:
        # 1. Create a dictionary with all 0s for the feature columns
        input_data = {col: 0 for col in feature_cols}
        
        # 2. Add the unscaled numerical data
        input_data['Crop_Year'] = year
        input_data['Area'] = area
        input_data['Production'] = production
        input_data['Annual_Rainfall'] = rainfall
        input_data['Fertilizer'] = fertilizer
        input_data['Pesticide'] = pesticide
        
        # 3. Set the appropriate dummy columns to 1
        if f'Crop_{crop}' in input_data:
            input_data[f'Crop_{crop}'] = 1
            
        if f'Season_{season}' in input_data:
            input_data[f'Season_{season}'] = 1
            
        if f'State_{state}' in input_data:
            input_data[f'State_{state}'] = 1
            
        if f'Zone_{zone}' in input_data:
            input_data[f'Zone_{zone}'] = 1
            
        # Create a single-row DataFrame
        df_input = pd.DataFrame([input_data])
        
        # 4. Scale the numerical columns 
        # (Must match the columns the scaler was fitted on: ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'])
        num_cols = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        df_input[num_cols] = scaler.transform(df_input[num_cols])
        
        # 5. Predict using the pipeline
        prediction = model.predict(df_input)[0]
        
        return {
            "status": "success", 
            "forecast": round(prediction, 4),
            "input_summary": {
                "Crop": crop,
                "State": state,
                "Rainfall": rainfall,
                "Area": area,
                "Production": production
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

    
    
