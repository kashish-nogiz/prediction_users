from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timezone, timedelta
import joblib
import pandas as pd
import math
import json
import os
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
import redis
import uvicorn

week = {
    'monday': 0,
    'tuesday': 1,
    'wednesday': 2,
    'thursday': 3,
    'friday': 4,
    'saturday': 5,
    'sunday': 6
}

# Initialize FastAPI app
app = FastAPI(
    title="Ride Prediction System API",
    description="API for predicting user demand at various locations",
    version="1.0.0"
)

# Global variables for models and connections
model_path = None
scaler = None
redis_client = None


# Pydantic models
class PredictionRequest(BaseModel):
    drop_longitude: float = Field(..., description="Drop point longitude")
    drop_latitude: float = Field(..., description="Drop point latitude")
    total_users_threshold_percent: float = Field(..., description="Threshold percentage for filtering results")
    hour_prm: float = Field(..., description="Hour send by user")
    minute_prm: float = Field(..., description="Minute send by user")
    max_radius_km: float = Field(2.5, gt=0, le=10, description="Maximum radius in km")
    grid_spacing_km: float = Field(1.0, gt=0, le=5, description="Grid spacing in km")
    polygon_csv_path: Optional[str] = Field(None, description="Path to polygon zones CSV file")


class SimplePredictionRequest(BaseModel):
    drop_longitude: float = Field(..., description="Drop point longitude")
    drop_latitude: float = Field(..., description="Drop point latitude")
    threshold: float = Field(10, ge=0, le=100, description="Threshold percentage for filtering results")


class PredictionPoint(BaseModel):
    latitude: float
    longitude: float
    distance_km: float
    predicted_users: int
    prediction_radius_m: int = 500
    prediction_probability: float = Field(..., description="Confidence probability of the prediction (0-1)")
    polygon_name: Optional[str] = None


class PredictionResponse(BaseModel):
    success: bool
    message: str
    total_points: int
    points_above_threshold: int
    total_predicted_users: float
    average_users_per_point: float
    threshold_value: float
    predictions: List[PredictionPoint]
    processing_time_seconds: float


class SystemStatusResponse(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool
    total_bookings: int


class BookingCoordinate(BaseModel):
    latitude: float
    longitude: float
    driver_id: str


# Startup event to initialize models and connections
@app.on_event("startup")
async def startup_event():
    global model_path, scaler, redis_client

    try:
        # Load ML models
        if os.path.exists('week_model.pkl') and os.path.exists(
                'week_scaler.pkl'):  # right now working with the simple model and scaler
            model_path = joblib.load('week_model.pkl')
            scaler = joblib.load('week_scaler.pkl')
            print("✅ ML models loaded successfully")
        else:
            print("⚠️  ML model files not found. Using dummy predictions.")

        # Initialize Redis (optional)
        try:
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'redis-16648.c278.us-east-1-4.ec2.redns.redis-cloud.com'),
                port=int(os.getenv('REDIS_PORT', '16648')),
                decode_responses=True,
                username=os.getenv('REDIS_USERNAME', 'default'),
                password=os.getenv('REDIS_PASSWORD', '3gyrWhHJoKrzUkQRxGYNZTBQcprp7VmG'),
            )
            redis_client.ping()  # Test connection
            print("✅ Redis connected successfully")
        except Exception as e:
            print(f"⚠️  Redis connection failed: {e}")
            redis_client = None

    except Exception as e:
        print(f"❌ Startup error: {e}")


def convert_to_utc(hour_prm=None, minute_prm=None):
    # Get current UTC time
    utc_now = datetime.now(timezone.utc)

    if hour_prm is not None and minute_prm is not None:
        # Assume user entered hour/minute is LOCAL, so we shift relative to UTC
        user_time = utc_now.replace(hour=hour_prm, minute=minute_prm, second=0, microsecond=0)
        utc_time = user_time.astimezone(timezone.utc)
    else:
        # If no input, just take current UTC
        utc_time = utc_now

    # Extract separately
    hour = utc_time.hour
    minute = utc_time.minute

    return hour, minute


def predict_at_droppoint(drop_longitude: float, drop_latitude: float, hour_prm: int, minute_prm: int) -> tuple:
    """
    Prediction function - predicts number of users within 500m radius of the given point
    Returns: (prediction_value, confidence_probability)
    """
    if model_path is None or scaler is None:
        # Return dummy prediction if models not loaded
        prediction = abs(hash(f"{drop_longitude}{drop_latitude}")) % 20
        probability = 0.75 + (abs(hash(f"{drop_longitude}{drop_latitude}")) % 25) / 100  # 0.75-1.0
        return prediction, probability

    try:
        utcTime = datetime.utcnow()
        time = str(utcTime.time())
        time = time.split(sep=':')
        if hour_prm is not None and minute_prm is not None:
            hour, minute = convert_to_utc(hour_prm, minute_prm)

            print("hour send by the user:", hour)
            print("Minute send by the user:", minute)
        else:
            hour = int(time[0])
            minute = int(time[1])
        second = int(float(time[2]))

        day = utcTime.weekday()

        # Formation Input
        input_data = [[drop_longitude, drop_latitude, hour, minute, second]]

        # Scaled Input
        scaled_input = scaler.transform(input_data)

        # Make Prediction
        prediction = model_path.predict(scaled_input)
        prediction_value = max(0, prediction[0])  # Ensure non-negative

        # Calculate prediction probability based on prediction confidence
        try:
            # If model supports predict_proba, use it
            if hasattr(model_path, 'predict_proba'):
                proba = model_path.predict_proba(scaled_input)
                probability = max(proba[0]) if len(proba[0]) > 0 else 0.8
            else:
                # Calculate probability based on prediction value and some heuristics
                normalized_prediction = min(prediction_value / 50, 1.0)  # Normalize to 0-1
                probability = 0.6 + (normalized_prediction * 0.35)  # Range 0.6-0.95
        except:
            # Fallback probability calculation
            probability = 0.75 + (prediction_value % 25) / 100

        return prediction_value, min(probability, 1.0)  # Ensure probability <= 1.0

    except Exception as e:
        print(f"Prediction error: {e}")
        # Return fallback prediction
        prediction = abs(hash(f"{drop_longitude}{drop_latitude}")) % 20
        probability = 0.75
        return prediction, probability


def load_polygon_zones(csv_file_path: str) -> List[Dict]:
    """Load polygon zones from CSV file and create Shapely Polygon objects"""
    try:
        if not os.path.exists(csv_file_path):
            print(f"Polygon file not found: {csv_file_path}")
            return []

        polygon_df = pd.read_csv(csv_file_path)
        print(f"CSV columns: {list(polygon_df.columns)}")
        print(f"CSV shape: {polygon_df.shape}")

        polygons = []

        for index, row in polygon_df.iterrows():
            try:
                coordinates = []

                # Get all coordinate columns that match the pattern
                coord_cols = [col for col in polygon_df.columns if 'geometry.coordinates[' in col]
                coord_cols.sort()  # Sort to maintain order

                print(f"Row {index} - Found coordinate columns: {coord_cols}")

                # Group coordinates by their array index [0], [1], [2], etc.
                coord_groups = {}
                for col in coord_cols:
                    # Extract the array indices from column name like 'geometry.coordinates[0][0]'
                    try:
                        # Find the pattern [X][Y] in the column name
                        import re
                        matches = re.findall(r'\[(\d+)\]\[(\d+)\]', col)
                        if matches:
                            outer_idx, inner_idx = int(matches[0][0]), int(matches[0][1])
                            if outer_idx not in coord_groups:
                                coord_groups[outer_idx] = {}
                            coord_groups[outer_idx][inner_idx] = row[col]
                    except Exception as e:
                        print(f"Error parsing column {col}: {e}")
                        continue

                # Convert grouped coordinates to list of (lon, lat) tuples
                for outer_idx in sorted(coord_groups.keys()):
                    coord_pair = coord_groups[outer_idx]
                    if 0 in coord_pair and 1 in coord_pair:  # lon and lat
                        lon = coord_pair[0]
                        lat = coord_pair[1]

                        if pd.notna(lon) and pd.notna(lat):
                            coordinates.append((float(lon), float(lat)))
                            print(f"Added coordinate: ({lon}, {lat})")

                print(f"Row {index} - Total coordinates found: {len(coordinates)}")

                if len(coordinates) >= 3:
                    # Close the polygon if not already closed
                    if len(coordinates) > 0 and coordinates[0] != coordinates[-1]:
                        coordinates.append(coordinates[0])

                    try:
                        polygon = Polygon(coordinates)

                        # Only add valid polygons
                        if polygon.is_valid:
                            polygon_name = str(row['name']) if 'name' in row and pd.notna(
                                row['name']) else f'Zone_{index}'
                            polygons.append({
                                'name': polygon_name,
                                'polygon': polygon,
                                'index': index
                            })
                            print(f"✅ Successfully loaded polygon: {polygon_name} with {len(coordinates)} coordinates")
                        else:
                            print(f"❌ Invalid polygon for row {index}")
                    except Exception as e:
                        print(f"❌ Error creating polygon for row {index}: {e}")
                else:
                    print(f"❌ Not enough coordinates for polygon at row {index}: {len(coordinates)} coordinates")

            except Exception as e:
                print(f"❌ Error processing polygon row {index}: {e}")
                continue

        print(f"Successfully loaded {len(polygons)} valid polygons out of {len(polygon_df)} rows")

        # Print summary of loaded polygons
        for p in polygons:
            print(f"Polygon loaded: {p['name']}")

        return polygons

    except Exception as e:
        print(f"❌ Error loading polygon zones: {e}")
        return []


def point_in_any_polygon(lat: float, lon: float, polygons: List[Dict]) -> tuple:
    """Check if a point is within any of the provided polygons"""
    try:
        point = Point(float(lon), float(lat))
        print(f"Checking point ({lat}, {lon}) against {len(polygons)} polygons")

        for polygon_info in polygons:
            try:
                polygon = polygon_info['polygon']
                polygon_name = polygon_info['name']

                # Check if point is contained in polygon
                if polygon.contains(point):
                    print(f"✅ Point ({lat}, {lon}) is inside polygon: {polygon_name}")
                    return True, polygon_name
                else:
                    # Also check if point is on the boundary
                    if polygon.touches(point):
                        print(f"✅ Point ({lat}, {lon}) is on boundary of polygon: {polygon_name}")
                        return True, polygon_name

            except Exception as e:
                print(f"❌ Error checking point in polygon {polygon_info['name']}: {e}")
                continue

        print(f"❌ Point ({lat}, {lon}) is not inside any polygon")
        return False, None

    except Exception as e:
        print(f"❌ Error in point_in_any_polygon: {e}")
        return False, None


def generate_systematic_grid_points(center_lat: float, center_lon: float,
                                    max_radius_km: float = 2.5, grid_spacing_km: float = 1.0) -> List[Dict]:
    """Generate systematic grid points at specified intervals within radius"""
    points = []

    # Convert spacing to approximate degree offsets
    lat_spacing = grid_spacing_km / 111.0
    lon_spacing = grid_spacing_km / (111.0 * math.cos(math.radians(center_lat)))

    # Calculate steps needed
    max_steps = int(max_radius_km / grid_spacing_km) + 1

    # Generate grid points
    for i in range(-max_steps, max_steps + 1):
        for j in range(-max_steps, max_steps + 1):
            lat = center_lat + (i * lat_spacing)
            lon = center_lon + (j * lon_spacing)

            # Check if point is within radius
            distance = geodesic((center_lat, center_lon), (lat, lon)).kilometers

            if distance <= max_radius_km:
                points.append({
                    'latitude': lat,
                    'longitude': lon,
                    'distance_km': distance
                })

    # Sort by distance from center
    points.sort(key=lambda x: x['distance_km'])
    return points


def enhanced_systematic_prediction_system(drop_longitude: float, drop_latitude: float,
                                          polygon_csv_path: Optional[str] = None,
                                          total_users_threshold_percent: float = 35,
                                          hour_prm: int = None, minute_prm: int = None,
                                          max_radius_km: float = 2.5,
                                          grid_spacing_km: float = 1.0) -> Dict:
    """Enhanced systematic prediction system with 1km grid spacing"""

    start_time = datetime.now()
    total_users_active_now = 30
    center_lat = float(drop_latitude)
    center_lon = float(drop_longitude)

    # Load polygon zones if provided
    polygons = []
    if polygon_csv_path:
        polygons = load_polygon_zones(polygon_csv_path)
        print(f"Loaded {len(polygons)} polygons for filtering")

    # Generate systematic grid points
    grid_points = generate_systematic_grid_points(center_lat, center_lon, max_radius_km, grid_spacing_km)
    print(f"Generated {len(grid_points)} grid points")

    # Make predictions and filter by polygons if provided
    all_points = []
    points_outside_polygon = 0

    for point in grid_points:
        lat = point['latitude']
        lon = point['longitude']

        # Check polygon constraint if polygons are provided
        poly_name = None
        if polygons:
            is_valid, poly_name = point_in_any_polygon(lat, lon, polygons)
            if not is_valid:
                points_outside_polygon += 1
                continue

        # Make prediction (now returns prediction and probability)
        prediction, probability = predict_at_droppoint(lon, lat, hour_prm, minute_prm)

        point_data = {
            'latitude': lat,
            'longitude': lon,
            'distance_km': point['distance_km'],
            'predicted_users': int(prediction),
            'prediction_radius_m': 500,
            'prediction_probability': round(probability, 3)
        }

        # Add polygon name if found
        if poly_name:
            point_data['polygon_name'] = poly_name

        all_points.append(point_data)

    print(f"Points outside polygons: {points_outside_polygon}")
    print(f"Valid points for prediction: {len(all_points)}")

    if not all_points:
        return {
            'success': False,
            'message': 'No valid prediction points found',
            'predictions': [],
            'processing_time_seconds': (datetime.now() - start_time).total_seconds()
        }

    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_points)

    print(f"Threshold percentage: {total_users_threshold_percent}")

    # Calculate statistics and apply threshold
    total_predicted_users = df['predicted_users'].sum()
    threshold_value = (total_users_threshold_percent / 100) * total_users_active_now

    print(f"Threshold value: {threshold_value}")

    # Filter points above threshold
    filtered_df = df[df['predicted_users'] >= threshold_value].copy()

    print(f"Points above threshold: {len(filtered_df)}")

    processing_time = (datetime.now() - start_time).total_seconds()

    # Return only points above threshold in the main predictions field
    return {
        'success': True,
        'message': 'Predictions generated successfully',
        'total_points': len(all_points),
        'points_above_threshold': len(filtered_df),
        'total_predicted_users': float(total_predicted_users),
        'average_users_per_point': float(df['predicted_users'].mean()),
        'threshold_value': threshold_value,
        'predictions': filtered_df.to_dict('records'),  # Only points >= threshold
        'processing_time_seconds': processing_time
    }


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Ride Prediction System API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/status"
    }


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status and health check"""
    total_bookings = 0

    # Count Redis bookings if available
    if redis_client:
        try:
            keys = redis_client.keys("booking-*")
            total_bookings = len(keys)
        except Exception:
            pass

    return SystemStatusResponse(
        status="healthy",
        model_loaded=model_path is not None and scaler is not None,
        redis_connected=redis_client is not None,
        total_bookings=total_bookings
    )


@app.post("/predict/simple", response_model=PredictionResponse)
async def make_simple_prediction(request: SimplePredictionRequest):
    """
    Simple prediction endpoint that takes longitude, latitude, and threshold.
    Returns predictions above the specified threshold with probability scores.
    """
    try:
        result = enhanced_systematic_prediction_system(
            drop_longitude=request.drop_longitude,
            drop_latitude=request.drop_latitude,
            polygon_csv_path='latestZones (1).csv',  # Default polygon file
            total_users_threshold_percent=request.threshold,
            max_radius_km=2.5,  # Default 2.5km radius
            grid_spacing_km=1.0  # Default 1km grid spacing
        )

        if not result['success']:
            raise HTTPException(status_code=400, detail=result['message'])

        # Convert predictions to Pydantic models
        predictions = [PredictionPoint(**pred) for pred in result['predictions']]

        return PredictionResponse(
            success=result['success'],
            message=result['message'],
            total_points=result['total_points'],
            points_above_threshold=result['points_above_threshold'],
            total_predicted_users=result['total_predicted_users'],
            average_users_per_point=result['average_users_per_point'],
            threshold_value=result['threshold_value'],
            predictions=predictions,
            processing_time_seconds=result['processing_time_seconds']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    print(request)
    """
    Make predictions for user demand in a grid around the specified drop point.
    Returns predictions that are above the specified threshold.
    """
    try:
        result = enhanced_systematic_prediction_system(
            drop_longitude=request.drop_longitude,
            drop_latitude=request.drop_latitude,
            polygon_csv_path=request.polygon_csv_path,
            total_users_threshold_percent=request.total_users_threshold_percent,
            hour_prm=request.hour_prm,
            minute_prm=request.minute_prm,
            max_radius_km=request.max_radius_km,
            grid_spacing_km=request.grid_spacing_km
        )

        if not result['success']:
            raise HTTPException(status_code=400, detail=result['message'])

        # Convert predictions to Pydantic models
        predictions = [PredictionPoint(**pred) for pred in result['predictions']]

        return PredictionResponse(
            success=result['success'],
            message=result['message'],
            total_points=result['total_points'],
            points_above_threshold=result['points_above_threshold'],
            total_predicted_users=result['total_predicted_users'],
            average_users_per_point=result['average_users_per_point'],
            threshold_value=result['threshold_value'],
            predictions=predictions,
            processing_time_seconds=result['processing_time_seconds']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/bookings/coordinates", response_model=List[BookingCoordinate])
async def get_all_booking_coordinates():
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        coordinates = []
        keys = redis_client.keys("booking-*")

        for key in keys:
            booking_data = redis_client.get(key)
            if booking_data:
                try:
                    parsed_data = json.loads(booking_data)

                    # Extract driver ID
                    driver_detail = parsed_data.get('driver')
                    if not driver_detail:
                        continue

                    driver_id = driver_detail['_id']

                    # Extract coordinates
                    if 'tripAddress' in parsed_data and len(parsed_data['tripAddress']) > 1:
                        trip_address_1 = parsed_data['tripAddress'][1]
                        location = trip_address_1.get('location', {})

                        latitude = location.get('latitude')
                        longitude = location.get('longitude')

                        if latitude is not None and longitude is not None:
                            coordinates.append(BookingCoordinate(
                                latitude=latitude,
                                longitude=longitude,
                                driver_id=driver_id
                            ))

                except json.JSONDecodeError:
                    continue

        return coordinates

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching coordinates: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "prediction_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
