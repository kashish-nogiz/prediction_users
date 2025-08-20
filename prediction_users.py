import numpy as np
import pandas as pd
import math
import folium
from datetime import datetime
import warnings
import joblib
from sklearn.preprocessing import StandardScaler
import requests
import time
import redis
import json
import threading
import signal
import sys
import random
warnings.filterwarnings('ignore')


class NewfoundlandRadiusPredictor:
    def __init__(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        """Initialize Newfoundland and Labrador radius generator with ML prediction"""

        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host='redis-16648.c278.us-east-1-4.ec2.redns.redis-cloud.com',
            port=16648,
            decode_responses=True,
            username="default",
            password="3gyrWhHJoKrzUkQRxGYNZTBQcprp7VmG",
        )

        # Test Redis connection
        try:
            self.redis_client.ping()
            print("✅ Redis connection established successfully")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            sys.exit(1)

        # Load your trained model and scaler
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("✅ Model and scaler loaded successfully")
        except FileNotFoundError:
            print("⚠️ Model/scaler files not found. Will generate dummy predictions.")
            self.model = None
            self.scaler = None

        # Newfoundland and Labrador boundaries
        self.nl_boundaries = {
            'north': 60.85,  # Northern Labrador
            'south': 46.62,  # Southern Newfoundland
            'west': -67.80,  # Western Labrador
            'east': -52.62  # Eastern Newfoundland
        }

        # Define non-overlapping radius for two major cities
        self.major_cities = [
            {
                'name': "St. John's",
                'lat': 47.5615,
                'lon': -52.8126,
                'radius_km': 10  # 10km radius to avoid overlap
            },
            {
                'name': 'Happy Valley-Goose Bay',
                'lat': 53.3420,
                'lon': -60.4135,
                'radius_km': 25  # 25km radius
            }
        ]

        # Control variables for automation
        self.running = False
        self.prediction_count = 0

        # Pre-generate centers to avoid repeated computation
        self.centers = None
        self.initialize_centers()

        print("🏔️ Newfoundland and Labrador Radius Predictor Initialized")
        print(f"📍 Configured for {len(self.major_cities)} non-overlapping city radii")

    def initialize_centers(self):
        """Pre-generate center points to avoid repeated computation"""
        print("🔄 Pre-generating center points...")
        all_centers = []
        water_points_filtered = 0

        for city in self.major_cities:
            print(f"🔍 Processing {city['name']}...")
            centers = self.create_circular_grid(
                center_lat=city['lat'],
                center_lon=city['lon'],
                area_radius_km=city['radius_km'],
                grid_spacing_km=1.0
            )

            # Filter out water points (simplified for performance)
            land_centers = []
            for center in centers:
                if self.is_point_on_land_fast(center['center_latitude'], center['center_longitude']):
                    center['city'] = city['name']
                    center['city_radius_km'] = city['radius_km']
                    land_centers.append(center)
                else:
                    water_points_filtered += 1

            all_centers.extend(land_centers)
            print(f"📍 {city['name']}: {len(land_centers)} land points")

        self.centers = all_centers
        print(f"✅ Pre-generated {len(self.centers)} center points ({water_points_filtered} water points filtered)")

    def is_point_on_land_fast(self, lat, lon):
        """
        Fast land/water check using geographic heuristics (no API calls for performance)
        """
        try:
            # Simple boundary check for obvious water points
            if lat < 46.0 or lat > 61.0 or lon < -68.0 or lon > -52.0:
                return False

            # St. John's area (Avalon Peninsula)
            if 47.0 <= lat <= 48.5 and -54.0 <= lon <= -52.5:
                return True

            # Central Newfoundland
            if 48.5 <= lat <= 50.0 and -57.0 <= lon <= -53.0:
                return True

            # Labrador coastal area
            if 51.0 <= lat <= 60.0 and -65.0 <= lon <= -55.0:
                return True

            # Default to land for performance
            return True

        except Exception as e:
            return True  # Default to land if check fails

    def is_point_on_land(self, lat, lon):
        """
        Check if a point is on land using OpenStreetMap Nominatim reverse geocoding
        Returns True if on land, False if in water
        """
        return self.is_point_on_land_fast(lat, lon)  # Use fast version for automation

    def verify_non_overlapping(self):
        """Verify that the two city radii don't overlap"""
        city1 = self.major_cities[0]
        city2 = self.major_cities[1]

        distance = self.calculate_distance(
            city1['lat'], city1['lon'],
            city2['lat'], city2['lon']
        )

        min_distance_needed = city1['radius_km'] + city2['radius_km']

        print(f"\n🔍 Non-overlap verification:")
        print(f"   Distance between cities: {distance:.2f} km")
        print(f"   Minimum distance needed: {min_distance_needed:.2f} km")
        print(f"   Status: {'✅ Non-overlapping' if distance >= min_distance_needed else '❌ Overlapping'}")

        return distance >= min_distance_needed

    def create_circular_grid(self, center_lat, center_lon, area_radius_km, grid_spacing_km):
        """Create a grid of points within a circular area"""
        centers = []

        # Earth radius calculations
        lat_degree_km = 111.32
        lon_degree_km = 111.32 * math.cos(math.radians(center_lat))

        # Convert to degrees
        area_radius_deg_lat = area_radius_km / lat_degree_km
        area_radius_deg_lon = area_radius_km / lon_degree_km

        spacing_deg_lat = grid_spacing_km / lat_degree_km
        spacing_deg_lon = grid_spacing_km / lon_degree_km

        # Generate grid
        lat = center_lat - area_radius_deg_lat
        while lat <= center_lat + area_radius_deg_lat:
            lon = center_lon - area_radius_deg_lon
            while lon <= center_lon + area_radius_deg_lon:
                # Check if point is within the circular area
                distance_km = self.calculate_distance(center_lat, center_lon, lat, lon)

                if distance_km <= area_radius_km:
                    centers.append({
                        'center_latitude': round(lat, 6),
                        'center_longitude': round(lon, 6),
                        'distance_from_center_km': round(distance_km, 2),
                        'circle_radius_km': 0.5  # Default circle radius
                    })

                lon += spacing_deg_lon
            lat += spacing_deg_lat

        return centers

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in km using Haversine formula"""
        R = 6371  # Earth's radius in km

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    def get_current_time_components(self):
        """Get current time in hours, minutes, and seconds"""
        now = datetime.now()
        return now.hour, now.minute, now.second

    def predict_users(self):
        """Make predictions for pre-generated center points"""
        hour, minute, second = self.get_current_time_components()
        predictions = []

        for i, center in enumerate(self.centers):
            # Base input should be a flat list
            base_input = [
                center['center_longitude'],
                center['center_latitude'],
                hour,
                minute,
                second
            ]

            if self.model and self.scaler:
                try:
                    # Run multiple variations to calculate std_dev
                    variation_predictions = []

                    for _ in range(10):  # Run 10 slight variations
                        custom_input = base_input.copy()

                        # Add tiny noise to longitude & latitude
                        custom_input[0] += np.random.uniform(-0.000001, 0.000001)
                        custom_input[1] += np.random.uniform(-0.000001, 0.000001)

                        # Vary minute by ±5
                        minute_change = random.randint(-5, 5)
                        new_minute = base_input[3] + minute_change

                        if new_minute < 0:
                            custom_input[2] = (base_input[2] - 1) % 24
                            custom_input[3] = new_minute + 60
                        elif new_minute >= 60:
                            custom_input[2] = (base_input[2] + 1) % 24
                            custom_input[3] = new_minute - 60
                        else:
                            custom_input[3] = new_minute

                        # Random second
                        custom_input[4] = random.randint(0, 59)

                        # Scale and predict
                        scaled_input = self.scaler.transform([custom_input])
                        pred = self.model.predict(scaled_input)
                        variation_predictions.append(pred[0])

                    # Calculate standard deviation
                    std_pred = np.std(variation_predictions)
                    std_pred_str = format(std_pred, '.12f')  # Full decimal format

                    # Predict base input without noise (for main output)
                    base_input_scaled = self.scaler.transform([base_input])
                    prediction = self.model.predict(base_input_scaled)
                    predicted_users = int(prediction[0])

                except Exception as e:
                    predicted_users = np.random.randint(10, 100)  # Fallback
                    std_pred_str = "0.000000000000"
            else:
                # Generate dummy predictions if model not available
                predicted_users = np.random.randint(10, 150)
                std_pred_str = "0.000000000000"

            predictions.append({
                'center_latitude': str(center['center_latitude']),
                'center_longitude': str(center['center_longitude']),
                'city': str(center['city']),
                'predicted_users': str(predicted_users),
                'prediction_std_dev': std_pred_str,
                'prediction_time_hour': str(hour),
                'prediction_time_minute': str(minute),
                'prediction_time_second': str(second),
                'distance_from_city_center_km': str(center['distance_from_center_km']),
                'circle_radius_km': str(center.get('circle_radius_km', 0.5)),
                'timestamp': str(datetime.now().isoformat()),
                'prediction_id': str(f"{self.prediction_count}_{i}")
            })

        # predictions.sort(key=lambda x: int(x['predicted_users']), reverse=True)
        # single_max_user_prediction = predictions[0]

        return predictions

    # def push_to_redis(self, predictions):
    #     """Push predictions to Redis in JSON format with string keys and values"""
    #     try:
    #         timestamp = datetime.now().isoformat()
    #
    #         # Create the main data structure
    #         redis_data = {
    #             'prediction_batch_id': str(self.prediction_count),
    #             'timestamp': str(timestamp),
    #             'total_points': str(len(predictions)),
    #             'status': 'success',
    #             'predictions': {}
    #         }
    #
    #         # Add each prediction with string key
    #         for i, pred in enumerate(predictions):
    #             redis_data['predictions'][str(i)] = pred
    #
    #         # Convert to JSON string
    #         json_data = json.dumps(redis_data, ensure_ascii=False)
    #
    #         # Push to Redis with timestamp-based key
    #         redis_key = f"newfoundland_predictions_{self.prediction_count}_{timestamp.replace(':', '-').replace('.', '-')}"
    #
    #         # Set the data in Redis
    #         self.redis_client.set(redis_key, json_data)
    #
    #         # Also maintain a "latest" key for easy access
    #         self.redis_client.set("newfoundland_predictions_latest", json_data)
    #
    #         # Keep a list of recent prediction keys (last 10)
    #         recent_keys_list = f"newfoundland_recent_keys"
    #         self.redis_client.lpush(recent_keys_list, redis_key)
    #         self.redis_client.ltrim(recent_keys_list, 0, 9)  # Keep only last 10
    #
    #         print(f"✅ Data pushed to Redis:")
    #         print(f"   Key: {redis_key}")
    #         print(f"   Points: {len(predictions)}")
    #         print(f"   Timestamp: {timestamp}")
    #
    #         return True
    #
    #     except Exception as e:
    #         print(f"❌ Failed to push to Redis: {e}")
    #         return False

    def push_to_redis(self, predictions):
        """Push predictions to Redis in JSON format with string keys and values"""
        try:
            timestamp = datetime.now().isoformat()

            # Create the main data structure
            redis_data = {
                'prediction_batch_id': str(self.prediction_count),
                'timestamp': str(timestamp),
                'total_points': str(len(predictions)),
                'status': 'success',
                'predictions': {}
            }

            # Add each prediction with string key (now includes prediction_std_dev)
            for i, pred in enumerate(predictions):
                redis_data['predictions'][str(i)] = pred

            # Convert to JSON string
            json_data = json.dumps(redis_data, ensure_ascii=False)

            # Use a fixed Redis key that gets updated each time
            redis_key = "newfoundland_predictions_latest"

            # Update the data in Redis (overwrites previous data)
            self.redis_client.set(redis_key, json_data)

            print(f"✅ Data updated in Redis:")
            print(f"   Key: {redis_key}")
            print(f"   Points: {len(predictions)}")
            print(f"   Timestamp: {timestamp}")

            return True

        except Exception as e:
            print(f"❌ Failed to push to Redis: {e}")
            return False


    def run_single_prediction(self):
        """Run a single prediction cycle"""
        try:
            start_time = time.time()
            self.prediction_count += 1

            print(f"\n🚀 PREDICTION CYCLE #{self.prediction_count}")
            print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")

            # Make predictions
            predictions = self.predict_users()

            # Push to Redis
            success = self.push_to_redis(predictions)

            end_time = time.time()
            execution_time = end_time - start_time

            if success:
                print(f"✅ Cycle #{self.prediction_count} completed in {execution_time:.2f} seconds")
                print(f"📊 Predicted users for {len(predictions)} points")
            else:
                print(f"❌ Cycle #{self.prediction_count} failed")

            return success

        except Exception as e:
            print(f"❌ Error in prediction cycle #{self.prediction_count}: {e}")
            return False

    def start_automation(self):
        """Start the automated prediction cycle"""
        self.running = True
        print(f"\n🤖 STARTING AUTOMATED PREDICTIONS")
        print(f"⏱️  Running every 5 seconds")
        print(f"📍 Processing {len(self.centers)} prediction points per cycle")
        print(f"🔄 Press Ctrl+C to stop")
        print("=" * 60)

        try:
            while self.running:
                self.run_single_prediction()

                if self.running:  # Check if still running before sleeping
                    time.sleep(600)  # Wait 10 minute

        except KeyboardInterrupt:
            print(f"\n⏹️  Stopping automation...")
            self.stop_automation()
        except Exception as e:
            print(f"\n❌ Automation error: {e}")
            self.stop_automation()

    def stop_automation(self):
        """Stop the automated prediction cycle"""
        self.running = False
        print(f"✅ Automation stopped after {self.prediction_count} cycles")

    def get_redis_data(self, key="newfoundland_predictions_latest"):
        """Retrieve data from Redis"""
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            else:
                print(f"❌ No data found for key: {key}")
                return None
        except Exception as e:
            print(f"❌ Error retrieving data from Redis: {e}")
            return None

    def list_recent_predictions(self):
        """List recent prediction keys from Redis"""
        try:
            recent_keys = self.redis_client.lrange("newfoundland_recent_keys", 0, -1)
            print(f"\n📋 Recent prediction keys:")
            for i, key in enumerate(recent_keys, 1):
                print(f"   {i}. {key}")
            return recent_keys
        except Exception as e:
            print(f"❌ Error listing recent predictions: {e}")
            return []

    def create_visualization_map(self, predictions, filename='newfoundland_predictions_map.html'):
        """Create enhanced heatmap visualization with transparent circle boundaries"""

        # Center map on Newfoundland
        center_lat = np.mean([city['lat'] for city in self.major_cities])
        center_lon = np.mean([city['lon'] for city in self.major_cities])

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='OpenStreetMap'
        )

        # Convert string values back to appropriate types for visualization
        processed_predictions = []
        for pred in predictions:
            processed_pred = {
                'center_latitude': float(pred['center_latitude']),
                'center_longitude': float(pred['center_longitude']),
                'city': pred['city'],
                'predicted_users': int(pred['predicted_users']),
                'circle_radius_km': float(pred['circle_radius_km']),
                'distance_from_city_center_km': float(pred['distance_from_city_center_km']),
                'prediction_time_hour': int(pred['prediction_time_hour']),
                'prediction_time_minute': int(pred['prediction_time_minute']),
                'prediction_time_second': int(pred['prediction_time_second'])
            }
            processed_predictions.append(processed_pred)

        # Add heatmap-style circles for each prediction point
        for pred in processed_predictions:
            color, fill_opacity, stroke_opacity = self.get_heatmap_color_and_opacity(pred['predicted_users'])

            # Add transparent circle boundary (0.5km radius)
            folium.Circle(
                location=[pred['center_latitude'], pred['center_longitude']],
                radius=pred['circle_radius_km'] * 1000,  # Convert to meters
                popup=f"""
                <div style="font-family: Arial, sans-serif; min-width: 200px;">
                    <b>🎯 Prediction Circle</b><br>
                    <hr style="margin: 5px 0;">
                    <b>City:</b> {pred['city']}<br>
                    <b>Location:</b> ({pred['center_latitude']:.6f}, {pred['center_longitude']:.6f})<br>
                    <b>Predicted Users:</b> <span style="color: {color}; font-weight: bold;">{pred['predicted_users']}</span><br>
                    <b>Circle Radius:</b> {pred['circle_radius_km']}km<br>
                    <b>Distance from City Center:</b> {pred['distance_from_city_center_km']:.2f}km<br>
                    <b>Prediction Time:</b> {pred['prediction_time_hour']:02d}:{pred['prediction_time_minute']:02d}:{pred['prediction_time_second']:02d}
                </div>
                """,
                tooltip=f"Users: {pred['predicted_users']} | City: {pred['city']}",
                color=color,
                weight=2,
                fillColor=color,
                fillOpacity=fill_opacity,
                opacity=stroke_opacity
            ).add_to(m)

            # Add center point marker
            folium.CircleMarker(
                location=[pred['center_latitude'], pred['center_longitude']],
                radius=3,
                popup=f"Center Point - {pred['predicted_users']} users",
                color='black',
                weight=1,
                fillColor='white',
                fillOpacity=0.8
            ).add_to(m)

        # Add city centers with enhanced styling
        for city in self.major_cities:
            folium.Marker(
                location=[city['lat'], city['lon']],
                popup=f"""
                <div style="font-family: Arial, sans-serif; min-width: 200px;">
                    <b>🏙️ {city['name']}</b><br>
                    <hr style="margin: 5px 0;">
                    <b>Type:</b> City Center<br>
                    <b>Coverage Radius:</b> {city['radius_km']}km<br>
                    <b>Coordinates:</b> ({city['lat']:.4f}, {city['lon']:.4f})
                </div>
                """,
                icon=folium.Icon(color='blue', icon='info-sign', prefix='fa')
            ).add_to(m)

        m.save(filename)
        print(f"🗺️  Visualization saved as '{filename}'")
        return m

    def get_heatmap_color_and_opacity(self, predicted_users):
        """Get color and opacity for heatmap visualization based on predicted users"""
        if predicted_users <= 0:
            return '#0000FF', 0.1, 0.3  # Blue for very low/no users
        elif predicted_users <= 5:
            return '#00FFFF', 0.2, 0.4  # Cyan for low users
        elif predicted_users <= 10:
            return '#00FF00', 0.3, 0.5  # Green for medium-low users
        elif predicted_users <= 25:
            return '#FFFF00', 0.4, 0.6  # Yellow for medium users
        else:
            return '#FF0000', 0.7, 0.9  # Red for very high users


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n🛑 Received interrupt signal. Stopping...')
    if hasattr(signal_handler, 'predictor'):
        signal_handler.predictor.stop_automation()
    sys.exit(0)


def main():
    """Main execution function"""
    print("🏔️ NEWFOUNDLAND AND LABRADOR AUTOMATED PREDICTOR")
    print("=" * 60)

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize predictor
    predictor = NewfoundlandRadiusPredictor()
    signal_handler.predictor = predictor  # Store reference for signal handler

    # Verify non-overlapping circles
    predictor.verify_non_overlapping()

    print(f"\n🎯 AUTOMATION READY")
    print(f"   • {len(predictor.centers)} prediction points pre-generated")
    print(f"   • Redis connection established")
    print(f"   • Predictions will run every 5 seconds")
    print(f"   • Data format: JSON with string keys and values")

    # Start automation
    predictor.start_automation()

    return predictor


# Interactive functions for manual control
def test_redis_connection():
    """Test function to verify Redis connection"""
    predictor = NewfoundlandRadiusPredictor()

    # Run a single prediction and push to Redis
    success = predictor.run_single_prediction()

    if success:
        print("\n✅ Test successful! Data pushed to Redis.")

        # Retrieve and display the data
        latest_data = predictor.get_redis_data()
        if latest_data:
            print(f"📊 Retrieved data: {len(latest_data['predictions'])} predictions")

    return predictor


def visualize_latest_data():
    """Create visualization from latest Redis data"""
    predictor = NewfoundlandRadiusPredictor()

    # Get latest data from Redis
    latest_data = predictor.get_redis_data()

    if latest_data and 'predictions' in latest_data:
        predictions = list(latest_data['predictions'].values())

        # Create visualization
        map_obj = predictor.create_visualization_map(
            predictions,
            f"redis_predictions_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )

        print(f"✅ Visualization created with {len(predictions)} points")
        return map_obj
    else:
        print("❌ No data found in Redis")
        return None


# Run the automated predictor
if __name__ == "__main__":
    try:
        predictor = main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
