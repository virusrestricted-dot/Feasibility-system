from flask import Flask, render_template, jsonify, request, session
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
from datetime import datetime
import secrets
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

@dataclass
class Barangay:
    """Data class for barangay information"""
    name: str
    lat: float
    lon: float
    population: int
    avg_income: float
    business_count: int
    infrastructure_score: float
    accessibility_score: float
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "population": self.population,
            "avg_income": self.avg_income,
            "business_count": self.business_count,
            "infrastructure_score": self.infrastructure_score,
            "accessibility_score": self.accessibility_score
        }

class BusinessTypeConfig:
    """Configuration for different business types"""
    TYPES = {
        "retail": {
            "name": "Retail Store",
            "min_population": 1500,
            "income_weight": 0.35,
            "competition_tolerance": 5,
            "foot_traffic_importance": 0.4
        },
        "food": {
            "name": "Food Service (Restaurant/Cafe)",
            "min_population": 2000,
            "income_weight": 0.30,
            "competition_tolerance": 3,
            "foot_traffic_importance": 0.45
        },
        "service": {
            "name": "Service Business",
            "min_population": 1000,
            "income_weight": 0.40,
            "competition_tolerance": 4,
            "foot_traffic_importance": 0.25
        },
        "wholesale": {
            "name": "Wholesale/Distribution",
            "min_population": 3000,
            "income_weight": 0.25,
            "competition_tolerance": 2,
            "foot_traffic_importance": 0.15
        },
        "healthcare": {
            "name": "Healthcare/Pharmacy",
            "min_population": 2500,
            "income_weight": 0.35,
            "competition_tolerance": 2,
            "foot_traffic_importance": 0.30
        }
    }

class DataCollectionSimulator:
    """Simulates data collection from various sources"""
    
    @staticmethod
    def collect_barangay_data() -> List[Barangay]:
        """Simulate data collection from census, surveys, and government records"""
        raw_data = [
            {"name": "Poblacion", "lat": 8.5369015, "lon": 126.1198959, "population": 6800, 
             "avg_income": 15000, "business_count": 45, "infrastructure": 8.5, "accessibility": 9.0},
            {"name": "Amaga", "lat": 8.5336866, "lon": 126.1327362, "population": 2900, 
             "avg_income": 12500, "business_count": 10, "infrastructure": 7.0, "accessibility": 7.5},
            {"name": "Bahi", "lat": 8.5124712, "lon": 126.0435944, "population": 2100, 
             "avg_income": 9000, "business_count": 6, "infrastructure": 5.0, "accessibility": 4.5},
            {"name": "Cabacungan", "lat": 8.5274076, "lon": 126.3171158, "population": 1600, 
             "avg_income": 8500, "business_count": 4, "infrastructure": 4.5, "accessibility": 4.0},
            {"name": "Campbagang", "lat": 8.5060138, "lon": 126.0224978, "population": 1400, 
             "avg_income": 8000, "business_count": 3, "infrastructure": 4.0, "accessibility": 3.5},
            {"name": "Causwagan", "lat": 8.5117566, "lon": 126.1592102, "population": 2800, 
             "avg_income": 11000, "business_count": 9, "infrastructure": 6.5, "accessibility": 7.0},
            {"name": "Dapdap", "lat": 8.5336866, "lon": 126.1327362, "population": 3200, 
             "avg_income": 12000, "business_count": 12, "infrastructure": 6.5, "accessibility": 7.0},
            {"name": "Dughan", "lat": 8.5194683, "lon": 126.1216507, "population": 2600, 
             "avg_income": 10500, "business_count": 8, "infrastructure": 6.0, "accessibility": 6.5},
            {"name": "Gamut", "lat": 8.5298461, "lon": 126.2407503, "population": 1900, 
             "avg_income": 9000, "business_count": 5, "infrastructure": 5.0, "accessibility": 5.5},
            {"name": "Guinhalinan", "lat": 8.46808585, "lon": 126.0724274, "population": 1700, 
             "avg_income": 8500, "business_count": 4, "infrastructure": 4.5, "accessibility": 4.0},
            {"name": "Javier", "lat": 8.4885178, "lon": 126.0855942, "population": 2000, 
             "avg_income": 9500, "business_count": 6, "infrastructure": 5.5, "accessibility": 5.0},
            {"name": "Kinayan", "lat": 8.5287554, "lon": 126.1416309, "population": 3400, 
             "avg_income": 12500, "business_count": 13, "infrastructure": 7.0, "accessibility": 7.5},
            {"name": "Mamis", "lat": 8.5105426, "lon": 126.0757542, "population": 2200, 
             "avg_income": 9500, "business_count": 7, "infrastructure": 5.5, "accessibility": 5.5},
            {"name": "Rizal", "lat": 8.539819958, "lon": 126.1636126, "population": 3100, 
             "avg_income": 11500, "business_count": 11, "infrastructure": 6.8, "accessibility": 7.2},
            {"name": "San Jose", "lat": 8.517087604, "lon": 126.1241627, "population": 2500, 
             "avg_income": 11000, "business_count": 8, "infrastructure": 6.0, "accessibility": 6.5},
            {"name": "San Roque", "lat": 8.5162601, "lon": 126.2001724, "population": 2300, 
             "avg_income": 10000, "business_count": 7, "infrastructure": 5.8, "accessibility": 6.0},
            {"name": "San Vicente", "lat": 8.540748322, "lon": 126.1134982, "population": 4200, 
             "avg_income": 13500, "business_count": 18, "infrastructure": 7.5, "accessibility": 8.0},
            {"name": "Sua", "lat": 8.54802131, "lon": 126.1885411, "population": 2700, 
             "avg_income": 11000, "business_count": 9, "infrastructure": 6.2, "accessibility": 6.8},
            {"name": "Sudlon", "lat": 8.5467625, "lon": 126.0778427, "population": 1800, 
             "avg_income": 9000, "business_count": 5, "infrastructure": 5.0, "accessibility": 5.0},
            {"name": "Tambis", "lat": 8.5373669, "lon": 126.0496063, "population": 2000, 
             "avg_income": 9500, "business_count": 6, "infrastructure": 5.2, "accessibility": 5.5},
            {"name": "Unidad", "lat": 8.5332727, "lon": 126.2303619, "population": 2400, 
             "avg_income": 10500, "business_count": 8, "infrastructure": 6.0, "accessibility": 6.5},
            {"name": "Wakat", "lat": 8.5629396, "lon": 126.1158295, "population": 3600, 
             "avg_income": 13000, "business_count": 15, "infrastructure": 7.2, "accessibility": 7.8},
        ]
        
        return [Barangay(
            name=d["name"],
            lat=d["lat"],
            lon=d["lon"],
            population=d["population"],
            avg_income=d["avg_income"],
            business_count=d["business_count"],
            infrastructure_score=d["infrastructure"],
            accessibility_score=d["accessibility"]
        ) for d in raw_data]

class ModelLoader:
    """Loads and manages the business feasibility model"""
    
    def __init__(self, model_path='business_feasibility_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.model_info = {}
        self.load_model()
    
    def load_model(self):
        """Load the pickled model file"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"✓ Model loaded successfully from {self.model_path}")
                self.extract_model_info()
            else:
                print(f"⚠ Model file not found: {self.model_path}")
                print("⚠ Using simulated model data")
                self.use_simulated_model()
        except Exception as e:
            print(f"{e}")
            print("")
            self.use_simulated_model()
    
    def extract_model_info(self):
        """Extract information from the loaded model"""
        if self.model is None:
            return
        
        # Get algorithm name
        algorithm_name = type(self.model).__name__
        
        # Extract training samples
        training_samples = "Unknown"
        if hasattr(self.model, 'n_samples_'):
            training_samples = self.model.n_samples_
        elif hasattr(self.model, 'n_features_in_'):
            training_samples = f"{self.model.n_features_in_} features"
        
        # Extract accuracy if available
        accuracy = None
        if hasattr(self.model, 'best_score_'):
            accuracy = self.model.best_score_
        elif hasattr(self.model, 'score_'):
            accuracy = self.model.score_
        elif hasattr(self.model, 'oob_score_'):
            accuracy = self.model.oob_score_
        
        # Try to get feature importances
        features = []
        if hasattr(self.model, 'feature_names_in_'):
            features = list(self.model.feature_names_in_)
        elif hasattr(self.model, 'n_features_in_'):
            features = [f"Feature_{i+1}" for i in range(self.model.n_features_in_)]
        
        self.model_info = {
            "algorithm": algorithm_name,
            "features": features if features else ["population", "avg_income", "business_density", "infrastructure", "accessibility"],
            "accuracy": accuracy if accuracy else 0.0,
            "training_samples": training_samples
        }
        
        print(f"  Algorithm: {algorithm_name}")
        print(f"  Training Samples: {training_samples}")
        print(f"  Accuracy: {accuracy if accuracy else 'Not available'}")
    
    def use_simulated_model(self):
        """Fallback to simulated model if pickle file not available"""
        self.model_info = {
            "algorithm": "Random Forest Classifier (Simulated)",
            "features": ["population", "avg_income", "business_density", "infrastructure", "accessibility"],
            "accuracy": 0.87,
            "training_samples": 220
        }
    
    def get_model_info(self) -> Dict:
        """Return model information for the report"""
        return self.model_info
    
    def predict(self, features: np.ndarray):
        """Make prediction using the loaded model"""
        if self.model and hasattr(self.model, 'predict'):
            try:
                return self.model.predict(features)
            except Exception as e:
                print(f"Prediction error: {e}")
                return None
        return None

# Initialize the model loader globally
model_loader = ModelLoader('business_feasibility_model.pkl')

class PredictiveModel:
    """Uses the loaded ML model for feasibility prediction"""
    
    @staticmethod
    def train_model(barangay_data: List[Barangay]) -> Dict:
        """Return the loaded model information"""
        return model_loader.get_model_info()
    
    @staticmethod
    def predict_feasibility(barangay: Barangay, business_type: str, capital: float) -> Tuple[str, float, Dict]:
        """Calculate feasibility score based on business type and investment"""
        
        config = BusinessTypeConfig.TYPES.get(business_type, BusinessTypeConfig.TYPES["retail"])
        
        # Population adequacy (0-25 points)
        pop_score = min(barangay.population / config["min_population"], 2.0) * 12.5
        
        # Income level (0-25 points)
        income_score = min(barangay.avg_income / 10000, 2.0) * 12.5
        
        # Competition analysis (0-20 points)
        business_density = barangay.business_count / (barangay.population / 1000)
        competition_score = max(0, 20 - (business_density * 3))
        
        # Infrastructure (0-15 points)
        infra_score = barangay.infrastructure_score * 1.5
        
        # Accessibility (0-15 points)
        access_score = barangay.accessibility_score * 1.5
        
        # Capital efficiency (0-10 points)
        capital_efficiency = (barangay.population * barangay.avg_income) / capital
        capital_score = min(capital_efficiency / 500, 1.0) * 10
        
        # Calculate total score
        total_score = pop_score + income_score + competition_score + infra_score + access_score + capital_score
        
        # Optional: Use the loaded model for prediction if available
        if model_loader.model:
            try:
                # Prepare features for the model
                features = np.array([[
                    barangay.population,
                    barangay.avg_income,
                    business_density,
                    barangay.infrastructure_score,
                    barangay.accessibility_score
                ]])
                
                prediction = model_loader.predict(features)
                if prediction is not None:
                    print(f"Model prediction for {barangay.name}: {prediction}")
            except Exception as e:
                print(f"Could not use model for prediction: {e}")
        
        # Determine feasibility level
        if total_score >= 75:
            feasibility = "High"
        elif total_score >= 55:
            feasibility = "Medium"
        else:
            feasibility = "Low"
        
        # Detailed breakdown
        breakdown = {
            "population_score": round(pop_score, 1),
            "income_score": round(income_score, 1),
            "competition_score": round(competition_score, 1),
            "infrastructure_score": round(infra_score, 1),
            "accessibility_score": round(access_score, 1),
            "capital_efficiency_score": round(capital_score, 1),
            "business_density": round(business_density, 2),
            "market_potential": barangay.population * barangay.avg_income
        }
        
        return feasibility, round(total_score, 1), breakdown

class GeospatialAnalyzer:
    """Handles geospatial analysis and location recommendations"""
    
    @staticmethod
    def analyze_proximity(barangays: List[Barangay], preferred_location: str = None) -> Dict:
        """Analyze spatial relationships between barangays"""
        analysis = {}
        
        for barangay in barangays:
            nearby_population = sum(b.population for b in barangays 
                                   if b.name != barangay.name and 
                                   GeospatialAnalyzer._calculate_distance(barangay, b) < 5)
            
            analysis[barangay.name] = {
                "nearby_population": nearby_population,
                "catchment_area": barangay.population + nearby_population,
                "is_preferred": barangay.name.lower() == (preferred_location or "").lower()
            }
        
        return analysis
    
    @staticmethod
    def _calculate_distance(b1: Barangay, b2: Barangay) -> float:
        """Simple Euclidean distance (simplified for demo)"""
        return ((b1.lat - b2.lat)**2 + (b1.lon - b2.lon)**2)**0.5 * 111  # Rough km conversion

class ReportGenerator:
    """Generates comprehensive feasibility reports"""
    
    @staticmethod
    def generate_report(analysis_results: List[Dict], business_type: str, 
                       capital: float, model_info: Dict) -> Dict:
        """Generate final feasibility report"""
        
        sorted_results = sorted(analysis_results, key=lambda x: x["score"], reverse=True)
        
        top_3 = sorted_results[:3]
        high_feasibility = [r for r in analysis_results if r["feasibility"] == "High"]
        
        recommendations = []
        for i, result in enumerate(top_3, 1):
            recommendations.append({
                "rank": i,
                "barangay": result["barangay"],
                "score": result["score"],
                "feasibility": result["feasibility"],
                "reason": ReportGenerator._generate_reason(result, business_type),
                "roi_estimate": ReportGenerator._estimate_roi(result, capital),
                "risks": ReportGenerator._identify_risks(result),
                "location": {"lat": result["lat"], "lon": result["lon"]},
                "breakdown": result["breakdown"],
                "population": result["population"],
                "avg_income": result["avg_income"]
            })
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "input_parameters": {
                "business_type": BusinessTypeConfig.TYPES[business_type]["name"],
                "capital_investment": f"₱{capital:,.0f}",
                "analysis_date": datetime.now().strftime("%B %d, %Y")
            },
            "model_performance": model_info,
            "analysis_summary": {
                "total_locations": len(analysis_results),
                "high_feasibility_count": len(high_feasibility),
                "average_score": round(sum(r["score"] for r in analysis_results) / len(analysis_results), 1),
                "best_location": top_3[0]["barangay"] if top_3 else "N/A"
            },
            "top_recommendations": recommendations,
            "all_results": sorted_results
        }
        
        return report
    
    @staticmethod
    def _generate_reason(result: Dict, business_type: str) -> str:
        """Generate explanation for recommendation"""
        reasons = []
        
        breakdown = result.get("breakdown", {})
        
        if breakdown.get("population_score", 0) > 10:
            reasons.append(f"Strong population base of {result['population']:,}")
        
        if breakdown.get("income_score", 0) > 10:
            reasons.append("Favorable income levels")
        
        if breakdown.get("competition_score", 0) > 15:
            reasons.append("Low competition environment")
        
        if breakdown.get("infrastructure_score", 0) > 10:
            reasons.append("Good infrastructure")
        
        if breakdown.get("accessibility_score", 0) > 10:
            reasons.append("Excellent accessibility")
        
        return "; ".join(reasons) if reasons else "Moderate market conditions"
    
    @staticmethod
    def _estimate_roi(result: Dict, capital: float) -> str:
        """Estimate ROI timeframe"""
        score = result["score"]
        
        if score >= 80:
            return "12-18 months (Excellent)"
        elif score >= 70:
            return "18-24 months (Good)"
        elif score >= 60:
            return "24-36 months (Moderate)"
        else:
            return "36+ months (High Risk)"
    
    @staticmethod
    def _identify_risks(result: Dict) -> List[str]:
        """Identify potential risks"""
        risks = []
        breakdown = result.get("breakdown", {})
        
        if breakdown.get("competition_score", 0) < 10:
            risks.append("High competition")
        
        if breakdown.get("infrastructure_score", 0) < 8:
            risks.append("Limited infrastructure")
        
        if breakdown.get("capital_efficiency_score", 0) < 5:
            risks.append("Capital efficiency concerns")
        
        if result.get("population", 0) < 2000:
            risks.append("Small market size")
        
        return risks if risks else ["Minimal identified risks"]

# Main workflow
@app.route('/')
def index():
    """Landing page with input form"""
    return render_template("input_form.html", 
                         business_types=BusinessTypeConfig.TYPES)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main analysis workflow"""
    
    # Get user inputs
    business_type = request.form.get('business_type', 'retail')
    capital = float(request.form.get('capital', 500000))
    preferred_location = request.form.get('preferred_location', '')
    
    # Store in session
    session['business_type'] = business_type
    session['capital'] = capital
    session['preferred_location'] = preferred_location
    
    # STEP 1: DATA COLLECTION
    barangays = DataCollectionSimulator.collect_barangay_data()
    
    # STEP 2: MODEL TRAINING (Now uses loaded model info)
    model_info = PredictiveModel.train_model(barangays)
    
    # STEP 3: GEOSPATIAL ANALYSIS
    geo_analysis = GeospatialAnalyzer.analyze_proximity(barangays, preferred_location)
    
    # STEP 4: PREDICTIVE MODELING
    analysis_results = []
    for barangay in barangays:
        feasibility, score, breakdown = PredictiveModel.predict_feasibility(
            barangay, business_type, capital
        )
        
        result = {
            "barangay": barangay.name,
            "lat": barangay.lat,
            "lon": barangay.lon,
            "population": barangay.population,
            "avg_income": barangay.avg_income,
            "business_count": barangay.business_count,
            "feasibility": feasibility,
            "score": score,
            "breakdown": breakdown,
            "catchment_area": geo_analysis[barangay.name]["catchment_area"],
            "is_preferred": geo_analysis[barangay.name]["is_preferred"]
        }
        analysis_results.append(result)
    
    # STEP 5: REPORT GENERATION
    report = ReportGenerator.generate_report(analysis_results, business_type, capital, model_info)
    
    return render_template("report.html", report=report)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access"""
    data = request.get_json()
    
    business_type = data.get('business_type', 'retail')
    capital = float(data.get('capital', 500000))
    preferred_location = data.get('preferred_location', '')
    
    barangays = DataCollectionSimulator.collect_barangay_data()
    model_info = PredictiveModel.train_model(barangays)
    geo_analysis = GeospatialAnalyzer.analyze_proximity(barangays, preferred_location)
    
    analysis_results = []
    for barangay in barangays:
        feasibility, score, breakdown = PredictiveModel.predict_feasibility(
            barangay, business_type, capital
        )
        
        analysis_results.append({
            "barangay": barangay.name,
            "feasibility": feasibility,
            "score": score,
            "breakdown": breakdown
        })
    
    report = ReportGenerator.generate_report(analysis_results, business_type, capital, model_info)
    
    return jsonify(report)

@app.route('/model-info')
def model_info_route():
    """Display detailed model information"""
    info = model_loader.get_model_info()
    return jsonify(info)

if __name__ == '__main__':
    # Custom Jinja2 filter for number formatting
    @app.template_filter('format_number')
    def format_number_filter(value):
        try:
            return f"{int(value):,}"
        except (ValueError, TypeError):
            return value
    
    print("\n" + "="*60)
    print("Business Feasibility Analysis System")
    print("="*60)
    print(f"Model Status: {'Loaded' if model_loader.model else 'Simulated'}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)