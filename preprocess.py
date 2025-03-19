import numpy as np
import pandas as pd

def transform_input(input_data, encoder, scaler, imputer=None, advanced=False, selector=None):
    # Basic information
    car_class = input_data.get("class")
    cylinders = float(input_data.get("cylinders"))
    drive = input_data.get("drive")
    fuel_type = input_data.get("fuel_type")
    transmission = input_data.get("transmission")
    
    # For advanced model
    if advanced:
        # Get additional inputs or use defaults
        displacement = float(input_data.get("displacement", 2.0))
        city_mpg = float(input_data.get("city_mpg", 25.0))  # Default value
        highway_mpg = float(input_data.get("highway_mpg", 30.0))  # Default value
        make = input_data.get("make", "other")  # Default to other
        year = int(input_data.get("year", 2020))
        
        # Engine efficiency metrics
        power_per_cylinder = displacement / cylinders
        
        # Interaction terms
        drive_map = {'fwd': 1, 'rwd': 2, '4wd': 3, 'awd': 4}
        drive_numeric = drive_map.get(drive, 1)
        cylinders_drive = cylinders * drive_numeric
        
        # Fuel type as numeric
        fuel_map = {'gas': 1, 'diesel': 2, 'electricity': 3}
        fuel_numeric = fuel_map.get(fuel_type, 1)
        
        # Transmission as numeric
        trans_map = {'a': 1, 'm': 2}
        trans_numeric = trans_map.get(transmission, 1)
        
        # MPG difference and efficiency ratios
        mpg_diff = highway_mpg - city_mpg
        mpg_per_liter = (city_mpg + highway_mpg) / 2 / displacement
        
        # Polynomial features
        cylinders_squared = cylinders ** 2
        displacement_squared = displacement ** 2
        
        # Car age feature
        car_age = 2025 - year
        
        # Top makes - handle this feature
        top_makes = ['ford', 'toyota', 'chevrolet', 'honda', 'bmw', 'mercedes-benz', 'hyundai', 'mazda', 'audi', 'nissan']
        make_top = make if make in top_makes else 'other'
        
        # Prepare categorical input
        categorical_input = pd.DataFrame({
            "class": [car_class],
            "drive": [drive],
            "fuel_type": [fuel_type],
            "transmission": [transmission],
            "make_top": [make_top]
        })
        
        # Transform categorical features
        encoded_input = encoder.transform(categorical_input)
        
        # Prepare numerical features
        numerical_features = np.array([[
            cylinders, displacement, power_per_cylinder, cylinders_drive, 
            cylinders_squared, displacement_squared, car_age, mpg_diff, 
            mpg_per_liter, city_mpg, highway_mpg, fuel_numeric, trans_numeric
        ]])
        
        # Apply imputer to numerical features if provided
        if imputer is not None:
            numerical_features = imputer.transform(numerical_features)
        
        # Combine features
        final_input = np.hstack((numerical_features[0], encoded_input[0]))
        
        # Scale features
        final_input_scaled = scaler.transform([final_input])
        
        # Apply feature selection if provided
        if selector is not None:
            final_input_scaled = selector.transform(final_input_scaled)
            
        return final_input_scaled[0]
    
    # For original models
    else:
        # Create DataFrame with the same column names as used during training
        categorical_input = pd.DataFrame({
            "class": [car_class],
            "drive": [drive],
            "fuel_type": [fuel_type],
            "transmission": [transmission]
        })
        
        # Transform categorical features
        encoded_input = encoder.transform(categorical_input)

        # Apply imputer to numerical features if provided
        numerical_features = np.array([[cylinders]])
        if imputer is not None:
            numerical_features = imputer.transform(numerical_features)
        
        final_input = np.concatenate((numerical_features[0], encoded_input[0]))
        final_input_scaled = scaler.transform([final_input])

        return final_input_scaled[0]