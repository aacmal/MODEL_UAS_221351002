import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle


def create_and_train_model():
    """
    Load data, preprocess, train model, and return everything needed for TFLite conversion
    """
    
    # Load data
    df = pd.read_csv('./Sleep_health_and_lifestyle_dataset.csv')
    
    # Preprocessing data
    data = df.copy()
    
    # Parse Blood Pressure
    bp_split = data['Blood Pressure'].str.split('/', expand=True)
    data['Systolic_BP'] = bp_split[0].astype(float)
    data['Diastolic_BP'] = bp_split[1].astype(float)
    
    # Encode categorical variables
    gender_encoder = LabelEncoder()
    occupation_encoder = LabelEncoder()
    bmi_encoder = LabelEncoder()
    
    data['Gender_Encoded'] = gender_encoder.fit_transform(data['Gender'])
    data['Occupation_Encoded'] = occupation_encoder.fit_transform(data['Occupation'])
    data['BMI_Encoded'] = bmi_encoder.fit_transform(data['BMI Category'])
    
    # Target variable
    sleep_disorder_encoder = LabelEncoder()
    data['Sleep Disorder'] = data['Sleep Disorder'].replace('', 'None')
    data['Sleep_Disorder_Encoded'] = sleep_disorder_encoder.fit_transform(data['Sleep Disorder'])
    
    # Features
    feature_columns = [
        'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
        'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP',
        'Gender_Encoded', 'Occupation_Encoded', 'BMI_Encoded'
    ]
    
    X = data[feature_columns].values.astype(np.float32)
    y = data['Sleep_Disorder_Encoded'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model with built-in preprocessing
    inputs = tf.keras.Input(shape=(12,), name='input_features')
    
    # Normalization layer - will be adapted to training data
    normalization = tf.keras.layers.Normalization(name='normalization')
    x = normalization(inputs)
    
    # Neural network layers
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout_1')(x)
    x = tf.keras.layers.Dense(32, activation='relu', name='dense_2')(x)
    x = tf.keras.layers.Dropout(0.2, name='dropout_2')(x)
    x = tf.keras.layers.Dense(16, activation='relu', name='dense_3')(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax', name='output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='sleep_disorder_model')
    
    # Adapt normalization layer to training data
    normalization.adapt(X_train)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train,  # Use raw data, normalization is built-in
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Create encoders info for Android
    encoders_info = {
        'target_classes': sleep_disorder_encoder.classes_.tolist(),
        'gender_mapping': dict(zip(gender_encoder.classes_, gender_encoder.transform(gender_encoder.classes_))),
        'occupation_mapping': dict(zip(occupation_encoder.classes_, occupation_encoder.transform(occupation_encoder.classes_))),
        'bmi_mapping': dict(zip(bmi_encoder.classes_, bmi_encoder.transform(bmi_encoder.classes_))),
        'feature_order': feature_columns
    }
    
    return model, encoders_info


def convert_to_tflite():
    """
    Convert trained model to TensorFlow Lite
    """
    
    print("Creating and training model...")
    model, encoders_info = create_and_train_model()
    
    # Convert to TFLite
    print("Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for quantization
    def representative_data_gen():
        # Generate some sample data for quantization
        for _ in range(100):
            # Create realistic sample data
            sample = np.array([[
                np.random.uniform(20, 60),      # Age
                np.random.uniform(5, 10),       # Sleep Duration
                np.random.uniform(1, 10),       # Quality of Sleep
                np.random.uniform(10, 100),     # Physical Activity Level
                np.random.uniform(1, 10),       # Stress Level
                np.random.uniform(60, 100),     # Heart Rate
                np.random.uniform(1000, 15000), # Daily Steps
                np.random.uniform(90, 180),     # Systolic BP
                np.random.uniform(60, 120),     # Diastolic BP
                np.random.randint(0, 2),        # Gender (0 or 1)
                np.random.randint(0, len(encoders_info['occupation_mapping'])), # Occupation
                np.random.randint(0, len(encoders_info['bmi_mapping']))  # BMI
            ]], dtype=np.float32)
            yield [sample]
    
    converter.representative_dataset = representative_data_gen
    
    # Convert the model
    try:
        tflite_model = converter.convert()
        
        # Save TFLite model
        model_filename = 'sleep_disorder_model.tflite'
        with open(model_filename, 'wb') as f:
            f.write(tflite_model)
        
        # Save encoders information
        with open('model_info.pkl', 'wb') as f:
            pickle.dump(encoders_info, f)
        
        print(f"✅ Model successfully converted to: {model_filename}")
        print(f"✅ Model info saved to: model_info.pkl")
        
        return tflite_model, encoders_info
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return None, None


def test_tflite_model():
    """
    Test the TFLite model with sample data
    """
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path='sleep_disorder_model.tflite')
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\n=== MODEL INFO ===")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type: {output_details[0]['dtype']}")
        
        # Load model info
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        # Test with sample data
        # Sample: 28-year-old male, software engineer, normal BMI, moderate sleep issues
        test_input = np.array([[
            28.0,    # Age
            6.5,     # Sleep Duration
            6.0,     # Quality of Sleep
            40.0,    # Physical Activity Level
            5.0,     # Stress Level
            75.0,    # Heart Rate
            5000.0,  # Daily Steps
            120.0,   # Systolic BP
            80.0,    # Diastolic BP
            0.0,     # Gender (Male)
            model_info['occupation_mapping'].get('Software Engineer', 0),  # Occupation
            model_info['bmi_mapping'].get('Normal', 0)  # BMI
        ]], dtype=np.float32)
        
        # Set input
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Interpret results
        predicted_class = np.argmax(output_data[0])
        predicted_label = model_info['target_classes'][predicted_class]
        confidence = output_data[0][predicted_class]
        
        print(f"\n=== TEST RESULTS ===")
        print(f"Input sample: 28-year-old male software engineer")
        print(f"Predicted class: {predicted_label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"All probabilities:")
        for i, prob in enumerate(output_data[0]):
            print(f"  {model_info['target_classes'][i]}: {prob:.4f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False


def print_android_instructions():
    """
    Print instructions for Android implementation
    """
    
    try:
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
    except:
        print("❌ Model info not found. Run conversion first.")
        return
    
    print("\n" + "="*50)
    print("📱 ANDROID IMPLEMENTATION GUIDE")
    print("="*50)
    
    print("\n📁 FILES NEEDED:")
    print("1. sleep_disorder_model.tflite")
    print("2. model_info.pkl (for reference)")
    
    print("\n📊 INPUT FORMAT:")
    print("Single float array with 12 elements in this exact order:")
    for i, feature in enumerate(model_info['feature_order']):
        print(f"{i:2d}. {feature}")
    
    print(f"\n🏷️ CATEGORICAL MAPPINGS:")
    print(f"Gender: {model_info['gender_mapping']}")
    print(f"Occupation: {model_info['occupation_mapping']}")
    print(f"BMI Category: {model_info['bmi_mapping']}")
    
    print(f"\n📤 OUTPUT:")
    print(f"Array of 3 probabilities for: {model_info['target_classes']}")
    
    print("\n💻 ANDROID CODE EXAMPLE:")
    print("""
// Load model
Interpreter tflite = new Interpreter(loadModelFile());

// Prepare input (12 features)
float[][] input = new float[1][12];
input[0][0] = 28.0f;  // Age
input[0][1] = 6.5f;   // Sleep Duration
// ... fill all 12 features

// Prepare output
float[][] output = new float[1][3];

// Run inference
tflite.run(input, output);

// Get results
float[] probabilities = output[0];
int predictedClass = argMax(probabilities);
""")
    
    print("\n✅ The model includes preprocessing, so just provide raw values!")


def main():
    """
    Main function - focused on TFLite creation for Android
    """
    
    print("🚀 SLEEP DISORDER MODEL - ANDROID TFLITE CONVERTER")
    print("="*55)
    
    # Convert model to TFLite
    tflite_model, encoders_info = convert_to_tflite()
    
    if tflite_model is not None:
        # Test the model
        print("\n🧪 Testing TFLite model...")
        if test_tflite_model():
            # Print Android instructions
            print_android_instructions()
        else:
            print("❌ Model testing failed")
    else:
        print("❌ Model conversion failed")


if __name__ == "__main__":
    main()