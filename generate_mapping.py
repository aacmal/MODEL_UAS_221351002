import json
import pickle
import numpy as np

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def generate_android_mappings():
    """
    Generate mapping files in formats suitable for Android
    """
    
    try:
        # Load the model info
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        print("📱 GENERATING ANDROID MAPPING FILES...")
        print("="*50)
        
        # Method 1: JSON file (easiest for Android)
        # Convert numpy types to Python native types
        android_mappings = {
            "target_classes": convert_numpy_types(model_info['target_classes']),
            "gender_mapping": convert_numpy_types(model_info['gender_mapping']),
            "occupation_mapping": convert_numpy_types(model_info['occupation_mapping']),
            "bmi_mapping": convert_numpy_types(model_info['bmi_mapping']),
            "feature_order": convert_numpy_types(model_info['feature_order'])
        }
        
        # Save as JSON
        with open('android_mappings.json', 'w') as f:
            json.dump(android_mappings, f, indent=2)
        
        print("✅ JSON mapping saved to: android_mappings.json")
        
        # Method 2: Generate Java constants file
        java_constants = generate_java_constants(model_info)
        with open('ModelConstants.java', 'w') as f:
            f.write(java_constants)
        
        print("✅ Java constants saved to: ModelConstants.java")
        
        # Method 3: Generate Kotlin constants file
        kotlin_constants = generate_kotlin_constants(model_info)
        with open('ModelConstants.kt', 'w') as f:
            f.write(kotlin_constants)
        
        print("✅ Kotlin constants saved to: ModelConstants.kt")
        
        # Print the mappings for reference
        print_mappings_info(model_info)
        
    except FileNotFoundError:
        print("❌ model_info.pkl not found. Run the model training first.")
    except Exception as e:
        print(f"❌ Error: {e}")

def generate_java_constants(model_info):
    """Generate Java constants class"""
    
    java_code = """package com.yourapp.ml;

import java.util.HashMap;
import java.util.Map;
import java.util.Arrays;
import java.util.List;

public class ModelConstants {
    
    // Target classes (output labels)
    public static final List<String> TARGET_CLASSES = Arrays.asList(
"""
    
    # Add target classes
    for i, class_name in enumerate(model_info['target_classes']):
        comma = "," if i < len(model_info['target_classes']) - 1 else ""
        java_code += f'        "{class_name}"{comma}\n'
    
    java_code += """    );
    
    // Gender mapping
    public static final Map<String, Integer> GENDER_MAPPING = new HashMap<String, Integer>() {{
"""
    
    # Add gender mapping
    for gender, value in model_info['gender_mapping'].items():
        java_code += f'        put("{gender}", {value});\n'
    
    java_code += """    }};
    
    // Occupation mapping
    public static final Map<String, Integer> OCCUPATION_MAPPING = new HashMap<String, Integer>() {{
"""
    
    # Add occupation mapping
    for occupation, value in model_info['occupation_mapping'].items():
        java_code += f'        put("{occupation}", {value});\n'
    
    java_code += """    }};
    
    // BMI Category mapping
    public static final Map<String, Integer> BMI_MAPPING = new HashMap<String, Integer>() {{
"""
    
    # Add BMI mapping
    for bmi, value in model_info['bmi_mapping'].items():
        java_code += f'        put("{bmi}", {value});\n'
    
    java_code += """    }};
    
    // Feature order (for reference)
    public static final List<String> FEATURE_ORDER = Arrays.asList(
"""
    
    # Add feature order
    for i, feature in enumerate(model_info['feature_order']):
        comma = "," if i < len(model_info['feature_order']) - 1 else ""
        java_code += f'        "{feature}"{comma}\n'
    
    java_code += """    );
    
    // Helper methods
    public static int getGenderCode(String gender) {
        return GENDER_MAPPING.getOrDefault(gender, 0);
    }
    
    public static int getOccupationCode(String occupation) {
        return OCCUPATION_MAPPING.getOrDefault(occupation, 0);
    }
    
    public static int getBMICode(String bmiCategory) {
        return BMI_MAPPING.getOrDefault(bmiCategory, 0);
    }
    
    public static String getPredictedClass(int classIndex) {
        if (classIndex >= 0 && classIndex < TARGET_CLASSES.size()) {
            return TARGET_CLASSES.get(classIndex);
        }
        return "Unknown";
    }
}
"""
    
    return java_code

def generate_kotlin_constants(model_info):
    """Generate Kotlin constants object"""
    
    kotlin_code = """package com.yourapp.ml

object ModelConstants {
    
    // Target classes (output labels)
    val TARGET_CLASSES = listOf(
"""
    
    # Add target classes
    for i, class_name in enumerate(model_info['target_classes']):
        comma = "," if i < len(model_info['target_classes']) - 1 else ""
        kotlin_code += f'        "{class_name}"{comma}\n'
    
    kotlin_code += """    )
    
    // Gender mapping
    val GENDER_MAPPING = mapOf(
"""
    
    # Add gender mapping
    gender_items = list(model_info['gender_mapping'].items())
    for i, (gender, value) in enumerate(gender_items):
        comma = "," if i < len(gender_items) - 1 else ""
        kotlin_code += f'        "{gender}" to {value}{comma}\n'
    
    kotlin_code += """    )
    
    // Occupation mapping
    val OCCUPATION_MAPPING = mapOf(
"""
    
    # Add occupation mapping
    occupation_items = list(model_info['occupation_mapping'].items())
    for i, (occupation, value) in enumerate(occupation_items):
        comma = "," if i < len(occupation_items) - 1 else ""
        kotlin_code += f'        "{occupation}" to {value}{comma}\n'
    
    kotlin_code += """    )
    
    // BMI Category mapping
    val BMI_MAPPING = mapOf(
"""
    
    # Add BMI mapping
    bmi_items = list(model_info['bmi_mapping'].items())
    for i, (bmi, value) in enumerate(bmi_items):
        comma = "," if i < len(bmi_items) - 1 else ""
        kotlin_code += f'        "{bmi}" to {value}{comma}\n'
    
    kotlin_code += """    )
    
    // Feature order (for reference)
    val FEATURE_ORDER = listOf(
"""
    
    # Add feature order
    for i, feature in enumerate(model_info['feature_order']):
        comma = "," if i < len(model_info['feature_order']) - 1 else ""
        kotlin_code += f'        "{feature}"{comma}\n'
    
    kotlin_code += """    )
    
    // Helper functions
    fun getGenderCode(gender: String): Int = GENDER_MAPPING[gender] ?: 0
    
    fun getOccupationCode(occupation: String): Int = OCCUPATION_MAPPING[occupation] ?: 0
    
    fun getBMICode(bmiCategory: String): Int = BMI_MAPPING[bmiCategory] ?: 0
    
    fun getPredictedClass(classIndex: Int): String {
        return if (classIndex in TARGET_CLASSES.indices) {
            TARGET_CLASSES[classIndex]
        } else {
            "Unknown"
        }
    }
}
"""
    
    return kotlin_code

def print_mappings_info(model_info):
    """Print mapping information for reference"""
    
    print(f"\n📋 MAPPING INFORMATION")
    print("="*50)
    
    print(f"\n🎯 Target Classes ({len(model_info['target_classes'])}):")
    for i, class_name in enumerate(model_info['target_classes']):
        print(f"  {i}: {class_name}")
    
    print(f"\n👤 Gender Mapping ({len(model_info['gender_mapping'])}):")
    for gender, code in model_info['gender_mapping'].items():
        print(f"  {gender}: {code}")
    
    print(f"\n💼 Occupation Mapping ({len(model_info['occupation_mapping'])}):")
    for occupation, code in model_info['occupation_mapping'].items():
        print(f"  {occupation}: {code}")
    
    print(f"\n⚖️ BMI Category Mapping ({len(model_info['bmi_mapping'])}):")
    for bmi, code in model_info['bmi_mapping'].items():
        print(f"  {bmi}: {code}")

def print_android_usage_examples():
    """Print usage examples for Android"""
    
    print(f"\n💻 ANDROID USAGE EXAMPLES")
    print("="*50)
    
    print("\n🔵 Java Example:")
    print("""
// Get encoded values
int genderCode = ModelConstants.getGenderCode("Male");          // Returns 1
int occupationCode = ModelConstants.getOccupationCode("Doctor"); // Returns specific code
int bmiCode = ModelConstants.getBMICode("Normal");              // Returns specific code

// Prepare input array
float[][] input = new float[1][12];
input[0][9] = genderCode;     // Gender encoded
input[0][10] = occupationCode; // Occupation encoded  
input[0][11] = bmiCode;       // BMI encoded

// Get prediction result
int predictedIndex = argMax(output[0]);
String predictedClass = ModelConstants.getPredictedClass(predictedIndex);
""")
    
    print("\n🟢 Kotlin Example:")
    print("""
// Get encoded values
val genderCode = ModelConstants.getGenderCode("Male")          // Returns 1
val occupationCode = ModelConstants.getOccupationCode("Doctor") // Returns specific code
val bmiCode = ModelConstants.getBMICode("Normal")              // Returns specific code

// Prepare input array
val input = Array(1) { FloatArray(12) }
input[0][9] = genderCode.toFloat()     // Gender encoded
input[0][10] = occupationCode.toFloat() // Occupation encoded
input[0][11] = bmiCode.toFloat()       // BMI encoded

// Get prediction result
val predictedIndex = output[0].indexOfMax()
val predictedClass = ModelConstants.getPredictedClass(predictedIndex)
""")

if __name__ == "__main__":
    generate_android_mappings()
    print_android_usage_examples()