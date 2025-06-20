package com.yourapp.ml;

import java.util.HashMap;
import java.util.Map;
import java.util.Arrays;
import java.util.List;

public class ModelConstants {
    
    // Target classes (output labels)
    public static final List<String> TARGET_CLASSES = Arrays.asList(
        "Insomnia",
        "Sleep Apnea",
        "nan"
    );
    
    // Gender mapping
    public static final Map<String, Integer> GENDER_MAPPING = new HashMap<String, Integer>() {{
        put("Female", 0);
        put("Male", 1);
    }};
    
    // Occupation mapping
    public static final Map<String, Integer> OCCUPATION_MAPPING = new HashMap<String, Integer>() {{
        put("Accountant", 0);
        put("Doctor", 1);
        put("Engineer", 2);
        put("Lawyer", 3);
        put("Manager", 4);
        put("Nurse", 5);
        put("Sales Representative", 6);
        put("Salesperson", 7);
        put("Scientist", 8);
        put("Software Engineer", 9);
        put("Teacher", 10);
    }};
    
    // BMI Category mapping
    public static final Map<String, Integer> BMI_MAPPING = new HashMap<String, Integer>() {{
        put("Normal", 0);
        put("Normal Weight", 1);
        put("Obese", 2);
        put("Overweight", 3);
    }};
    
    // Feature order (for reference)
    public static final List<String> FEATURE_ORDER = Arrays.asList(
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "Systolic_BP",
        "Diastolic_BP",
        "Gender_Encoded",
        "Occupation_Encoded",
        "BMI_Encoded"
    );
    
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
