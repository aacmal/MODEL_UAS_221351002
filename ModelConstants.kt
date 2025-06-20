package com.yourapp.ml

object ModelConstants {
    
    // Target classes (output labels)
    val TARGET_CLASSES = listOf(
        "Insomnia",
        "Sleep Apnea",
        "nan"
    )
    
    // Gender mapping
    val GENDER_MAPPING = mapOf(
        "Female" to 0,
        "Male" to 1
    )
    
    // Occupation mapping
    val OCCUPATION_MAPPING = mapOf(
        "Accountant" to 0,
        "Doctor" to 1,
        "Engineer" to 2,
        "Lawyer" to 3,
        "Manager" to 4,
        "Nurse" to 5,
        "Sales Representative" to 6,
        "Salesperson" to 7,
        "Scientist" to 8,
        "Software Engineer" to 9,
        "Teacher" to 10
    )
    
    // BMI Category mapping
    val BMI_MAPPING = mapOf(
        "Normal" to 0,
        "Normal Weight" to 1,
        "Obese" to 2,
        "Overweight" to 3
    )
    
    // Feature order (for reference)
    val FEATURE_ORDER = listOf(
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
    )
    
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
