# Skin Cancer Detection System

## Overview

This project introduces a novel approach for the early detection of various types of skin cancer through a Distributed Skin Cancer Care Model (DSCCM). Skin cancer is one of the most prevalent types of cancer globally, with three main forms: basal cell carcinoma, squamous cell carcinoma, and melanoma. The DSCCM utilizes a hub-and-spoke architecture with five distinct levels (L1 to L5) and incorporates Artificial Intelligence (AI) interventions for efficient and early diagnosis.

Skin cancer is often caused by prolonged exposure to ultraviolet (UV) radiation from the sun or artificial sources. Early detection is crucial, as it significantly improves the chances of successful treatment. The DSCCM aims to revolutionize the detection and treatment process, providing a systematic and technology-driven approach to skin cancer care.

## Implementation Levels on the Ground

### Level 1 (L1) - Apex Centers

At the highest level, L1 comprises well-established skin cancer treatment centers recognized for their excellence. These apex centers offer a comprehensive range of services, including advanced treatments and specialized care for complex cases. Patients visiting L1 centers can access a full spectrum of skin cancer care, from diagnosis to cutting-edge therapeutic interventions.

### Level 2 (L2) - Comprehensive Care Units (Hubs)

L2 represents dedicated Comprehensive Care Units strategically located to serve as hubs for skin cancer care. These units handle less complex cases than L1 but focus on providing frequent diagnostic and therapeutic procedures. L2 acts as an intermediary level, ensuring that patients receive specialized care without the need for immediate referral to apex centers.

### Level 3 (L3) - Diagnosis Centers

L3 centers play a crucial role in providing in-depth diagnostic services, utilizing invasive methods and offering day care services. These centers are equipped to conduct various diagnostic procedures for common types of skin cancers. Patients at L3 may undergo surgeries for specific cases, and the center acts as a vital component in the multi-tiered approach to skin cancer care.

### Level 4 (L4) - Screening and Early Diagnosis Facilities

Primary facilities at L4 are designed for screening, early diagnosis, and initial treatment options. These facilities, located across accessible areas, cater to a broad spectrum of skin cancer cases. L4 provides chemotherapy and palliative care, ensuring that patients receive timely interventions for their conditions.

### Level 5 (L5) - Digital Layer for Early Detection

L5 represents the digital layer where users actively participate in the early detection process by uploading photos of suspected skin areas through a user-friendly mobile application. The aim is to empower individuals and increase awareness about skin health. The process involves:

1. **User Photo Upload:**
   - Users capture and upload clear photos of suspected skin areas through the dedicated mobile application.

2. **AI Instruction and Analysis:**
   - Upon photo submission, the AI model provides immediate instructions and guidance to users regarding the uploaded images.

3. **Real-time Feedback:**
   - Users receive real-time feedback on the potential risk of skin cancer based on the AI analysis.

#### Output Categories

The AI model categorizes results into different possibilities, each guiding users on the appropriate next steps:

1. **Low Risk (Category 1):**
   - *Instruction:* No immediate concern detected.
   - *Next Steps:* Continue regular skin monitoring and follow general skin health practices.

2. **Moderate Risk (Category 2):**
   - *Instruction:* Some irregularities detected; cautious monitoring advised.
   - *Next Steps:* Schedule a visit to a local screening facility for a professional evaluation.

3. **High Risk (Category 3):**
   - *Instruction:* Significant irregularities detected; professional evaluation recommended.
   - *Next Steps:* Urgent consultation with a dermatologist or visit a Level 4 or Level 3 facility for further diagnostics.

4. **Potential Cancer (Category 4):**
   - *Instruction:* High likelihood of skin cancer; immediate professional intervention required.
   - *Next Steps:* Urgent consultation with a dermatologist or visit a Level 4 or Level 3 facility for thorough diagnosis and treatment planning.

5. **Uncertain (Category 5):**
   - *Instruction:* Inconclusive results; seek professional evaluation for clarity.
   - *Next Steps:* Schedule a visit to a local screening facility or consult with a dermatologist for further assessment.

The user-centric approach in Level 5 aims to promote early detection, educate users about their skin health, and guide them toward appropriate actions based on the AI analysis.

![Screenshot 2023-11-19 at 3 47 08 PM](https://github.com/Shobhit-Singhh/Skin_cancer/assets/117563572/287e360b-a6e0-41a6-b172-5239d9975baf)

## AI Intervention Process

1. **Photo Capture:** Images of suspected skin areas are captured at Level 5.

2. **Cloud-Based AI Model:** Photos are uploaded to a cloud-based CNN model for analysis.

3. **Early Detection:** The AI model assesses the images for potential skin cancer, assigning a probability score.

4. **Possibility Categories:** Results are categorized into different possibilities, determining the need for further diagnostics.

5. **Referral for Diagnosis:** Cases with a high probability of skin cancer are referred to appropriate levels for thorough diagnosis and treatment.

## Benefits

- **Early Detection:** Leveraging AI at the initial level enhances the chances of early skin cancer detection.
  
- **Standardized Care:** The model ensures standardized care processes across all levels.

- **Efficient Referral System:** Intelligent referrals streamline the diagnostic process, optimizing resource utilization.

## How to Use

To utilize this system, follow these steps:

1. **Capture Images:** Take clear photos of suspected skin areas.
  
2. **Upload to the System:** Upload the images to the designated cloud-based system.

3. **Receive AI Assessment:** Await the AI model's assessment, indicating the likelihood of skin cancer.

4. **Follow Referral Recommendations:** Act on the referral recommendations for further diagnosis if necessary.
