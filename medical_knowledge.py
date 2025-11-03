"""
Medical Knowledge Base for RAG.

This module contains medical information that will be used by the RAG system
to provide more accurate and context-aware health recommendations.
"""

MEDICAL_DOCUMENTS = [
    # Headache Information
    """
    Headache: Common causes include tension, migraines, cluster headaches, and sinus infections.
    Tension headaches are the most common type and feel like a tight band around the head.
    Migraines are severe, throbbing headaches often accompanied by nausea and light sensitivity.
    Red flags requiring immediate attention: sudden severe headache (thunderclap), headache with fever and stiff neck,
    headache after head injury, or headache with confusion or vision changes.
    Recommended tests: CT scan or MRI for severe or persistent cases, blood tests to rule out infections.
    """,
    
    # Fever Information
    """
    Fever: Body temperature above 100.4°F (38°C). Common causes include viral infections (flu, cold),
    bacterial infections, heat exhaustion, and inflammatory conditions.
    Mild fever (100.4-102°F) can often be managed at home with rest and fluids.
    High fever (above 103°F) requires medical attention, especially if persistent.
    Seek immediate care for: fever above 104°F, fever lasting more than 3 days, fever with severe headache,
    stiff neck, confusion, or difficulty breathing.
    Recommended tests: Complete blood count (CBC), blood culture, urinalysis, chest X-ray if respiratory symptoms present.
    """,
    
    # Chest Pain Information
    """
    Chest Pain: Can range from minor muscle strain to life-threatening conditions.
    Cardiac causes: Angina, heart attack (myocardial infarction), pericarditis.
    Warning signs of heart attack: crushing chest pressure, pain radiating to arm/jaw/back, shortness of breath,
    cold sweats, nausea.
    Non-cardiac causes: Gastroesophageal reflux (GERD), costochondritis, panic attacks, pulmonary embolism.
    Always seek immediate emergency care for: sudden severe chest pain, chest pain with breathing difficulty,
    chest pain with fainting or dizziness.
    Recommended tests: ECG (electrocardiogram), cardiac enzymes (troponin), chest X-ray, stress test, coronary angiography.
    """,
    
    # Shortness of Breath Information
    """
    Shortness of Breath (Dyspnea): Difficulty breathing or feeling unable to get enough air.
    Common causes: Asthma, COPD, pneumonia, heart failure, anxiety, anemia, pulmonary embolism.
    Acute onset requires immediate evaluation. Chronic dyspnea needs thorough workup.
    Seek emergency care for: sudden severe shortness of breath, blue lips or fingernails,
    chest pain with breathing, confusion or altered mental status.
    Recommended tests: Pulse oximetry, chest X-ray, pulmonary function tests, arterial blood gas,
    CT pulmonary angiography if pulmonary embolism suspected.
    """,
    
    # Abdominal Pain Information
    """
    Abdominal Pain: Location and characteristics help determine cause.
    Right upper quadrant: Gallbladder disease, hepatitis, liver problems.
    Right lower quadrant: Appendicitis, ovarian issues in women.
    Left lower quadrant: Diverticulitis, kidney stones.
    Epigastric pain: Gastritis, peptic ulcer, pancreatitis.
    Red flags: Severe sudden pain, pain with fever and vomiting, rigid abdomen, pain with bloody stools.
    Recommended tests: Abdominal ultrasound, CT scan of abdomen, blood tests (amylase, lipase for pancreatitis),
    complete blood count, urinalysis.
    """,
    
    # Cough Information
    """
    Cough: Can be acute (less than 3 weeks) or chronic (more than 8 weeks).
    Acute cough: Usually viral upper respiratory infection, bronchitis, pneumonia.
    Chronic cough: Postnasal drip, asthma, GERD, chronic bronchitis, ACE inhibitor side effect.
    Seek medical attention for: Cough with blood, severe shortness of breath, high fever,
    cough lasting more than 3 weeks, chest pain with coughing.
    Recommended tests: Chest X-ray, spirometry for suspected asthma, sputum culture if infection suspected,
    CT chest for chronic cough evaluation.
    """,
    
    # Fatigue Information
    """
    Fatigue: Persistent tiredness not relieved by rest.
    Common causes: Anemia, thyroid disorders, diabetes, depression, sleep apnea, chronic fatigue syndrome,
    infections (including COVID-19), heart disease, cancer.
    Important to evaluate: Duration, associated symptoms, impact on daily activities, sleep quality.
    Seek evaluation for: Unexplained fatigue lasting more than 2 weeks, fatigue with weight loss,
    fatigue with fever or night sweats, severe fatigue limiting daily activities.
    Recommended tests: Complete blood count, thyroid function tests (TSH, T4), comprehensive metabolic panel,
    vitamin B12 and iron levels, sleep study if sleep apnea suspected.
    """,
    
    # Nausea and Vomiting Information
    """
    Nausea and Vomiting: Can result from gastrointestinal, neurological, or metabolic causes.
    Common causes: Gastroenteritis (stomach flu), food poisoning, medication side effects, pregnancy,
    motion sickness, migraines, appendicitis.
    Seek immediate care for: Vomiting blood, severe abdominal pain, signs of dehydration,
    severe headache with vomiting, inability to keep any fluids down for 24 hours.
    Recommended tests: Electrolyte panel, pregnancy test if applicable, abdominal imaging,
    head CT if neurological symptoms present.
    """,
    
    # Dizziness Information
    """
    Dizziness: Can be vertigo (spinning sensation) or lightheadedness.
    Vertigo causes: Benign positional vertigo (BPPV), Meniere's disease, vestibular neuritis, labyrinthitis.
    Lightheadedness causes: Low blood pressure, dehydration, anemia, heart rhythm problems, anxiety.
    Seek emergency care for: Dizziness with chest pain, severe headache, difficulty speaking,
    weakness or numbness, double vision, loss of consciousness.
    Recommended tests: Blood pressure measurements, ECG, blood tests (CBC, glucose, electrolytes),
    hearing test, vestibular function tests, MRI brain if central cause suspected.
    """,
    
    # Joint Pain Information
    """
    Joint Pain (Arthralgia): Pain in one or more joints.
    Common causes: Osteoarthritis, rheumatoid arthritis, gout, lupus, injury, infection (septic arthritis).
    Osteoarthritis: Wear-and-tear arthritis, worse with activity, common in older adults.
    Rheumatoid arthritis: Autoimmune, morning stiffness, symmetric joint involvement.
    Gout: Sudden severe pain, often big toe, caused by uric acid crystals.
    Seek prompt care for: Hot, red, swollen joint with fever (possible infection), severe pain preventing movement.
    Recommended tests: X-rays, blood tests (rheumatoid factor, anti-CCP, ESR, CRP, uric acid),
    joint fluid analysis if infection or gout suspected, MRI for detailed evaluation.
    """,
    
    # Back Pain Information
    """
    Back Pain: Very common, often muscular but can indicate serious conditions.
    Common causes: Muscle strain, herniated disc, spinal stenosis, osteoarthritis, poor posture.
    Red flags (require immediate evaluation): Pain with fever, bowel/bladder dysfunction,
    progressive leg weakness, saddle anesthesia, pain after trauma, unexplained weight loss.
    Most back pain improves with conservative treatment within 4-6 weeks.
    Recommended tests: X-rays for trauma or red flags, MRI for suspected disc herniation or spinal stenosis,
    CT scan, bone scan if malignancy suspected.
    """,
    
    # Skin Rash Information
    """
    Skin Rash: Varied presentations requiring careful evaluation.
    Common causes: Eczema, psoriasis, allergic reactions, viral infections (measles, chickenpox),
    bacterial infections (cellulitis), fungal infections, drug reactions.
    Seek urgent care for: Rash with fever, rapidly spreading rash, painful rash, rash with breathing difficulty
    (possible anaphylaxis), purple non-blanching rash (possible meningococcemia).
    Recommended tests: Skin biopsy for unclear diagnosis, allergy testing, blood tests if systemic disease suspected,
    cultures for suspected infection.
    """,
    
    # Diabetes Information
    """
    Diabetes: Chronic condition affecting blood sugar regulation.
    Type 1: Autoimmune destruction of insulin-producing cells, requires insulin therapy.
    Type 2: Insulin resistance and relative insulin deficiency, managed with lifestyle and medications.
    Symptoms: Increased thirst, frequent urination, fatigue, blurred vision, slow wound healing.
    Complications: Cardiovascular disease, kidney disease, neuropathy, retinopathy, foot problems.
    Recommended tests: Fasting blood glucose, HbA1c (glycated hemoglobin), oral glucose tolerance test,
    lipid profile, kidney function tests, urine albumin.
    """,
    
    # Hypertension Information
    """
    Hypertension (High Blood Pressure): Blood pressure consistently above 130/80 mmHg.
    Often asymptomatic ("silent killer") but can cause headaches, nosebleeds, shortness of breath.
    Complications: Heart attack, stroke, heart failure, kidney disease, vision loss.
    Management: Lifestyle modifications (diet, exercise, weight loss, reduced sodium), medications if needed.
    Seek emergency care for: Blood pressure above 180/120, severe headache, chest pain, vision changes,
    difficulty breathing (hypertensive emergency).
    Recommended tests: Blood pressure monitoring, ECG, echocardiogram, kidney function tests,
    urinalysis, lipid profile.
    """,
    
    # Depression and Anxiety Information
    """
    Depression: Persistent sadness, loss of interest, affecting daily functioning.
    Symptoms: Depressed mood, sleep changes, appetite changes, fatigue, difficulty concentrating,
    feelings of worthlessness, thoughts of death or suicide.
    Anxiety: Excessive worry and fear interfering with daily life.
    Symptoms: Restlessness, fatigue, difficulty concentrating, irritability, muscle tension, sleep disturbance.
    Seek immediate help for: Suicidal thoughts or plans, self-harm, severe panic attacks.
    Treatment: Psychotherapy, medications (antidepressants, anxiolytics), lifestyle modifications.
    Recommended evaluation: Mental health screening questionnaires, thyroid tests to rule out medical causes.
    """
]
