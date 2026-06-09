# Data
DATA_DIR = r"data"    # we should put OMOP CVS file into this folder
OUTPUT_DIR = r"outputs"

# Cohort definition
# Depression-related ancestor concept IDs (SNOMED).
DEPRESSION_SNOMED_CONCEPTS = [
    432883, 433440, 433991, 434911, 435220, 435520,
    438406, 438727, 438998, 440383, 441534, 4025677,
    4049623, 4077577, 4098302, 4141454, 4195572,
    4228802, 4263748, 4282096, 4282316, 4323418,
]

# Antidepressant ancestor concept IDs grouped by class.
ANTIDEPRESSANT_CONCEPTS = {
    'SSRI': [
        797617,   # Citalopram / Sertraline
        715939,   # Escitalopram
        755695,   # Fluoxetine
        751412,   # Fluvoxamine
        722031,   # Paroxetine
    ],
    'SNRI': [
        715259,   # Duloxetine
        743670,   # Venlafaxine
        717607,   # Desvenlafaxine
        43560354, # Levomilnacipran
        19084693, # Milnacipran
    ],
    'Tricyclic': [
        778268,   # Imipramine
        721724,   # Nortriptyline
        710062,   # Amitriptyline
        738156,   # Doxepin
        716968,   # Desipramine
        754270,   # Protriptyline
        705755,   # Trimipramine
        794147,   # Maprotiline
        713109,   # Amoxapine
    ],
    'MAOI': [
        703470,   # Tranylcypromine
        733896,   # Phenelzine
        781705,   # Isocarboxazid
        766209,   # Selegiline
        19010652, # Moclobemide
    ],
    'Atypical': [
        714684,   # Nefazodone
        703547,   # Trazodone
        725131,   # Mirtazapine
        44507700, # Vortioxetine
        40234834, # Vilazodone
        750982,   # Bupropion
        36878783, # Agomelatine
    ],
    'Augmentation': [
        19010886, # Zuclopenthixol
        757688,   # Aripiprazole
        766814,   # Quetiapine
        19124477, # Lithium
        19017241, # Iloperidone
    ],
    'Other': [
        1366610,  # Esketamine
        785649,   # Ketamine
    ],
}

# Minimum days between first database record and depression diagnosis.
COHORT_LOOKBACK_DAYS = 365

# Grace period: patients who started antidepressants up to this many days
TREATMENT_NAIVE_GRACE_DAYS = 30

# Therapy SNOMED ancestor concept IDs
THERAPY_SNOMED_CONCEPTS = {
    'therapy_cbt':   [228557008],  # Cognitive and behavioural therapy (CBT)
    'therapy_ipt':   [443730003],  # Interpersonal psychotherapy (IPT)
    'therapy_group': [76168009],   # Group psychotherapy
}

# Outcome
READMISSION_DAYS = 30

# Feature engineering
TOP_N_DRUGS = 10

# Action rule mining
AR_MIN_STABLE = 1
AR_MIN_FLEXIBLE = 1
AR_MIN_SUPPORT = 10
AR_MIN_CONFIDENCE = 0.6

# Uplift tree
UPLIFT_MAX_DEPTH = 4
UPLIFT_MIN_LEAF = 10
