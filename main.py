import numpy as np
from sklearn.metrics import precision_recall_curve, auc

from algorithms.AlgorithmFactory import AlgorithmFactory

DATASETS_NAMES = [
        {'source': "annthyroid", 'name': "Annthyroid", "real-world-topic": "Healthcare"},
        {'source': "arrhythmia", 'name': "Arrhythmia", "real-world-topic": "Healthcare"},
        {'source': "cover", 'name': "ForestCover", "real-world-topic": "Botany"},
        {'source': "creditcard", 'name': "Creditcard", "real-world-topic": "Finance"},
        {'source': "fault", 'name': "Fault", "real-world-topic": "Physical"},
        {'source': "fraud", 'name': "Fraud", "real-world-topic": "Finance"},
        {'source': "http", 'name': "Http", "real-world-topic": "Web"},
        {'source': "internet_ads", 'name': "InternetAds", "real-world-topic": "Image"},
        {'source': "ionosphere", 'name': "Ionosphere", "real-world-topic": "Oryctognosy"},
        {'source': "landsat", 'name': "Landsat", "real-world-topic": "Astronautics"},
        {'source': "letter", 'name': "Letter", "real-world-topic": "Image"},
        {'source': "lympho", 'name': "Lympho", "real-world-topic": "Healthcare"},
        {'source': "magic_gamma", 'name': "MagicGamma", "real-world-topic": "Physical"},
        {'source': "musk", 'name': "Musk", "real-world-topic": "Chemistry"},
        {'source': "nad", 'name': "Nad", "real-world-topic": "Network"},
        {'source': "optdigits", 'name': "Optdigits", "real-world-topic": "Image"},
        {'source': "page_blocks", 'name': "PageBlocks", "real-world-topic": "Document"},
        {'source': "satimage-2", 'name': "Satimage-2", "real-world-topic": "Astronautics"},
        {'source': "smtp", 'name': "Smtp", "real-world-topic": "Web"},
        {'source': "spam_base", 'name': "SpamBase", "real-world-topic": "Document"},
        {'source': "thyroid", 'name': "Thyroid", "real-world-topic": "Healthcare"},
        {'source': "unsw0", 'name': "Unsw", "real-world-topic": "Network"},
        {'source': "unsw1", 'name': "Unsw1", "real-world-topic": "Network"},
        {'source': "vertebral", 'name': "Vertebral", "real-world-topic": "Biology"},
        {'source': "vowels", 'name': "Vowels", "real-world-topic": "Linguistics"},
        {'source': "waveform", 'name': "Waveform", "real-world-topic": "Physics"},
        {'source': "wbc", 'name': "WBC", "real-world-topic": "Healthcare"},
        {'source': "wdbc", 'name': "WDBC", "real-world-topic": "Healthcare"},
        {'source': "wilt", 'name': "Wilt", "real-world-topic": "Botany"},
        {'source': "yeast", 'name': "Yeast", "real-world-topic": "Biology"}
    ]

algorithms = [
        ("ActualIsolationForest", "AIF", {"trees_number": 100, "samples_per_tree": 256}),
        ("ActualExtendedIsolationForest", "AEIF", {"trees_number": 100, "samples_per_tree": 256}),
        ("ActualProximityIsolationForest", "APIF", {"trees_number": 100, "samples_per_tree": 256})
    ]

for dataset in DATASETS_NAMES:
    filePath = f'datasets/{dataset["source"]}.npz'
    with np.load(filePath) as data:
        xIn = data['X'].astype(np.float64)
        yIn = data['y'].astype(np.float64)

    for algorithm in algorithms:
        model = AlgorithmFactory.create(algorithm)

        model.fit(xIn)
        scores = model.decision_function(xIn)

        precision, recall, thresholds = precision_recall_curve(y_true=yIn, y_score=scores, pos_label=1)
        pr_auc = auc(recall, precision)

        print(f"Dataset={dataset["name"]}; Algorithm={algorithm[0]}; PR AUC={pr_auc*100}[%]")

