
from datasets import load_dataset
from . import config
import re
from sklearn.model_selection import train_test_split

def load_and_filter_dataset():
    """Load dataset and filter samples with single label"""
    print("Loading dataset")
    ds = load_dataset("UniverseTBD/arxiv-abstracts-large", cache_dir=config.CACHE_DIR)

    samples = []
    for s in ds['train']:
        if len(s['categories'].split(' ')) != 1:
            continue 

        cur_category = s['categories'].strip().split('.')[0]

        if cur_category not in config.CATEGORIES_TO_SELECT:
            continue

        samples.append(s)

        if len(samples) >= config.NUM_SAMPLES:
            break
    
    return samples

def preprocess_text(text):
    """Clean and preprocess text"""
    text = text.strip().replace("\n", " ")
    text = re.sub(r'[^\w\s]', '', text) # Remove everything except letters, numbers, underscores, and spaces
    text = re.sub(r'\d+', '', text) # Remove 1 or more digits
    text = re.sub(r'\s+', ' ', text).strip() # "hello    world   " → "hello world"
    text = text.lower()
    return text

def preprocess_samples(samples):
    """Preprocess all samples"""
    print("Preprocessing samples")
    preprocessed = []

    for s in samples:
        abstarct = preprocess_text(s['abstract'])
        category = s['categories'].strip().split('.')[0]

        preprocessed.append({
            'text': abstarct,
            'label': category
        })
    
    return preprocessed

def create_label_mapping(preprocessed_samples):
    """Create label to ID mappings"""
    labels = sorted(set([s['label'] for s in preprocessed_samples]))
    label_to_id = {label: id for id, label in enumerate(labels)}
    id_to_label = {id: label for id, label in enumerate(labels)}

    for label, id in label_to_id.items():
        print(f"{label} --> {id}")
    
    return label_to_id, id_to_label

def prepare_data():
    """Main function to load and prepare data"""
    samples = load_and_filter_dataset()
    preprocessed_samples = preprocess_samples(samples)
    label_to_id, id_to_label = create_label_mapping(preprocessed_samples)

    X = [s['text'] for s in preprocessed_samples]
    y = [label_to_id[s['label']] for s in preprocessed_samples]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test, label_to_id, id_to_label







