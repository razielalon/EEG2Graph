import scipy.io as sio
import numpy as np
import pandas as pd
import spacy
import os
import glob

# Load spaCy for Subject/Object detection
print("Loading spaCy NLP model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: Model not found. Please run: python -m spacy download en_core_web_sm")
    exit()

class ZuCoProcessor:
    def __init__(self, data_dir="."):
        """
        Args:
            data_dir: Folder containing the .mat files (default is current folder)
        """
        self.data_dir = data_dir
        self.dataset = [] 

    def load_mat_file(self, file_path):
        """
        Loads a single .mat file and handles the specific ZuCo 1.0 structure.
        """
        try:
            # squeeze_me=True removes unnecessary array dimensions
            # struct_as_record=False loads structs as Python objects (obj.field)
            mat = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
            
            if 'sentenceData' in mat:
                return mat['sentenceData']
            else:
                # Fallback: check other keys just in case
                keys = [k for k in mat.keys() if not k.startswith('_')]
                if len(keys) > 0:
                    return mat[keys[0]]
                return None
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None

    def extract_graph_nodes(self, text):
        """
        Uses spaCy to identify entities. 
        Heuristic: First Entity = Subject, Last Entity = Object.
        """
        doc = nlp(text)
        ents = [e.text for e in doc.ents]
        
        if len(ents) >= 2:
            return ents[0], ents[-1] # Subject, Object
        elif len(ents) == 1:
            return ents[0], "Unknown"
        else:
            return "Unknown", "Unknown"

    def extract_eeg_features(self, word_data):
        """
        Extracts EEG Gamma band (mean_g1) for every word.
        Returns a Numpy array: (Sequence_Length, 105)
        """
        eeg_sequence = []
        
        # ZuCo 1.0 sometimes wraps the word array differently, we normalize it here
        if not isinstance(word_data, (np.ndarray, list)):
            word_data = [word_data]

        for word in word_data:
            # We focus on Gamma band (mean_g1) as it relates to binding concepts
            if hasattr(word, 'mean_g1'):
                features = word.mean_g1
                
                # Handle cases where features might be a single number instead of an array
                if np.isscalar(features):
                    features = np.array([features])
                    
                # Handle NaNs (missing data due to blink removal etc.)
                if np.isnan(features).any():
                    features = np.nan_to_num(features)
                
                # Ensure it's not empty
                if len(features) > 0:
                    eeg_sequence.append(features)
                else:
                    # Padding for words with no fixation (approx 105 channels)
                    eeg_sequence.append(np.zeros(105))
            else:
                eeg_sequence.append(np.zeros(105)) 
                
        return np.array(eeg_sequence)

    def process_all(self):
        # Look for all ZuCo 1.0 TSR files
        files = glob.glob(os.path.join(self.data_dir, "*_TSR.mat"))
        
        if not files:
            print("No '_TSR.mat' files found! Please check your directory.")
            return

        print(f"Found {len(files)} files. Starting processing...")

        total_sentences = 0
        
        for file_path in files:
            print(f"Processing {os.path.basename(file_path)}...")
            data = self.load_mat_file(file_path)
            
            if data is None:
                continue

            # Iterate through every sentence in the file
            for item in data:
                try:
                    # 1. Text & Label
                    text = item.content
                    relation = getattr(item, 'relation_type', 'UNKNOWN')
                    
                    # 2. Graph Nodes (Subject/Object)
                    subj, obj = self.extract_graph_nodes(text)
                    
                    # 3. EEG Signal
                    # Check if 'word' field exists and is not empty
                    if hasattr(item, 'word'):
                        eeg_matrix = self.extract_eeg_features(item.word)
                    else:
                        continue

                    # 4. Save
                    self.dataset.append({
                        "file_source": os.path.basename(file_path),
                        "text": text,
                        "eeg_features": eeg_matrix,      # Input (X)
                        "target_relation": relation,     # Output 1
                        "target_subject": subj,          # Output 2
                        "target_object": obj             # Output 3
                    })
                    total_sentences += 1
                    
                except AttributeError:
                    continue

        print(f"\nDone! Processed {total_sentences} sentences.")

    def save(self, output_file="zuco_graph_dataset.pkl"):
        if not self.dataset:
            print("Dataset is empty. Nothing to save.")
            return

        df = pd.DataFrame(self.dataset)
        df.to_pickle(output_file)
        print(f"Saved dataset to {output_file}")
        
        # Show a preview
        print("\n" + "="*30)
        print("DATA PREVIEW")
        print("="*30)
        sample = df.iloc[0]
        print(f"Text: {sample['text']}")
        print(f"Graph: ({sample['target_subject']}) --[{sample['target_relation']}]--> ({sample['target_object']})")
        print(f"EEG Shape: {sample['eeg_features'].shape} (Words x Electrodes)")

if __name__ == "__main__":
    # Initialize logic in current directory
    processor = ZuCoProcessor(data_dir=".")
    
    # Run
    processor.process_all()
    processor.save()