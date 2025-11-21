import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_simulated_data(num_samples=1000):
    """
    Creates a dummy dataset simulating content with human and AI-generated data.
    """
    # Simulate data types and quality scores
    data_type = np.random.choice(['human', 'ai'], size=num_samples, p=[0.3, 0.7])
    # Quality is high for human (Anchor), mixed/lower for AI
    quality_score = np.where(data_type == 'human', 
                             np.random.uniform(0.8, 1.0, num_samples), 
                             np.random.uniform(0.1, 0.7, num_samples))
    
    # Simulate the actual content (simplified for prototype)
    content = [f"This is a {dtype} sample with quality {q:.2f}. Data Index: {i}" 
               for i, (dtype, q) in enumerate(zip(data_type, quality_score))]
    
    df = pd.DataFrame({'content': content, 'source': data_type, 'quality': quality_score})
    return df

def apply_blending_strategy(df, anchor_ratio=0.25, quality_threshold=0.5):
    """
    Implements the core Model Collapse solution: Strategic Data Blending.
    """
    
    # --- 1. Separate Human Anchor Data ---
    human_data = df[df['source'] == 'human']
    
    # Split the human data to reserve the 'anchor set'
    if len(human_data) > 0:
        # We discard the larger 'train' set and keep the 'test' set as the anchor
        _, anchor_set = train_test_split(
            human_data, test_size=anchor_ratio, random_state=42
        )
    else:
        anchor_set = pd.DataFrame()

    # --- 2. Filter and Select Synthetic Data ---
    non_anchor_df = df.drop(anchor_set.index, errors='ignore')
    
    # Filter the non-anchor data for only high-quality AI content
    high_quality_ai = non_anchor_df[
        (non_anchor_df['source'] == 'ai') & 
        (non_anchor_df['quality'] >= quality_threshold)
    ]
    
    # --- 3. Final Blend ---
    # Concatenate the guaranteed anchor set and the filtered AI data.
    final_training_set = pd.concat([anchor_set, high_quality_ai])
    # Shuffle the final set and reset index for training readiness
    final_training_set = final_training_set.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return final_training_set, anchor_set, high_quality_ai